"""
mp2mujoco.py  –  Unitree G1 edition
=====================================
Pipeline: MediaPipe World Landmarks → Unitree G1 MuJoCo Joint Angles (23 DOF)

G1 additions vs H1
───────────────────
  • ankle_roll  (×2)  – lateral ankle tilt, estimated from foot-plane orientation
  • wrist_roll  (×2)  – forearm supination/pronation, estimated from hand-plane
  • waist gains roll + pitch on top of yaw  (3-DOF waist)
  → total DOF: 1(waist_yaw) + 6(L-leg) + 6(R-leg) + 5(L-arm) + 5(R-arm) = 23

Coordinate conventions
────────────────────────
  MediaPipe world :  x→right,   y→up,      z→toward-camera   (right-hand)
  MuJoCo / G1    :  x→forward, y→left,    z→up               (right-hand)
  Rotation used  :  v_mj = R_MP2MJ @ v_mp
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1.  MEDIAPIPE LANDMARK INDICES
# ──────────────────────────────────────────────────────────────────────────────

class MP:
    """MediaPipe BlazePose 33-landmark indices."""
    NOSE             = 0
    LEFT_SHOULDER    = 11;  RIGHT_SHOULDER   = 12
    LEFT_ELBOW       = 13;  RIGHT_ELBOW      = 14
    LEFT_WRIST       = 15;  RIGHT_WRIST      = 16
    LEFT_PINKY       = 17;  RIGHT_PINKY      = 18
    LEFT_INDEX       = 19;  RIGHT_INDEX      = 20
    LEFT_THUMB       = 21;  RIGHT_THUMB      = 22
    LEFT_HIP         = 23;  RIGHT_HIP        = 24
    LEFT_KNEE        = 25;  RIGHT_KNEE       = 26
    LEFT_ANKLE       = 27;  RIGHT_ANKLE      = 28
    LEFT_HEEL        = 29;  RIGHT_HEEL       = 30
    LEFT_FOOT_INDEX  = 31;  RIGHT_FOOT_INDEX = 32


# ──────────────────────────────────────────────────────────────────────────────
# 2.  UNITREE G1 JOINT SPECIFICATION  (23 DOF)
#     Limits from official Unitree G1 URDF / MuJoCo Menagerie g1.xml
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class JointSpec:
    name: str
    lo:   float   # lower limit (rad)
    hi:   float   # upper limit (rad)

G1_JOINTS: list[JointSpec] = [

    # ── Waist (3 DOF) ──────────────────────────────────────────────────────
    JointSpec("waist_yaw",             -2.618,  2.618),
    JointSpec("waist_roll",            -0.523,  0.523),
    JointSpec("waist_pitch",           -0.785,  0.785),

    # ── Left Leg (6 DOF) ───────────────────────────────────────────────────
    JointSpec("left_hip_pitch",        -1.047,  2.094),
    JointSpec("left_hip_roll",         -0.524,  2.967),
    JointSpec("left_hip_yaw",          -2.758,  2.758),
    JointSpec("left_knee",             -0.087,  2.880),
    JointSpec("left_ankle_pitch",      -0.873,  0.524),
    JointSpec("left_ankle_roll",       -0.262,  0.262),

    # ── Right Leg (6 DOF) ──────────────────────────────────────────────────
    JointSpec("right_hip_pitch",       -1.047,  2.094),
    JointSpec("right_hip_roll",        -2.967,  0.524),
    JointSpec("right_hip_yaw",         -2.758,  2.758),
    JointSpec("right_knee",            -0.087,  2.880),
    JointSpec("right_ankle_pitch",     -0.873,  0.524),
    JointSpec("right_ankle_roll",      -0.262,  0.262),

    # ── Left Arm (5 DOF) ───────────────────────────────────────────────────
    JointSpec("left_shoulder_pitch",   -3.089,  2.670),
    JointSpec("left_shoulder_roll",    -1.588,  2.252),
    JointSpec("left_shoulder_yaw",     -2.618,  2.618),
    JointSpec("left_elbow",            -1.047,  2.094),
    JointSpec("left_wrist_roll",       -1.972,  1.972),

    # ── Right Arm (5 DOF) ──────────────────────────────────────────────────
    JointSpec("right_shoulder_pitch",  -3.089,  2.670),
    JointSpec("right_shoulder_roll",   -2.252,  1.588),
    JointSpec("right_shoulder_yaw",    -2.618,  2.618),
    JointSpec("right_elbow",           -1.047,  2.094),
    JointSpec("right_wrist_roll",      -1.972,  1.972),
]

JOINT_NAMES = [j.name for j in G1_JOINTS]
DOF         = len(G1_JOINTS)   # 23


# ──────────────────────────────────────────────────────────────────────────────
# 3.  OUTPUT FRAME
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class G1Frame:
    timestamp:     float
    angles:        np.ndarray   # (23,) radians
    confidence:    float        # mean landmark visibility [0, 1]
    raw_landmarks: np.ndarray   # (33, 3) MediaPipe world coords


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MATH HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros_like(v)

def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle between two 3-vectors (rad)."""
    c = np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0)
    return float(np.arccos(c))

def _signed_angle(v1: np.ndarray, v2: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle from v1 → v2 around `axis` (right-hand rule)."""
    cross = np.cross(v1, v2)
    angle = np.arctan2(np.linalg.norm(cross), np.dot(v1, v2))
    return float(-angle if np.dot(cross, axis) < 0 else angle)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  COORDINATE FRAME ALIGNMENT
#     v_mujoco = R_MP2MJ @ v_mediapipe
# ──────────────────────────────────────────────────────────────────────────────

_R_MP2MJ = np.array([
    [ 0,  0, -1],   # mj-x ←  -mp-z  (forward = depth into scene)
    [-1,  0,  0],   # mj-y ←  -mp-x  (left)
    [ 0,  1,  0],   # mj-z ←   mp-y  (up)
], dtype=float)

def _p(lms, idx: int) -> np.ndarray:
    """Landmark → MuJoCo-frame position vector."""
    lm = lms[idx]
    return _R_MP2MJ @ np.array([lm.x, lm.y, lm.z])


# ──────────────────────────────────────────────────────────────────────────────
# 6.  PELVIS / TRUNK FRAME  (shared across hip + waist)
# ──────────────────────────────────────────────────────────────────────────────

def _pelvis_frame(lms):
    """
    Returns (right, fwd, up) orthonormal basis of the pelvis in MuJoCo world.
    right = from left-hip to right-hip
    fwd   = pelvis forward (cross of right × world_up, projected)
    up    = completes right-hand frame
    """
    lh = _p(lms, MP.LEFT_HIP)
    rh = _p(lms, MP.RIGHT_HIP)
    right = _unit(rh - lh)
    world_up = np.array([0., 0., 1.])
    fwd = _unit(np.cross(right, world_up))
    up  = _unit(np.cross(fwd, right))       # ≈ world_up corrected for tilt
    return right, fwd, up


def _shoulder_mid(lms) -> np.ndarray:
    return (_p(lms, MP.LEFT_SHOULDER) + _p(lms, MP.RIGHT_SHOULDER)) * 0.5

def _hip_mid(lms) -> np.ndarray:
    return (_p(lms, MP.LEFT_HIP) + _p(lms, MP.RIGHT_HIP)) * 0.5


# ──────────────────────────────────────────────────────────────────────────────
# 7.  ANGLE ESTIMATORS
# ──────────────────────────────────────────────────────────────────────────────

# ── 7a. Waist ─────────────────────────────────────────────────────────────────

def _waist_angles(lms):
    """
    Returns (yaw, roll, pitch) of the torso relative to the pelvis.

    Strategy
    ────────
    Pelvis frame  – defined by hip line (right axis) + world up
    Torso frame   – defined by shoulder line (right axis) + shoulder-to-hip vector
    We decompose the rotation between them into yaw / roll / pitch components.
    """
    p_right, p_fwd, p_up = _pelvis_frame(lms)

    ls  = _p(lms, MP.LEFT_SHOULDER)
    rs  = _p(lms, MP.RIGHT_SHOULDER)
    t_right = _unit(rs - ls)                      # torso right axis

    world_up = np.array([0., 0., 1.])
    t_fwd    = _unit(np.cross(t_right, world_up))
    t_up     = _unit(np.cross(t_fwd, t_right))

    # Torso spine vector (hip-mid → shoulder-mid)
    spine = _unit(_shoulder_mid(lms) - _hip_mid(lms))

    # Yaw: rotation of torso-right around world-up relative to pelvis-right
    yaw   = _signed_angle(
        np.array([p_right[0], p_right[1], 0.]),
        np.array([t_right[0], t_right[1], 0.]),
        world_up
    )

    # Roll: lateral lean – angle of spine away from p_up in coronal plane
    # Project spine onto (p_right, p_up) plane
    roll  = float(np.arcsin(np.clip(np.dot(spine, p_right), -1, 1)))

    # Pitch: forward lean – angle of spine away from p_up in sagittal plane
    pitch = float(np.arcsin(np.clip(-np.dot(spine, p_fwd),  -1, 1)))

    return yaw, roll, pitch


# ── 7b. Hip ───────────────────────────────────────────────────────────────────

def _hip_angles(lms, side: str):
    """
    Returns (pitch, roll, yaw) for one hip in the pelvis frame.
    G1 URDF order: hip_pitch, hip_roll, hip_yaw.
    """
    hip_idx  = MP.LEFT_HIP   if side == "left" else MP.RIGHT_HIP
    knee_idx = MP.LEFT_KNEE  if side == "left" else MP.RIGHT_KNEE
    sign     = 1.0            if side == "left" else -1.0

    p_right, p_fwd, p_up = _pelvis_frame(lms)

    hip  = _p(lms, hip_idx)
    knee = _p(lms, knee_idx)
    thigh = _unit(knee - hip)          # points downward in neutral

    # Pitch = flexion / extension  (forward tilt of thigh)
    # Positive pitch = hip flexion (thigh forward)
    pitch = float(np.arcsin(np.clip(np.dot(thigh, p_fwd), -1, 1)))

    # Roll = abduction / adduction (lateral tilt, sign-flipped for right side)
    roll  = float(np.arcsin(np.clip(np.dot(thigh, p_right) * sign, -1, 1)))

    # Yaw = internal / external rotation (horizontal plane)
    thigh_horiz = _unit(np.array([thigh[0], thigh[1], 0.]) + 1e-9)
    ref_horiz   = _unit(np.array([p_fwd[0],  p_fwd[1],  0.]) + 1e-9)
    yaw = _signed_angle(ref_horiz, thigh_horiz, np.array([0., 0., 1.])) * sign

    return pitch, roll, yaw


# ── 7c. Knee ──────────────────────────────────────────────────────────────────

def _knee_angle(lms, side: str) -> float:
    """Knee flexion (hip-knee-ankle). 0 = fully extended, positive = flexed."""
    h = MP.LEFT_HIP   if side == "left" else MP.RIGHT_HIP
    k = MP.LEFT_KNEE  if side == "left" else MP.RIGHT_KNEE
    a = MP.LEFT_ANKLE if side == "left" else MP.RIGHT_ANKLE

    thigh = _p(lms, k) - _p(lms, h)
    shank = _p(lms, a) - _p(lms, k)
    # Supplementary angle → 0 when leg is straight
    return max(0.0, float(np.pi - _angle_between(thigh, shank)))


# ── 7d. Ankle ─────────────────────────────────────────────────────────────────

def _ankle_angles(lms, side: str):
    """
    Returns (pitch, roll) for the ankle.

    pitch – dorsi/plantarflexion  (sagittal, shank vs foot longitudinal axis)
    roll  – inversion/eversion    (coronal,  foot tilt; NEW in G1 vs H1)
    """
    k  = MP.LEFT_KNEE       if side == "left" else MP.RIGHT_KNEE
    a  = MP.LEFT_ANKLE      if side == "left" else MP.RIGHT_ANKLE
    h  = MP.LEFT_HEEL       if side == "left" else MP.RIGHT_HEEL
    fi = MP.LEFT_FOOT_INDEX if side == "left" else MP.RIGHT_FOOT_INDEX

    knee  = _p(lms, k)
    ankle = _p(lms, a)
    heel  = _p(lms, h)
    foot  = _p(lms, fi)

    shank      = _unit(ankle - knee)
    foot_long  = _unit(foot  - heel)        # heel → toe = longitudinal foot axis
    foot_lat   = _unit(np.cross(foot_long, shank))  # lateral foot axis

    # Sagittal axis = lateral foot direction
    pitch = _signed_angle(shank, foot_long, foot_lat)

    # Roll: foot lateral tilt relative to world-up, projected in coronal plane
    world_up  = np.array([0., 0., 1.])
    foot_norm = _unit(np.cross(foot_long, world_up) + 1e-9)  # ≈ medial direction
    roll = _signed_angle(world_up, _unit(np.cross(foot_long, foot_lat)), foot_long)
    roll *= (1.0 if side == "left" else -1.0)

    return float(pitch), float(roll)


# ── 7e. Shoulder ──────────────────────────────────────────────────────────────
def _shoulder_angles(lms, side: str):
    """
    Returns (pitch, roll, yaw) in G1 shoulder convention.
    Fixed right-arm inversion and symmetrical resting offsets.
    """
    # --- TUNING ---
    ROLL_SENSITIVITY = 1.5 
    # Left needs a negative offset to move "in" to rest. 
    # Right needs a positive offset to move "in" to rest.
    offset = -0.15 if side == "left" else 0.15

    s_idx = MP.LEFT_SHOULDER if side == "left" else MP.RIGHT_SHOULDER
    e_idx = MP.LEFT_ELBOW    if side == "left" else MP.RIGHT_ELBOW

    shoulder = _p(lms, s_idx)
    elbow    = _p(lms, e_idx)
    ls, rs   = _p(lms, MP.LEFT_SHOULDER), _p(lms, MP.RIGHT_SHOULDER)

    # DEFINE AXES
    body_left = _unit(ls - rs) # Points to the Left
    world_up  = np.array([0., 0., 1.])
    body_fwd  = _unit(np.cross(body_left, world_up))
    upper_arm = _unit(elbow - shoulder)

    # PITCH (Sagittal)
    v_fwd = np.dot(upper_arm, body_fwd)
    pitch = float(np.arcsin(np.clip(v_fwd, -1.0, 1.0)))

    # ROLL (Coronal)
    # Left-ward movement (Left outward) = Positive (+)
    # Right-ward movement (Right outward) = Negative (-)
    # Since body_left points left, np.dot already gives these correct signs!
    v_lat = np.dot(upper_arm, body_left)
    roll_raw = float(np.arcsin(np.clip(v_lat, -1.0, 1.0)))
    
    # We removed side_sign to fix the inversion.
    roll = (roll_raw * ROLL_SENSITIVITY) + offset

    return pitch, roll, 0.0

# ── 7f. Elbow ─────────────────────────────────────────────────────────────────
def _elbow_angle(lms, side: str) -> float:
    s_idx = MP.LEFT_SHOULDER if side == "left" else MP.RIGHT_SHOULDER
    e_idx = MP.LEFT_ELBOW    if side == "left" else MP.RIGHT_ELBOW
    w_idx = MP.LEFT_WRIST    if side == "left" else MP.RIGHT_WRIST

    shoulder = _p(lms, s_idx)
    elbow    = _p(lms, e_idx)
    wrist    = _p(lms, w_idx)

    upper_arm = _unit(elbow - shoulder)
    forearm   = _unit(wrist  - elbow)

    # ---- BODY RIGHT AXIS ----
    ls = _p(lms, MP.LEFT_SHOULDER)
    rs = _p(lms, MP.RIGHT_SHOULDER)
    body_right = _unit(rs - ls)

    # ---- BUILD ARM SAGITTAL PLANE ----
    br_perp = body_right - np.dot(body_right, upper_arm) * upper_arm

    if np.linalg.norm(br_perp) < 1e-6:
        world_up = np.array([0., 0., 1.])
        br_perp = world_up - np.dot(world_up, upper_arm) * upper_arm

    plane_normal = _unit(br_perp)

    # ---- FLEXION DIRECTION ----
    e_flex = _unit(np.cross(plane_normal, upper_arm))

    # ---- DECOMPOSE ----
    ext = np.dot(forearm, upper_arm)
    flex = np.dot(forearm, e_flex)

    angle = float(np.arctan2(flex, ext))

    # ---- SMALL DEAD ZONE ----
    DEAD_ZONE = np.deg2rad(10)
    angle = max(0.0, angle - DEAD_ZONE)

    return float(np.clip(-angle, -2.09, 0.0))

# ── 7g. Wrist Roll  (NEW – G1 specific) ──────────────────────────────────────

def _wrist_roll(lms, side: str) -> float:
    """
    Forearm supination / pronation (rotation around the forearm long axis).

    Strategy
    ────────
    1.  Forearm axis  = unit(wrist − elbow)
    2.  Reference hand-plane normal: elbow-wrist × elbow-upward  (neutral = palm down)
    3.  Actual hand-plane normal:    forearm × (index − pinky)
    4.  Roll = signed angle between reference and actual normals, around forearm axis.

    MediaPipe hand landmarks used: index_tip, pinky_tip (wrist-level proxies).
    Positive = supination (palm up), Negative = pronation (palm down).
    """
    e_idx = MP.LEFT_ELBOW  if side == "left" else MP.RIGHT_ELBOW
    w_idx = MP.LEFT_WRIST  if side == "left" else MP.RIGHT_WRIST
    i_idx = MP.LEFT_INDEX  if side == "left" else MP.RIGHT_INDEX
    p_idx = MP.LEFT_PINKY  if side == "left" else MP.RIGHT_PINKY

    elbow  = _p(lms, e_idx)
    wrist  = _p(lms, w_idx)
    index  = _p(lms, i_idx)
    pinky  = _p(lms, p_idx)

    forearm = _unit(wrist - elbow)

    # Actual hand lateral vector (index → pinky, projected perpendicular to forearm)
    hand_lat_raw = pinky - index
    hand_lat = _unit(hand_lat_raw - np.dot(hand_lat_raw, forearm) * forearm)

    # Reference lateral vector (world-up projected perpendicular to forearm)
    world_up  = np.array([0., 0., 1.])
    ref_lat   = world_up - np.dot(world_up, forearm) * forearm
    ref_lat   = _unit(ref_lat + 1e-9)

    roll = _signed_angle(ref_lat, hand_lat, forearm)
    # Flip sign for right arm so that "palm up" is positive on both sides
    if side == "right":
        roll = -roll

    return float(roll)


# ──────────────────────────────────────────────────────────────────────────────
# 8.  MAIN CONVERTER
# ──────────────────────────────────────────────────────────────────────────────

class MediaPipeToG1:
    """
    Converts a single MediaPipe world-landmark frame into a 23-DOF G1Frame.

    Usage
    ─────
    converter = MediaPipeToG1()
    frame = converter.convert(results.pose_world_landmarks.landmark)
    print(frame.angles.shape)   # (23,)
    print(dict(zip(JOINT_NAMES, frame.angles)))
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._lo = np.array([j.lo for j in G1_JOINTS], dtype=np.float32)
        self._hi = np.array([j.hi for j in G1_JOINTS], dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    def convert(self, world_landmarks, timestamp: float | None = None) -> G1Frame:
        lms = world_landmarks
        ts  = timestamp or time.time()

        confidence = float(np.mean([lm.visibility for lm in lms]))
        raw        = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

        # ── Waist ──────────────────────────────────────────────────────────
        w_yaw, w_roll, w_pitch = _waist_angles(lms)

        # ── Left Leg ───────────────────────────────────────────────────────
        lhp, lhr, lhy  = _hip_angles(lms, "left")
        lk             = _knee_angle(lms, "left")
        lap, lar       = _ankle_angles(lms, "left")

        # ── Right Leg ──────────────────────────────────────────────────────
        rhp, rhr, rhy  = _hip_angles(lms, "right")
        rk             = _knee_angle(lms, "right")
        rap, rar       = _ankle_angles(lms, "right")

        # ── Left Arm ───────────────────────────────────────────────────────
        lsp, lsr, lsy  = _shoulder_angles(lms, "left")
        le             = _elbow_angle(lms, "left")
        lwr            = _wrist_roll(lms, "left")

        # ── Right Arm ──────────────────────────────────────────────────────
        rsp, rsr, rsy  = _shoulder_angles(lms, "right")
        re             = _elbow_angle(lms, "right")
        rwr            = _wrist_roll(lms, "right")

        # ── Pack in G1 joint order ─────────────────────────────────────────
        angles = np.array([
            w_yaw, w_roll, w_pitch,           # waist  (3)
            lhp, lhr, lhy, lk, lap, lar,      # L-leg  (6)
            rhp, rhr, rhy, rk, rap, rar,      # R-leg  (6)
            lsp, lsr, lsy, le, lwr,            # L-arm  (5)
            rsp, rsr, rsy, re, rwr,            # R-arm  (5)
        ], dtype=np.float32)

        # ── Clamp to hardware limits ───────────────────────────────────────
        angles = np.clip(angles, self._lo, self._hi)

        return G1Frame(
            timestamp     = ts,
            angles        = angles,
            confidence    = confidence,
            raw_landmarks = raw,
        )

    # ──────────────────────────────────────────────────────────────────────────
    def is_valid(self, frame: G1Frame) -> bool:
        return frame.confidence >= self.confidence_threshold


# ──────────────────────────────────────────────────────────────────────────────
# 9.  DATASET RECORDER
# ──────────────────────────────────────────────────────────────────────────────

class DatasetRecorder:
    """
    Accumulates G1Frame objects and persists them to disk.

    Formats
    ───────
    .h5  / .hdf5  – HDF5 (recommended; stores metadata, compresses well)
    .npz           – NumPy compressed archive
    .csv           – human-readable angles-only table
    """

    def __init__(self, max_frames: int = 30_000):
        self.max_frames    = max_frames
        self._timestamps:  list[float]      = []
        self._angles:      list[np.ndarray] = []
        self._confidences: list[float]      = []
        self._landmarks:   list[np.ndarray] = []

    # ──────────────────────────────────────────────────────────────────────────
    def record(self, frame: G1Frame) -> bool:
        """Append frame. Returns False when buffer is full."""
        if len(self._timestamps) >= self.max_frames:
            return False
        self._timestamps.append(frame.timestamp)
        self._angles.append(frame.angles)
        self._confidences.append(frame.confidence)
        self._landmarks.append(frame.raw_landmarks)
        return True

    def clear(self):
        self._timestamps.clear()
        self._angles.clear()
        self._confidences.clear()
        self._landmarks.clear()

    @property
    def n_frames(self) -> int:
        return len(self._timestamps)

    # ──────────────────────────────────────────────────────────────────────────
    def save(self, path: str | Path, label: str = "") -> Path:
        """
        Save dataset to disk. Format inferred from extension.
        `label` is an optional gesture/action tag stored as metadata.
        """
        path  = Path(path)
        fmt   = path.suffix.lstrip(".")
        N     = self.n_frames

        if N == 0:
            raise RuntimeError("No frames to save.")

        ts_arr  = np.array(self._timestamps,                dtype=np.float64)
        ang_arr = np.stack(self._angles).astype(np.float32)   # (N, 23)
        cf_arr  = np.array(self._confidences,               dtype=np.float32)
        lm_arr  = np.stack(self._landmarks).astype(np.float32) # (N, 33, 3)

        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt in ("h5", "hdf5"):
            with h5py.File(path, "w") as f:
                f.attrs["robot"]       = "unitree_g1"
                f.attrs["source"]      = "mediapipe_world_landmarks"
                f.attrs["dof"]         = DOF
                f.attrs["label"]       = label
                f.attrs["joint_names"] = [n.encode() for n in JOINT_NAMES]
                kw = dict(compression="gzip", compression_opts=6)
                f.create_dataset("timestamps", data=ts_arr,  **kw)
                f.create_dataset("angles",     data=ang_arr, **kw)
                f.create_dataset("confidence", data=cf_arr,  **kw)
                f.create_dataset("landmarks",  data=lm_arr,  **kw)

        elif fmt == "npz":
            np.savez_compressed(
                path,
                timestamps  = ts_arr,
                angles      = ang_arr,
                confidence  = cf_arr,
                landmarks   = lm_arr,
                joint_names = np.array(JOINT_NAMES),
                label       = np.array(label),
            )

        elif fmt == "csv":
            import csv
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "confidence"] + JOINT_NAMES)
                for t, c, a in zip(ts_arr, cf_arr, ang_arr):
                    w.writerow([f"{t:.6f}", f"{c:.4f}"] + a.tolist())

        else:
            raise ValueError(f"Unknown format {fmt!r}. Use h5 / npz / csv.")

        print(f"[Recorder] Saved {N} frames → {path}"
              + (f"  (label='{label}')" if label else ""))
        return path