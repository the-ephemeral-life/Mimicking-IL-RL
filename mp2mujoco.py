"""
mp2mujoco.py
============
Pipeline: MediaPipe World Landmarks → Unitree G1 MuJoCo Joint Angles

Stages:
  1. Landmark extraction  – pull pose landmarks from MediaPipe
  2. Coordinate alignment – rotate MP frame to G1/MuJoCo frame
  3. Angle computation    – vectors → joint angles (rotation decomposition)
  4. Joint limit clamping – enforce G1 hardware limits
  5. Dataset recording    – accumulate frames, save to HDF5 / CSV / NPZ
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import h5py
import mediapipe as mp
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1.  MEDIAPIPE LANDMARK INDEX MAP
# ──────────────────────────────────────────────────────────────────────────────

class MP:
    """MediaPipe Pose landmark indices (WORLD coordinates, hip-centred, metres)."""
    NOSE            = 0
    LEFT_SHOULDER   = 11
    RIGHT_SHOULDER  = 12
    LEFT_ELBOW      = 13
    RIGHT_ELBOW     = 14
    LEFT_WRIST      = 15
    RIGHT_WRIST     = 16
    LEFT_HIP        = 23
    RIGHT_HIP       = 24
    LEFT_KNEE       = 25
    RIGHT_KNEE      = 26
    LEFT_ANKLE      = 27
    RIGHT_ANKLE     = 28
    LEFT_HEEL       = 29
    RIGHT_HEEL      = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX= 32


# ──────────────────────────────────────────────────────────────────────────────
# 2.  UNITREE G1 JOINT SPEC
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# 2.  UNITREE G1 JOINT SPEC (29 DOFs)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class JointSpec:
    name:  str
    lo:    float
    hi:    float

G1_JOINTS: list[JointSpec] = [
    # Legs (0-11) - We will output 0 here because RL handles legs
    JointSpec("left_hip_pitch", -2.53, 2.87), JointSpec("left_hip_roll", -0.52, 2.96), JointSpec("left_hip_yaw", -2.75, 2.75),
    JointSpec("left_knee", -0.08, 2.87), JointSpec("left_ankle_pitch", -0.87, 0.52), JointSpec("left_ankle_roll", -0.26, 0.26),
    JointSpec("right_hip_pitch", -2.53, 2.87), JointSpec("right_hip_roll", -2.96, 0.52), JointSpec("right_hip_yaw", -2.75, 2.75),
    JointSpec("right_knee", -0.08, 2.87), JointSpec("right_ankle_pitch", -0.87, 0.52), JointSpec("right_ankle_roll", -0.26, 0.26),

    # Waist (12-14)
    JointSpec("waist_yaw", -2.618, 2.618), JointSpec("waist_roll", -0.52, 0.52), JointSpec("waist_pitch", -0.52, 0.52),

    # Left Arm (15-21)
    JointSpec("left_shoulder_pitch", -3.08, 2.67), JointSpec("left_shoulder_roll", -1.58, 2.25), JointSpec("left_shoulder_yaw", -2.61, 2.61),
    JointSpec("left_elbow", -1.04, 2.09),
    JointSpec("left_wrist_roll", -1.97, 1.97), JointSpec("left_wrist_pitch", -1.61, 1.61), JointSpec("left_wrist_yaw", -1.61, 1.61),

    # Right Arm (22-28)
    JointSpec("right_shoulder_pitch", -3.08, 2.67), JointSpec("right_shoulder_roll", -2.25, 1.58), JointSpec("right_shoulder_yaw", -2.61, 2.61),
    JointSpec("right_elbow", -1.04, 2.09),
    JointSpec("right_wrist_roll", -1.97, 1.97), JointSpec("right_wrist_pitch", -1.61, 1.61), JointSpec("right_wrist_yaw", -1.61, 1.61),
]

JOINT_NAMES = [j.name for j in G1_JOINTS]
DOF = len(G1_JOINTS) # 29

# ──────────────────────────────────────────────────────────────────────────────
# 3.  FRAME  (one timestep of output)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class G1Frame:
    timestamp:  float
    angles:     np.ndarray          # shape (DOF,) in radians
    confidence: float               # mean landmark visibility [0,1]
    raw_landmarks: np.ndarray       # shape (33, 3) world coords


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MATH HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle between two vectors (rad)."""
    cos = np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0)
    return float(np.arccos(cos))

def _signed_angle(v1: np.ndarray, v2: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle from v1 to v2 around `axis` (right-hand rule)."""
    cross = np.cross(v1, v2)
    s = np.linalg.norm(cross)
    c = np.dot(v1, v2)
    angle = np.arctan2(s, c)
    if np.dot(cross, axis) < 0:
        angle = -angle
    return float(angle)

def _decompose_elbow(proximal: np.ndarray, distal: np.ndarray) -> float:
    """Return elbow flexion angle (always ≥ 0 for G1)."""
    return _angle_between(proximal, distal)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  COORDINATE FRAME ALIGNMENT
#     MediaPipe world:  x→right, y→up,   z→toward camera  (right-hand)
#     MuJoCo / G1:      x→forward, y→left, z→up            (right-hand)
# ──────────────────────────────────────────────────────────────────────────────

# Rotation matrix  R such that  v_mujoco = R @ v_mediapipe
_R_MP2MJ = np.array([
    [ 0,  0, -1],   # mj-x ← -mp-z  (forward = depth into scene)
    [-1,  0,  0],   # mj-y ← -mp-x  (left)
    [ 0,  1,  0],   # mj-z ←  mp-y  (up)
], dtype=float)

def _lm_to_mj(lm) -> np.ndarray:
    """Extract world landmark as MuJoCo-frame 3-vector."""
    mp_vec = np.array([lm.x, lm.y, lm.z])
    return _R_MP2MJ @ mp_vec

def _get_pos(lms, idx: int) -> np.ndarray:
    return _lm_to_mj(lms[idx])


# ──────────────────────────────────────────────────────────────────────────────
# 6.  ANGLE COMPUTATION FUNCTIONS
#     Each function receives the full landmark array and returns radians.
# ──────────────────────────────────────────────────────────────────────────────

def _torso_yaw(lms) -> float:
    """Shoulder-line yaw relative to world forward (+x)."""
    ls = _get_pos(lms, MP.LEFT_SHOULDER)
    rs = _get_pos(lms, MP.RIGHT_SHOULDER)
    shoulder_vec = ls - rs          # points left
    # project onto horizontal plane
    shoulder_flat = np.array([shoulder_vec[0], shoulder_vec[1], 0.0])
    world_left    = np.array([0.0, 1.0, 0.0])
    return _signed_angle(world_left, shoulder_flat, np.array([0, 0, 1]))


def _hip_angles(lms, side: str):
    """
    Returns (yaw, roll, pitch) for one hip.
    Uses pelvis-relative thigh vector decomposed onto anatomical axes.
    """
    if side == "left":
        hip_idx, knee_idx = MP.LEFT_HIP, MP.LEFT_KNEE
        sign = 1.0
    else:
        hip_idx, knee_idx = MP.RIGHT_HIP, MP.RIGHT_KNEE
        sign = -1.0

    # Pelvis frame axes (in MuJoCo world frame)
    lh = _get_pos(lms, MP.LEFT_HIP)
    rh = _get_pos(lms, MP.RIGHT_HIP)
    pelvis_right = _unit(rh - lh)               # +x of pelvis (points right)
    world_up     = np.array([0., 0., 1.])
    pelvis_fwd   = _unit(np.cross(pelvis_right, world_up))  # +y pelvis
    pelvis_up    = _unit(np.cross(pelvis_fwd, pelvis_right))

    # Thigh vector in pelvis frame
    hip  = _get_pos(lms, hip_idx)
    knee = _get_pos(lms, knee_idx)
    thigh_world = _unit(knee - hip)

    # Decompose
    pitch = float(np.arcsin(np.clip(np.dot(thigh_world, pelvis_fwd), -1, 1)))
    roll  = float(np.arcsin(np.clip(np.dot(thigh_world, pelvis_right) * sign, -1, 1)))

    # Yaw: rotation of thigh around vertical axis (abduction/adduction plane)
    thigh_horiz = _unit(np.array([thigh_world[0], thigh_world[1], 0.0]) + 1e-9)
    ref_horiz   = _unit(np.array([pelvis_fwd[0], pelvis_fwd[1], 0.0]) + 1e-9)
    yaw = _signed_angle(ref_horiz, thigh_horiz, world_up) * sign

    return yaw, roll, pitch


def _knee_angle(lms, side: str) -> float:
    """Knee flexion via hip-knee-ankle angle (always positive for G1)."""
    if side == "left":
        h, k, a = MP.LEFT_HIP, MP.LEFT_KNEE, MP.LEFT_ANKLE
    else:
        h, k, a = MP.RIGHT_HIP, MP.RIGHT_KNEE, MP.RIGHT_ANKLE

    hip   = _get_pos(lms, h)
    knee  = _get_pos(lms, k)
    ankle = _get_pos(lms, a)

    thigh  = knee  - hip
    shank  = ankle - knee
    angle  = np.pi - _angle_between(thigh, shank)   # 0 = straight
    return max(0.0, float(angle))


def _ankle_angle(lms, side: str) -> float:
    """
    Ankle dorsi/plantarflexion from shank-foot angle.
    Negative = dorsiflexion, Positive = plantarflexion (G1 convention).
    """
    if side == "left":
        k, a, f = MP.LEFT_KNEE, MP.LEFT_ANKLE, MP.LEFT_FOOT_INDEX
    else:
        k, a, f = MP.RIGHT_KNEE, MP.RIGHT_ANKLE, MP.RIGHT_FOOT_INDEX

    knee  = _get_pos(lms, k)
    ankle = _get_pos(lms, a)
    foot  = _get_pos(lms, f)

    shank = _unit(ankle - knee)
    foot_vec = _unit(foot  - ankle)
    # Signed in sagittal plane
    sag_axis = np.array([0., 1., 0.])      # Y = left (sagittal normal)
    return _signed_angle(shank, foot_vec, sag_axis)


def _shoulder_angles(lms, side: str):
    """
    Returns (pitch, roll, yaw) in G1 shoulder convention.
    Pitch  – flexion/extension (sagittal)
    Roll   – abduction/adduction (coronal)
    Yaw    – internal/external rotation (transverse)
    """
    if side == "left":
        s_idx, e_idx, opp_s = MP.LEFT_SHOULDER, MP.LEFT_ELBOW,  MP.RIGHT_SHOULDER
        roll_sign = 1.0
    else:
        s_idx, e_idx, opp_s = MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW, MP.LEFT_SHOULDER
        roll_sign = -1.0

    shoulder = _get_pos(lms, s_idx)
    elbow    = _get_pos(lms, e_idx)
    opp_sh   = _get_pos(lms, opp_s)

    # Shoulder frame
    trunk_right = _unit(opp_sh - shoulder) * (-1 if side == "left" else 1)
    world_up    = np.array([0., 0., 1.])
    trunk_fwd   = _unit(np.cross(trunk_right, world_up))

    upper_arm = _unit(elbow - shoulder)

    pitch = float(np.arcsin(np.clip( np.dot(upper_arm, trunk_fwd),  -1, 1)))
    roll  = float(np.arcsin(np.clip( np.dot(upper_arm, world_up),   -1, 1))) * roll_sign
    # Yaw from projection onto horizontal plane
    ua_horiz = _unit(np.array([upper_arm[0], upper_arm[1], 0.0]) + 1e-9)
    ref      = _unit(np.array([trunk_fwd[0], trunk_fwd[1], 0.0]) + 1e-9)
    yaw = _signed_angle(ref, ua_horiz, world_up)

    return pitch, roll, yaw


def _elbow_angle(lms, side: str) -> float:
    """Elbow flexion (shoulder-elbow-wrist angle, always ≥ 0)."""
    if side == "left":
        s, e, w = MP.LEFT_SHOULDER, MP.LEFT_ELBOW, MP.LEFT_WRIST
    else:
        s, e, w = MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW, MP.RIGHT_WRIST

    shoulder = _get_pos(lms, s)
    elbow    = _get_pos(lms, e)
    wrist    = _get_pos(lms, w)

    upper = elbow - shoulder
    lower = wrist - elbow
    return max(0.0, np.pi - _angle_between(upper, lower))


# ──────────────────────────────────────────────────────────────────────────────
# 7.  MAIN CONVERTER
# ──────────────────────────────────────────────────────────────────────────────

class MediaPipeToG1:
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        # These now load 29 limits from G1_JOINTS
        self._lo = np.array([j.lo for j in G1_JOINTS])
        self._hi = np.array([j.hi for j in G1_JOINTS])

    def convert(self, world_landmarks, timestamp: float | None = None) -> G1Frame:
        lms = world_landmarks

        # Mean landmark visibility as quality proxy
        visibilities = np.array([lm.visibility for lm in lms])
        confidence   = float(np.mean(visibilities))

        # Raw landmark array (for dataset storage)
        raw = np.array([[lm.x, lm.y, lm.z] for lm in lms])

        # ── Compute Upper Body Angles ───────────────────────────────────────
        # We skip computing legs here because the RL policy handles them natively!
        torso_y = _torso_yaw(lms)
        lsp, lsr, lsy = _shoulder_angles(lms, "left")
        le             = _elbow_angle(lms, "left")
        rsp, rsr, rsy = _shoulder_angles(lms, "right")
        re             = _elbow_angle(lms, "right")

        # ── Pack in G1 joint order (29 DOFs) ────────────────────────────────
        # Initialize an array of 29 zeros
        angles = np.zeros(29, dtype=np.float32)
        
        # Insert the calculated angles into the specific G1 arm/waist indices
        angles[12] = torso_y # Waist Yaw
        
        # Left Arm (Shoulder Pitch, Roll, Yaw, Elbow)
        angles[15], angles[16], angles[17], angles[18] = lsp, lsr, lsy, le 
        
        # Right Arm (Shoulder Pitch, Roll, Yaw, Elbow)
        angles[22], angles[23], angles[24], angles[25] = rsp, rsr, rsy, re 

        # ── Clamp to hardware limits ────────────────────────────────────────
        # Now shapes match: (29,) clamped against (29,) and (29,)
        angles = np.clip(angles, self._lo, self._hi).astype(np.float32)

        return G1Frame(
            timestamp     = timestamp or time.time(),
            angles        = angles,
            confidence    = confidence,
            raw_landmarks = raw,
        )

    def is_valid(self, frame: G1Frame) -> bool:
        """Check whether the frame meets minimum quality."""
        return frame.confidence >= self.confidence_threshold
# ──────────────────────────────────────────────────────────────────────────────
# 8.  DATASET RECORDER
# ──────────────────────────────────────────────────────────────────────────────

class DatasetRecorder:
    """
    Accumulates G1Frame objects and writes them to disk.

    Supported formats:
      • HDF5  (.h5)  – recommended for large datasets with metadata
      • NPZ   (.npz) – NumPy compressed archive
      • CSV   (.csv) – human-readable, angles only

    Usage
    -----
    rec = DatasetRecorder(max_frames=3000)
    rec.record(frame)
    rec.save("dataset.h5")
    """

    def __init__(self, max_frames: int = 10_000):
        self.max_frames = max_frames
        self._timestamps:  list[float]     = []
        self._angles:      list[np.ndarray]= []
        self._confidences: list[float]     = []
        self._landmarks:   list[np.ndarray]= []

    # ------------------------------------------------------------------
    def record(self, frame: G1Frame) -> bool:
        """Returns False when buffer is full."""
        if len(self._timestamps) >= self.max_frames:
            return False
        self._timestamps.append(frame.timestamp)
        self._angles.append(frame.angles)
        self._confidences.append(frame.confidence)
        self._landmarks.append(frame.raw_landmarks)
        return True

    @property
    def n_frames(self) -> int:
        return len(self._timestamps)

    # ------------------------------------------------------------------
    def save(self, path: str | Path, fmt: str = "h5") -> Path:
        path = Path(path)
        fmt  = path.suffix.lstrip(".") or fmt

        angles_arr = np.stack(self._angles)                 # (N, 19)
        ts_arr     = np.array(self._timestamps)             # (N,)
        conf_arr   = np.array(self._confidences)            # (N,)
        lm_arr     = np.stack(self._landmarks)              # (N, 33, 3)

        if fmt in ("h5", "hdf5"):
            with h5py.File(path, "w") as f:
                f.attrs["robot"]       = "unitree_G1"
                f.attrs["source"]      = "mediapipe_world_landmarks"
                f.attrs["dof"]         = DOF
                f.attrs["joint_names"] = np.bytes_(JOINT_NAMES)
                f.create_dataset("timestamps",  data=ts_arr,     compression="gzip")
                f.create_dataset("angles",      data=angles_arr, compression="gzip")
                f.create_dataset("confidence",  data=conf_arr,   compression="gzip")
                f.create_dataset("landmarks",   data=lm_arr,     compression="gzip")

        elif fmt == "npz":
            np.savez_compressed(
                path,
                timestamps  = ts_arr,
                angles      = angles_arr,
                confidence  = conf_arr,
                landmarks   = lm_arr,
                joint_names = np.array(JOINT_NAMES),
            )

        elif fmt == "csv":
            import csv
            with open(path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["timestamp", "confidence"] + JOINT_NAMES)
                for t, c, a in zip(ts_arr, conf_arr, angles_arr):
                    writer.writerow([f"{t:.4f}", f"{c:.3f}"] + a.tolist())

        else:
            raise ValueError(f"Unknown format: {fmt!r}. Use h5 / npz / csv.")

        print(f"[Recorder] Saved {self.n_frames} frames → {path}")
        return path