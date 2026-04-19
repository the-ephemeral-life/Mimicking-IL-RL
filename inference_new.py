import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import yaml
import json
import zmq

# ================= ARM/LEG CONFIG =================
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
]

LEG_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

# ================= HELPER FUNCTIONS =================
class BehavioralCloningMLP(nn.Module):
    def __init__(self, input_dim=99, output_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x): return self.net(x)

def get_joint_state_ids(model, names):
    qpos_ids, qvel_ids = [], []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_ids.append(model.jnt_qposadr[jid])
        qvel_ids.append(model.jnt_dofadr[jid])
    return qpos_ids, qvel_ids

def get_actuator_ids(model, names):
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names]

def get_gravity_orientation(q):
    qw, qx, qy, qz = q
    return np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)])

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def _print_inference_dashboard(step_counter, arm_targets, leg_targets):
    print("\033[2J\033[H", end="")
    print("="*55)
    print(f"🤖 G1 SIMULATION TELEMETRY  |  Step: {step_counter}")
    print("="*55)
    print("─── ARM TARGETS (Imitation Learning) ───")
    for name, angle in zip(ARM_JOINT_NAMES, arm_targets):
        print(f"{name:<30} : {angle:>+6.3f} rad")
    print("\n─── LEG TARGETS (RL Walking Policy) ────")
    for name, angle in zip(LEG_JOINT_NAMES, leg_targets):
        print(f"{name:<30} : {angle:>+6.3f} rad")
    print("="*55)

# ================= INITIALIZATION =================
if __name__ == "__main__":

    # Update these paths to your system!
    CONFIG_FILE      = r"C:\Users\arora\OneDrive\Desktop\IIT_Mandi\Sem_4\CS671 Deep Learning\Hackathon\Imitation Learning\unitree_rl_gym\deploy\deploy_mujoco\configs\g1.yaml"
    IL_MODEL_PATH    = r"C:\Users\arora\OneDrive\Desktop\IIT_Mandi\Sem_4\CS671 Deep Learning\Hackathon\Imitation Learning\G1_bc_brain.pth"
    LEGGED_GYM_ROOT_DIR = r"C:\Users\arora\OneDrive\Desktop\IIT_Mandi\Sem_4\CS671 Deep Learning\Hackathon\Imitation Learning\unitree_rl_gym"

    # Smoothing: lower = smoother but slower arm response
    EMA_ALPHA = 0.3

    with open(CONFIG_FILE, "r", encoding="utf-8", errors="ignore") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path    = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    simulation_dt      = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    kps            = np.array(config["kps"],            dtype=np.float32)
    kds            = np.array(config["kds"],            dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale  = config["action_scale"]
    cmd_scale     = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs     = config["num_obs"]

    cmd = np.array([0.2, 0.0, 0.0], dtype=np.float32)

    print("⚙️  Initializing MuJoCo...")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Resolve joint → qpos/qvel/actuator IDs from the XML (correct for any joint ordering)
    arm_qpos_ids, arm_qvel_ids = get_joint_state_ids(m, ARM_JOINT_NAMES)
    arm_act_ids                = get_actuator_ids(m, ARM_JOINT_NAMES)
    leg_qpos_ids, leg_qvel_ids = get_joint_state_ids(m, LEG_JOINT_NAMES)
    leg_act_ids                = get_actuator_ids(m, LEG_JOINT_NAMES)

    print("  Leg qpos indices:", leg_qpos_ids)   # sanity-check printout
    print("  Arm qpos indices:", arm_qpos_ids)

    print("🧠 Loading AI Brains...")
    policy   = torch.jit.load(policy_path)
    il_brain = BehavioralCloningMLP(output_dim=8)
    # weights_only=False keeps compatibility across PyTorch versions
    il_brain.load_state_dict(torch.load(IL_MODEL_PATH, map_location="cpu", weights_only=False))
    il_brain.eval()

    print("📡 Connecting to Team A Vision Node...")
    context = zmq.Context()
    socket  = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "VISION ")

    time.sleep(0.5)

    action         = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs            = np.zeros(num_obs,     dtype=np.float32)
    counter        = 0

    arm_target          = np.zeros(len(ARM_JOINT_NAMES))
    il_smoothed_output  = np.zeros(8)
    # BUG FIX 2: initialise raw_landmarks so it's always defined even before
    # the first ZMQ message arrives (avoids NameError on very first frame).
    raw_landmarks = [0.0] * 99

    kp_arm = 200
    kd_arm = 10

    # ================= SIMULATION LOOP =================
    with mujoco.viewer.launch_passive(m, d) as viewer:
        print("🚀 G1 Continuous Teleoperation Live!")

        while viewer.is_running():
            step_start = time.time()

            # ===== 1. VISION INFERENCE =====
            received_new_data = False
            try:
                while True:
                    message       = socket.recv_string(flags=zmq.NOBLOCK)
                    raw_landmarks = json.loads(message.replace("VISION ", ""))["landmarks"]
                    received_new_data = True
            except zmq.Again:
                pass

            if received_new_data:
                with torch.no_grad():
                    il_action = il_brain(
                        torch.tensor(raw_landmarks, dtype=torch.float32).unsqueeze(0)
                    ).squeeze().numpy()

                il_smoothed_output = (il_action * EMA_ALPHA) + (il_smoothed_output * (1.0 - EMA_ALPHA))

            # --- LEFT ARM ---
            arm_target[0] = il_smoothed_output[0]   # Pitch
            arm_target[1] = il_smoothed_output[1]   # Roll
            arm_target[2] = il_smoothed_output[2]   # Yaw
            arm_target[3] = il_smoothed_output[3]   # Elbow
            arm_target[4] = 0.0                     # Wrist locked

            # --- RIGHT ARM ---
            arm_target[5] = il_smoothed_output[4]   # Pitch
            arm_target[6] = il_smoothed_output[5]   # Roll
            arm_target[7] = il_smoothed_output[6]   # Yaw
            arm_target[8] = il_smoothed_output[7]   # Elbow
            arm_target[9] = 0.0                     # Wrist locked

            # ===== 2. ARM CONTROL (PD TORQUE) =====
            arm_tau = np.zeros(len(arm_qpos_ids))
            for i in range(len(arm_qpos_ids)):
                q   = d.qpos[arm_qpos_ids[i]]
                dq  = d.qvel[arm_qvel_ids[i]]
                arm_tau[i] = (arm_target[i] - q) * kp_arm + (-dq) * kd_arm

            # ===== 3. RL INFERENCE (LEGS) =====
            if counter % control_decimation == 0:
                # Use pre-resolved IDs — correct regardless of joint ordering in XML
                qj   = d.qpos[leg_qpos_ids]
                dqj  = d.qvel[leg_qvel_ids]
                quat  = d.qpos[3:7]
                omega = d.qvel[3:6]

                obs[:3]                          = omega * ang_vel_scale
                obs[3:6]                         = get_gravity_orientation(quat)
                obs[6:9]                         = cmd * cmd_scale
                obs[9:9+num_actions]             = (qj - default_angles) * dof_pos_scale
                obs[9+num_actions:9+2*num_actions]   = dqj * dof_vel_scale
                obs[9+2*num_actions:9+3*num_actions] = action

                obs_tensor     = torch.from_numpy(obs).unsqueeze(0)
                action         = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

            # ===== 4. LEG CONTROL (PD TORQUE) =====
            # BUG FIX 1: was d.qpos[7:7+12] / d.qvel[6:6+12] — those are
            # hardcoded offsets that break when waist or other joints sit
            # between the freejoint and the legs in the kinematic tree.
            # Use the IDs resolved from joint names instead.
            leg_tau = pd_control(
                target_dof_pos,
                d.qpos[leg_qpos_ids],   # ← fixed (was d.qpos[7:7+12])
                kps,
                np.zeros_like(kds),
                d.qvel[leg_qvel_ids],   # ← fixed (was d.qvel[6:6+12])
                kds,
            )

            # ===== 5. APPLY CONTROL & STEP =====
            d.ctrl[leg_act_ids] = leg_tau
            d.ctrl[arm_act_ids] = arm_tau

            mujoco.mj_step(m, d)
            viewer.sync()
            counter += 1

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)