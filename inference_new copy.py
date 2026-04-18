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
    def __init__(self, input_dim=99, output_dim=29):
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

# ================= INITIALIZATION =================
if __name__ == "__main__":
    
    # HARDCODED PATHS FOR HACKATHON SPEED
    CONFIG_FILE = r"C:\Users\arora\OneDrive\Desktop\IIT_Mandi\Sem_4\CS671 Deep Learning\Hackathon\Imitation Learning\unitree_rl_gym\deploy\deploy_mujoco\configs\g1.yaml" 
    IL_MODEL_PATH = r"C:\Users\arora\OneDrive\Desktop\IIT_Mandi\Sem_4\CS671 Deep Learning\Hackathon\Imitation Learning\G1_bc_brain.pth"
    LEGGED_GYM_ROOT_DIR = r"C:\Users\arora\OneDrive\Desktop\IIT_Mandi\Sem_4\CS671 Deep Learning\Hackathon\Imitation Learning\unitree_rl_gym"
    
    EMA_ALPHA = 0.3

    with open(CONFIG_FILE, "r", encoding="utf-8", errors="ignore") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = r"C:\Users\arora\OneDrive\Desktop\IIT_Mandi\Sem_4\AR524 Robot Simulators\MuJoCo\mujoco_menagerie\unitree_g1\scene.xml"  # ← ADD THIS

    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    
    # Positive X = Walk Forward!
    cmd = np.array([0.2, 0.0, 0.0], dtype=np.float32) 

    print("⚙️ Initializing MuJoCo...")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    print(f"Total actuators in loaded XML: {m.nu}")
    print(f"XML being loaded: {xml_path}")
    m.opt.timestep = simulation_dt

    leg_act_ids  = get_actuator_ids(m, LEG_JOINT_NAMES)
    leg_qpos_ids, leg_qvel_ids = get_joint_state_ids(m, LEG_JOINT_NAMES)
    arm_act_ids  = get_actuator_ids(m, ARM_JOINT_NAMES)

# ← ADD THIS
    for name, aid in zip(ARM_JOINT_NAMES, arm_act_ids):
        print(f"  {name} → actuator id: {aid}")
    arm_qpos_ids, arm_qvel_ids = get_joint_state_ids(m, ARM_JOINT_NAMES)

    print("🧠 Loading AI Brains...")
    policy = torch.jit.load(policy_path)
    il_brain = BehavioralCloningMLP(output_dim=29)
    il_brain.load_state_dict(torch.load(IL_MODEL_PATH))
    il_brain.eval()
    
    print("📡 Connecting to Team A Vision Node...")
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "VISION ")

    import time
    time.sleep(0.5)  # ← ADD THIS LINE, gives ZMQ time to handshake

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # Initialize arms to the safe "Bent A-Pose" to preserve RL balance
    arm_target = np.zeros(len(ARM_JOINT_NAMES))
    arm_target[0] = 0.2  # left_shoulder_pitch
    arm_target[1] = 0.2  # left_shoulder_roll
    arm_target[3] = 1.28 # left_elbow
    arm_target[5] = 0.2  # right_shoulder_pitch
    arm_target[6] = -0.2 # right_shoulder_roll
    arm_target[8] = 1.28 # right_elbow
    
    # Initialize the AI array to the SAFE BENT POSE so the robot balances instantly
    il_smoothed_output = np.zeros(29)
    il_smoothed_output[15:19] = [0.2, 0.2, 0.0, 1.28]  # Left arm bent
    il_smoothed_output[22:26] = [0.2, -0.2, 0.0, 1.28] # Right arm bent


    # ================= SIMULATION LOOP =================
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        print("🚀 G1 Vision Fusion Live!")

        while viewer.is_running():
            step_start = time.time()

            # ===== 1. VISION INFERENCE (BUFFER DRAIN) =====
            received_new_data = False
            try:
                while True:
                    message = socket.recv_string(flags=zmq.NOBLOCK)
                    raw_landmarks = json.loads(message.replace("VISION ", ""))["landmarks"]
                    received_new_data = True
            except zmq.Again:
                pass

            if received_new_data:
                with torch.no_grad():
                    il_action = il_brain(torch.tensor(raw_landmarks, dtype=torch.float32).unsqueeze(0)).squeeze().numpy()
                il_smoothed_output = (il_action * EMA_ALPHA) + (il_smoothed_output * (1.0 - EMA_ALPHA))

            # ADD THIS right after the arm_target mapping block:
            arm_target[0:4] = il_smoothed_output[15:19]
            arm_target[4] = 0.0
            arm_target[5:9] = il_smoothed_output[22:26]
            arm_target[9] = 0.0

            # ← ADD THIS:
            

            # ===== 2. MANUAL PD CONTROL (Angles to Torque) =====
            # ===== 2. DIRECT POSITION CONTROL (correct for position actuators) =====
            d.ctrl[15:20] = arm_target[0:5]   # left arm: shoulder pitch, roll, yaw, elbow, wrist_roll
            d.ctrl[22:27] = arm_target[5:10]  # right arm: shoulder pitch, roll, yaw, elbow, wrist_roll
            if counter % 60 == 0:
                print(f"[DEBUG] ZMQ alive: {received_new_data} | L arm: {arm_target[0:4].round(3)} | R arm: {arm_target[5:9].round(3)}")
                print(f"[DEBUG] il_action L: {il_smoothed_output[15:19].round(3)} | il_action R: {il_smoothed_output[22:26].round(3)}")  # ← ADD
                print(f"[DEBUG] arm_act_ids: {arm_act_ids}")   # ← ADD
            # ===== 3. RL INFERENCE (LEGS) =====
            if counter % control_decimation == 0:
                qj = d.qpos[leg_qpos_ids]
                dqj = d.qvel[leg_qvel_ids]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                obs[:3] = omega * ang_vel_scale
                obs[3:6] = get_gravity_orientation(quat)
                obs[6:9] = cmd * cmd_scale
                obs[9:9+num_actions] = (qj - default_angles) * dof_pos_scale
                obs[9+num_actions:9+2*num_actions] = dqj * dof_vel_scale
                obs[9+2*num_actions:9+3*num_actions] = action

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()

                target_dof_pos = action * action_scale + default_angles

            # ===== 4. LEG CONTROL (TORQUE) =====
            # ===== 4. LEG CONTROL (POSITION) =====
            d.ctrl[leg_act_ids] = target_dof_pos

            # ===== 5. STEP =====
            mujoco.mj_step(m, d)
            viewer.sync()
            counter += 1

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)