"""
sim2mujoco_g1_frontflip.py — Unitree G1 29DOF Front-Flip → MuJoCo
==================================================================
直接读取训练时导出的 deploy.yaml，无需手动填写任何超参数。

用法:
  python sim2mujoco_g1_frontflip.py \\
      --model   /path/to/g1_29dof_rev_1_0.xml \\
      --policy  /path/to/exported/policy.pt \\
      --deploy  /path/to/exported/deploy.yaml \\
      --motion  /path/to/front_flip.npz \\
      --render  [--record out.mp4]

python scripts/mimic/sim2mujoco_g1_frontflip.py  \\                                                  
  --model /home/user/Workspace/Noetix_GMT/source/general_motion_tracking/general_motion_tracking/assets/unitree_description/mjcf/g1.xml \
  --policy logs/rsl_rl/unitree_g1_29dof_mimic_front_flip/2026-03-10_07-20-11/exported/policy.pt \
  --motion source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/front_flip/front_flip.npz  \
  --deploy logs/rsl_rl/unitree_g1_29dof_mimic_front_flip/2026-03-10_07-20-11/params/deploy.yaml \
  --render 
"""

from __future__ import annotations
import argparse, time
from pathlib import Path

import numpy as np
import torch
import yaml
import mujoco
import mujoco.viewer

# ══════════════════════════════════════════════════════════════════
# 训练 cfg 固定超参数
# ══════════════════════════════════════════════════════════════════
SIM_DT         = 0.005
DECIMATION     = 4              # step_dt = 0.02 = SIM_DT × DECIMATION
EPISODE_LENGTH = 30.0

# cfg body_names（14 个追踪 body，顺序来自 CommandsCfg）
TRACKED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
NUM_BODIES              = len(TRACKED_BODY_NAMES)   # 14
ANCHOR_BODY             = "torso_link"
ANCHOR_IDX_IN_TRACKED   = TRACKED_BODY_NAMES.index(ANCHOR_BODY)   # 7


# ══════════════════════════════════════════════════════════════════
# 四元数工具（w,x,y,z 约定）
# ══════════════════════════════════════════════════════════════════

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1;  w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)

def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

def quat_apply(q, v):
    w,x,y,z = q
    t = 2.0 * np.array([y*v[2]-z*v[1], z*v[0]-x*v[2], x*v[1]-y*v[0]], dtype=np.float32)
    return v + w*t + np.cross(np.array([x,y,z], dtype=np.float32), t)

def quat_rotate_inverse(q, v):
    return quat_apply(quat_inv(q), v)

def quat_to_rot_matrix(q):
    """(w,x,y,z) → 3×3 rotation matrix"""
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float32)

def rot6d(q):
    """四元数 → 旋转矩阵前两列展平 [6]，对齐 matrix_from_quat()[..,:2].reshape(-1)"""
    return quat_to_rot_matrix(q)[:, :2].flatten()

def yaw_quat(q):
    w,x,y,z = q
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return np.array([cy, 0., 0., sy], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════
# MotionClip
# ══════════════════════════════════════════════════════════════════

class MotionClip:
    def __init__(self, path: str, body_indexes: list[int]):
        data            = np.load(path)
        self.fps        = int(data["fps"][0])
        self.motion_dt  = 1.0 / self.fps
        self.joint_pos  = data["joint_pos"].astype(np.float32)
        self.joint_vel  = data["joint_vel"].astype(np.float32)
        self._bpos      = data["body_pos_w"].astype(np.float32)
        self._bquat     = data["body_quat_w"].astype(np.float32)
        self._blvel     = data["body_lin_vel_w"].astype(np.float32)
        self._bavel     = data["body_ang_vel_w"].astype(np.float32)
        self._idx       = body_indexes
        self.T          = self.joint_pos.shape[0]
        print(f"[MotionClip] fps={self.fps}, T={self.T}, duration={self.T*self.motion_dt:.2f}s")

    @property
    def body_pos_w(self):     return self._bpos[:, self._idx]
    @property
    def body_quat_w(self):    return self._bquat[:, self._idx]

    def get_frame(self, t: float) -> dict:
        i = min(int(t / self.motion_dt), self.T - 1)
        return {
            "joint_pos":   self.joint_pos[i],
            "joint_vel":   self.joint_vel[i],
            "body_pos_w":  self.body_pos_w[i],
            "body_quat_w": self.body_quat_w[i],
        }


# ══════════════════════════════════════════════════════════════════
# Deploy yaml 加载
# ══════════════════════════════════════════════════════════════════

class DeployConfig:
    """
    从 deploy.yaml 读取所有部署所需参数。

    关键字段：
      joint_ids_map       IsaacLab 训练关节顺序 → MuJoCo xml 关节顺序的映射
                          deploy[i] = mujoco_joint_index[i]
                          即: mujoco_ctrl[joint_ids_map[i]] ← action[i]
      default_joint_pos   训练顺序下的默认关节角 [29]
      stiffness / damping PD 增益，训练顺序 [29]
      actions.JointPositionAction.scale   动作缩放，训练顺序 [29]
      actions.JointPositionAction.offset  等同于 default_joint_pos
      step_dt             控制周期（验证用）
    """
    def __init__(self, path: str):
        with open(path) as f:
            cfg = yaml.safe_load(f)

        self.joint_ids_map    = np.array(cfg["joint_ids_map"], dtype=int)
        self.step_dt          = float(cfg["step_dt"])
        self.stiffness        = np.array(cfg["stiffness"],        dtype=np.float32)
        self.damping          = np.array(cfg["damping"],          dtype=np.float32)
        self.default_joint_pos= np.array(cfg["default_joint_pos"],dtype=np.float32)

        act_cfg = cfg["actions"]["JointPositionAction"]
        self.action_scale  = np.array(act_cfg["scale"],  dtype=np.float32)
        self.action_offset = np.array(act_cfg["offset"], dtype=np.float32)

        n = len(self.joint_ids_map)
        assert len(self.stiffness)        == n
        assert len(self.damping)          == n
        assert len(self.default_joint_pos)== n
        assert len(self.action_scale)     == n
        self.num_joints = n

        print(f"[DeployConfig] num_joints={n}, step_dt={self.step_dt}")
        print(f"[DeployConfig] joint_ids_map={self.joint_ids_map.tolist()}")
        print(f"[DeployConfig] default_joint_pos[:6]={self.default_joint_pos[:6]}")
        print(f"[DeployConfig] action_scale[:6]={self.action_scale[:6]}")
        print(f"[DeployConfig] stiffness[:6]={self.stiffness[:6]}")
        print(f"[DeployConfig] damping[:6]={self.damping[:6]}")

        # 验证 step_dt
        expected_dt = SIM_DT * DECIMATION
        if abs(self.step_dt - expected_dt) > 1e-6:
            print(f"[Warning] deploy step_dt={self.step_dt} != SIM_DT×DECIMATION={expected_dt}")


# ══════════════════════════════════════════════════════════════════
# MuJoCo body 工具
# ══════════════════════════════════════════════════════════════════

def get_body_xpos_xquat(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[bid].copy(), data.xquat[bid].copy()

def get_body_cvel(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.cvel[bid, 3:].copy(), data.cvel[bid, :3].copy()  # lin, ang

def build_body_indexes(model: mujoco.MjModel) -> list[int]:
    """TRACKED_BODY_NAMES → npz body_pos_w 中的下标（去掉 world body 后的 id）"""
    robot_bodies = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i - 1
        for i in range(1, model.nbody)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    }
    print(f"[MuJoCo] robot body 数量: {len(robot_bodies)}")
    indexes = []
    for name in TRACKED_BODY_NAMES:
        if name not in robot_bodies:
            raise ValueError(f"Body '{name}' 不在 MuJoCo model 中。可用: {list(robot_bodies)}")
        indexes.append(robot_bodies[name])
    print(f"[MotionClip] body_indexes: {indexes}")
    return indexes


# ══════════════════════════════════════════════════════════════════
# 观测构建（对齐 observations.py + commands.py）
# ══════════════════════════════════════════════════════════════════

def compute_relative_bodies(ref, robot_anchor_pos_w, robot_anchor_quat_w):
    """对齐 commands.py::_update_command"""
    ref_anchor_pos  = ref["body_pos_w"][ANCHOR_IDX_IN_TRACKED]
    ref_anchor_quat = ref["body_quat_w"][ANCHOR_IDX_IN_TRACKED]

    delta_pos = robot_anchor_pos_w.copy()
    delta_pos[2] = ref_anchor_pos[2]   # Z from ref

    delta_ori = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat)))

    diff = ref["body_pos_w"] - ref_anchor_pos[None, :]
    rotated = np.stack([quat_apply(delta_ori, diff[i]) for i in range(NUM_BODIES)])
    body_pos_rel_w  = delta_pos[None, :] + rotated
    body_quat_rel_w = np.stack([quat_mul(delta_ori, ref["body_quat_w"][i])
                                 for i in range(NUM_BODIES)])
    return body_pos_rel_w, body_quat_rel_w


def build_observation(model, data, motion: MotionClip, deploy: DeployConfig,
                       sim_time: float, last_action: np.ndarray) -> np.ndarray:
    """
    obs 拼接（对齐 PolicyCfg，训练关节顺序）：
      motion_command      [58]   cat(ref_joint_pos, ref_joint_vel)  训练顺序
      motion_anchor_pos_b  [3]   ref anchor 相对 robot anchor frame
      motion_anchor_ori_b  [6]   rot6d
      body_pos            [42]   robot body 相对 robot anchor [14×3]
      body_ori            [84]   rot6d [14×6]
      base_lin_vel         [3]
      base_ang_vel         [3]
      joint_pos_rel       [29]   训练顺序
      joint_vel           [29]   训练顺序
      last_action         [29]   训练顺序
    total = 286
    """
    ref = motion.get_frame(sim_time)

    # base (pelvis)
    base_pos, base_quat = get_body_xpos_xquat(model, data, "pelvis")
    base_lin_w, base_ang_w = get_body_cvel(model, data, "pelvis")
    base_lin_vel = quat_rotate_inverse(base_quat, base_lin_w)
    base_ang_vel = quat_rotate_inverse(base_quat, base_ang_w)

    # robot anchor (torso_link)
    robot_anchor_pos, robot_anchor_quat = get_body_xpos_xquat(model, data, ANCHOR_BODY)

    # body_pos_b / body_ori_b：robot 实际 body 位置/朝向，相对 robot anchor frame
    # 对齐 observations.py robot_body_pos_b / robot_body_ori_b
    robot_body_pos_w  = np.stack([
        get_body_xpos_xquat(model, data, name)[0] for name in TRACKED_BODY_NAMES
    ])   # [14,3]
    robot_body_quat_w = np.stack([
        get_body_xpos_xquat(model, data, name)[1] for name in TRACKED_BODY_NAMES
    ])   # [14,4]

    body_pos_b = np.stack([
        quat_rotate_inverse(robot_anchor_quat,
                            robot_body_pos_w[i] - robot_anchor_pos)
        for i in range(NUM_BODIES)
    ])   # [14,3]
    body_quat_b = np.stack([
        quat_mul(quat_inv(robot_anchor_quat), robot_body_quat_w[i])
        for i in range(NUM_BODIES)
    ])   # [14,4]
    body_ori_b = np.stack([rot6d(body_quat_b[i]) for i in range(NUM_BODIES)])  # [14,6]

    # motion anchor obs（ref anchor 相对 robot anchor frame）
    ref_anchor_pos  = ref["body_pos_w"][ANCHOR_IDX_IN_TRACKED]
    ref_anchor_quat = ref["body_quat_w"][ANCHOR_IDX_IN_TRACKED]
    motion_anchor_pos_b = quat_rotate_inverse(robot_anchor_quat,
                                               ref_anchor_pos - robot_anchor_pos)
    motion_anchor_ori_b = rot6d(quat_mul(quat_inv(robot_anchor_quat), ref_anchor_quat))

    # 关节状态：MuJoCo xml 顺序 → 训练顺序
    # ids[i] = 训练第i个关节在MuJoCo xml中的下标
    # 所以 q_train[i] = q_xml[ids[i]]
    ids = deploy.joint_ids_map
    q_xml  = data.qpos[7:7+deploy.num_joints].copy().astype(np.float32)
    dq_xml = data.qvel[6:6+deploy.num_joints].copy().astype(np.float32)
    joint_pos  = q_xml[ids]    # 训练顺序
    joint_vel  = dq_xml[ids]
    joint_pos_rel = joint_pos - deploy.default_joint_pos

    # motion_command（训练顺序）
    # npz joint_pos 已经是训练关节顺序（IsaacLab 顺序）
    motion_command = np.concatenate([ref["joint_pos"], ref["joint_vel"]])   # [58]

    obs = np.concatenate([
        motion_command,           # 58
        motion_anchor_pos_b,      # 3
        motion_anchor_ori_b,      # 6
        body_pos_b.flatten(),     # 42
        body_ori_b.flatten(),     # 84
        base_lin_vel,             # 3
        base_ang_vel,             # 3
        joint_pos_rel,            # 29
        joint_vel,                # 29
        last_action,              # 29
    ])
    return obs.astype(np.float32)   # 286


# ══════════════════════════════════════════════════════════════════
# Policy 加载（TorchScript）
# ══════════════════════════════════════════════════════════════════

def load_policy(path: str, obs_dim: int):
    policy = torch.jit.load(path, map_location="cpu")
    policy.eval()
    with torch.no_grad():
        out = policy(torch.zeros(1, obs_dim))
    print(f"[Policy] 加载成功: in={obs_dim}, out={out.shape[1]}")
    return policy


# ══════════════════════════════════════════════════════════════════
# 仿真初始化 & 循环
# ══════════════════════════════════════════════════════════════════

def reset_to_frame(model, data, deploy: DeployConfig, ref: dict):
    """用参考帧初始化仿真状态"""
    mujoco.mj_resetData(model, data)
    ids = deploy.joint_ids_map

    # 关节：训练顺序 → MuJoCo xml 顺序（先设关节，再算 forward kinematics）
    q_xml  = np.zeros(deploy.num_joints, dtype=np.float32)
    dq_xml = np.zeros(deploy.num_joints, dtype=np.float32)
    q_xml[ids]  = ref["joint_pos"]
    dq_xml[ids] = ref["joint_vel"]
    data.qpos[7:7+deploy.num_joints] = q_xml
    data.qvel[6:6+deploy.num_joints] = dq_xml

    # root 朝向：qpos[3:7] 是 pelvis 的四元数
    # ref 给的是 torso_link 的四元数，需要反推 pelvis 四元数
    # pelvis_quat = ref_torso_quat * inv(local_torso_quat_in_pelvis_frame)
    # 用 FK：先以单位朝向做 forward，得到 torso 相对 pelvis 的局部旋转
    data.qpos[3:7] = [1., 0., 0., 0.]   # 先设 pelvis 为单位朝向
    data.qpos[:3]  = [0., 0., 0.]
    mujoco.mj_forward(model, data)

    torso_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ANCHOR_BODY)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    # torso 相对 pelvis 的局部四元数（pelvis 为单位时就是世界系的 torso quat）
    torso_local_quat  = data.xquat[torso_id].copy()    # (w,x,y,z)
    torso_local_pos   = data.xpos[torso_id].copy()
    pelvis_local_pos  = data.xpos[pelvis_id].copy()
    pelvis_to_torso_local = torso_local_pos - pelvis_local_pos

    # 目标 torso 朝向
    ref_anchor_quat = ref["body_quat_w"][ANCHOR_IDX_IN_TRACKED]
    ref_anchor_pos  = ref["body_pos_w"][ANCHOR_IDX_IN_TRACKED]

    # pelvis_quat = ref_torso_quat * inv(torso_local_quat)
    pelvis_quat = quat_mul(ref_anchor_quat, quat_inv(torso_local_quat))
    data.qpos[3:7] = pelvis_quat

    # pelvis 位置：ref_torso_pos - rotate(pelvis_quat, pelvis_to_torso_local)
    torso_offset_world = quat_apply(pelvis_quat, pelvis_to_torso_local)
    data.qpos[:3] = ref_anchor_pos - torso_offset_world
    mujoco.mj_forward(model, data)


def run(args: argparse.Namespace):
    # ── 加载配置 ─────────────────────────────────────────────────
    deploy = DeployConfig(args.deploy)
    NUM_JOINTS = deploy.num_joints

    # ── 加载 MuJoCo 模型 ─────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(args.model)
    data  = mujoco.MjData(model)
    model.opt.timestep = SIM_DT
    print(f"[MuJoCo] nu={model.nu}, nbody={model.nbody}")
    if model.nu != NUM_JOINTS:
        print(f"[Warning] MuJoCo nu={model.nu} != deploy num_joints={NUM_JOINTS}")

    # ── 对齐 IsaacLab 训练时的物理参数（可选）─────────────────────
    if args.align_physics:
        model.opt.iterations    = 100
        model.opt.ls_iterations = 50
        model.opt.tolerance     = 1e-10

        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id >= 0:
            model.geom_friction[floor_id] = [1.0, 0.005, 0.0001]
            model.geom_solref[floor_id]   = [0.005, 1.0]
            model.geom_solimp[floor_id]   = [0.9, 0.95, 0.001, 0.5, 2.0]

        for name in ["left_ankle_roll_link", "right_ankle_roll_link"]:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] == bid:
                    model.geom_friction[gid] = [1.0, 0.005, 0.0001]
                    model.geom_solref[gid]   = [0.005, 1.0]
                    model.geom_solimp[gid]   = [0.9, 0.95, 0.001, 0.5, 2.0]

        model.dof_damping[6:6+NUM_JOINTS] = 0.0
        print("[Physics] IsaacLab 物理对齐已启用（摩擦/接触/damping）")
    else:
        print("[Physics] 使用 MuJoCo 默认物理参数（加 --align-physics 可对齐 IsaacLab）")

    # armature 始终设置（训练时一定有，不受默认参数影响）
    # ── 加载参考动作 ─────────────────────────────────────────────
    body_indexes = build_body_indexes(model)
    motion       = MotionClip(args.motion, body_indexes)

    # ── 验证 obs_dim ─────────────────────────────────────────────
    obs_dim = 286
    print(f"[Obs] obs_dim={obs_dim}")

    # ── 加载 policy ──────────────────────────────────────────────
    policy = load_policy(args.policy, obs_dim)

    # ── 初始化仿真 ────────────────────────────────────────────────
    ref0 = motion.get_frame(0.0)
    reset_to_frame(model, data, deploy, ref0)

    # PD 增益（训练顺序）
    # deploy.yaml 的 stiffness 顺序可能和 joint_ids_map 不对齐
    # 改为从关节名按类型推导，确保正确
    KP  = deploy.stiffness.copy()
    KD  = deploy.damping.copy()
    ids = deploy.joint_ids_map

    XML_JOINT_NAMES = [
        'left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint',
        'left_knee_joint','left_ankle_pitch_joint','left_ankle_roll_joint',
        'right_hip_pitch_joint','right_hip_roll_joint','right_hip_yaw_joint',
        'right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint',
        'waist_yaw_joint','waist_roll_joint','waist_pitch_joint',
        'left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint',
        'left_elbow_joint','left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint',
        'right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint',
        'right_elbow_joint','right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint',
    ]
    # 按关节名推导正确的 KP/KD
    KP_MAP = {
        'hip_pitch': 40.2,  'hip_yaw': 40.2,   'waist_yaw': 40.2,
        'waist_roll': 28.5, 'waist_pitch': 28.5,
        'hip_roll': 99.1,   'knee': 99.1,
        'ankle': 28.5,
        'shoulder': 14.3,   'elbow': 14.3,      'wrist': 14.3,
    }
    KD_MAP = {
        'hip_pitch': 2.56,  'hip_yaw': 2.56,    'waist_yaw': 2.56,
        'waist_roll': 1.81, 'waist_pitch': 1.81,
        'hip_roll': 6.31,   'knee': 6.31,
        'ankle': 1.81,
        'shoulder': 0.907,  'elbow': 0.907,     'wrist': 0.907,
    }
    def _lookup(name, table):
        for k, v in table.items():
            if k in name:
                return v
        return None

    KP_correct = np.zeros(NUM_JOINTS, dtype=np.float32)
    KD_correct = np.zeros(NUM_JOINTS, dtype=np.float32)
    print("[KP/KD 修正] train_i | joint | KP_yaml→KP_correct | KD_yaml→KD_correct")
    for train_i, xml_i in enumerate(ids):
        jname = XML_JOINT_NAMES[xml_i]
        kp_c = _lookup(jname, KP_MAP)
        kd_c = _lookup(jname, KD_MAP)
        if kp_c is None:
            kp_c = KP[train_i]; kd_c = KD[train_i]
            print(f"  [{train_i:2d}] {jname} → 未匹配，保留 yaml 值")
        else:
            if abs(kp_c - KP[train_i]) > 0.5:
                print(f"  [{train_i:2d}] {jname:35s} KP {KP[train_i]:.1f}→{kp_c:.1f}  KD {KD[train_i]:.3f}→{kd_c:.3f}  ← 修正")
        KP_correct[train_i] = kp_c
        KD_correct[train_i] = kd_c

    KP = KP_correct
    KD = KD_correct

    # 力矩限幅：从 URDF actuatorfrcrange 读取（xml 顺序），按 ids 映射到训练顺序
    _FRCRANGE_XML = np.array([
        88, 139, 88, 139, 50, 50,    # left leg: pitch,roll,yaw,knee,ankle_pitch,ankle_roll
        88, 139, 88, 139, 50, 50,    # right leg
        88, 50, 50,                  # waist: yaw,roll,pitch
        25, 25, 25, 25, 25, 5, 5,    # left arm: shoulder_pitch,roll,yaw,elbow,wrist_roll,pitch,yaw
        25, 25, 25, 25, 25, 5, 5,    # right arm
    ], dtype=np.float32)
    EFFORT_LIMITS = _FRCRANGE_XML[ids]   # [29] 训练顺序
    print(f"[Config] EFFORT_LIMITS[:6] = {EFFORT_LIMITS[:6]}")

    def _pd_warmup(target_train, n_ctrl_steps=10):
        """用固定目标姿态做 PD hold，让接触力稳定后再跑 policy"""
        for _ in range(n_ctrl_steps):
            for _ in range(DECIMATION):
                q_tr  = data.qpos[7:7+NUM_JOINTS].astype(np.float32)[ids]
                dq_tr = data.qvel[6:6+NUM_JOINTS].astype(np.float32)[ids]
                tau   = np.clip(KP*(target_train - q_tr) - KD*dq_tr,
                                -EFFORT_LIMITS, EFFORT_LIMITS)
                torque_xml = np.zeros(NUM_JOINTS, dtype=np.float32)
                torque_xml[ids] = tau
                data.ctrl[:] = torque_xml
                mujoco.mj_step(model, data)

    print("[Sim] warmup 中...")
    _pd_warmup(ref0["joint_pos"])   # 用第0帧关节角稳定初始姿态

    last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
    sim_time    = 0.0
    step_count  = 0

    renderer = None
    frames   = []
    if args.record:
        renderer = mujoco.Renderer(model, height=720, width=1280)

    def _step():
        nonlocal last_action, sim_time, step_count

        if args.static_hold:
            act = np.zeros(NUM_JOINTS, dtype=np.float32)
            target_train = deploy.default_joint_pos.copy()
        else:
            obs = build_observation(model, data, motion, deploy, sim_time, last_action)
            with torch.no_grad():
                act = policy(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()
            target_train = deploy.action_offset + act * deploy.action_scale
            lo = model.jnt_range[1:1+NUM_JOINTS, 0][ids]
            hi = model.jnt_range[1:1+NUM_JOINTS, 1][ids]
            target_train = np.clip(target_train, lo, hi)

        # 第一步打印调试信息
        if step_count == 0:
            q_xml0  = data.qpos[7:7+NUM_JOINTS].astype(np.float32)
            q_tr0   = q_xml0[ids]
            err0    = target_train - q_tr0
            tau0    = KP * err0
            print(f"[Debug] target_train[:6] = {target_train[:6]}")
            print(f"[Debug] q_train[:6]      = {q_tr0[:6]}")
            print(f"[Debug] error[:6]        = {err0[:6]}")
            print(f"[Debug] torque(KP*err)[:6] = {tau0[:6]}")

        for _ in range(DECIMATION):
            # 当前关节状态：MuJoCo xml 顺序 → 训练顺序
            # ids[i] = 训练第i个关节在MuJoCo中的下标
            q_xml  = data.qpos[7:7+NUM_JOINTS].astype(np.float32)
            dq_xml = data.qvel[6:6+NUM_JOINTS].astype(np.float32)
            q_train  = q_xml[ids]    # q_train[i]  = q_xml[ids[i]]
            dq_train = dq_xml[ids]

            # PD 力矩（训练顺序计算）
            torque_train = KP * (target_train - q_train) - KD * dq_train
            # 力矩限幅（对齐 IsaacLab effort_limit）
            torque_train = np.clip(torque_train, -EFFORT_LIMITS, EFFORT_LIMITS)

            # 写回 MuJoCo ctrl：训练顺序 → MuJoCo xml 顺序
            # data.ctrl[ids[i]] = torque_train[i]
            torque_xml = np.zeros(NUM_JOINTS, dtype=np.float32)
            torque_xml[ids] = torque_train
            data.ctrl[:] = torque_xml
            mujoco.mj_step(model, data)
            sim_time += SIM_DT

        last_action = act.copy()
        step_count += 1

        if renderer is not None:
            renderer.update_scene(data, camera=0)
            frames.append(renderer.render().copy())

    def _do_reset():
        nonlocal sim_time, last_action
        print("[Sim] episode 结束，重置...")
        ref0 = motion.get_frame(0.0)
        reset_to_frame(model, data, deploy, ref0)
        sim_time = 0.0
        last_action[:] = 0.0
        _pd_warmup(ref0["joint_pos"])

    if args.render:
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ANCHOR_BODY)
        motion_duration = motion.T * motion.motion_dt
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = torso_id
            viewer.cam.distance    = 3.5
            viewer.cam.elevation   = -15.0
            print("[Sim] viewer 已启动，按 ESC 退出...")
            while viewer.is_running():
                if sim_time >= motion_duration:
                    print("[Sim] motion 播放完毕，退出。")
                    break
                t0 = time.time()
                _step()
                viewer.sync()
                sleep = CTRL_DT - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)
    else:
        total = int(motion.T * motion.motion_dt / (SIM_DT * DECIMATION))
        print(f"[Sim] headless，{total} 步...")
        for _ in range(total):
            _step()
        print(f"[Sim] 完成，{step_count} policy 步。")

    if args.record and frames:
        try:
            import imageio
            imageio.mimwrite(args.record, frames, fps=int(1.0/(SIM_DT*DECIMATION)))
            print(f"[Record] 保存: {args.record}")
        except ImportError:
            print("[Record] pip install imageio[ffmpeg]")


CTRL_DT = SIM_DT * DECIMATION

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True, help="MuJoCo XML 路径")
    p.add_argument("--policy", required=True, help="exported/policy.pt 路径")
    p.add_argument("--deploy", required=True, help="exported/deploy.yaml 路径")
    p.add_argument("--motion", required=True, help="front_flip.npz 路径")
    p.add_argument("--render", action="store_true")
    p.add_argument("--record", default=None,  help="录制视频路径")
    p.add_argument("--align-physics", action="store_true",
                   help="对齐 IsaacLab 物理参数（地面摩擦/接触硬度/joint damping清零）")
    p.add_argument("--static-hold", action="store_true",
                   help="调试模式：只用 default_joint_pos 做 PD 站立，不跑 policy")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())