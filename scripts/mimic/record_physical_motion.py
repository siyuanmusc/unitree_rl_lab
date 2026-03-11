"""
record_physical_motion.py — 从 MuJoCo 仿真录制物理合理的 reference motion
==========================================================================
用已训练的 policy 跑一遍 MuJoCo 仿真，录制机器人实际的物理状态，
存成和原始 front_flip.npz 格式完全一样的新 npz。

格式说明：
  joint_pos/vel [T, 29]  : MuJoCo xml 关节顺序（= IsaacLab robot.joint_names 顺序）
                           即 left_hip_pitch[0]...right_wrist_yaw[28]
                           和 csv2npz 存的 robot.data.joint_pos 顺序一致
  body_pos_w    [T,37,3] : IsaacLab robot.body_names 顺序（已通过实际打印确认）
  body_quat_w   [T,37,4] : 同上，四元数 (w,x,y,z)

用法:
  python record_physical_motion.py \\
      --model   /path/to/g1_29dof_rev_1_0.xml \\
      --policy  /path/to/exported/policy.pt \\
      --deploy  /path/to/exported/deploy.yaml \\
      --motion  /path/to/front_flip.npz \\
      --output  /path/to/front_flip_physical.npz \\
      [--align-physics]
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import mujoco

sys.path.insert(0, str(Path(__file__).parent))
from sim2mujoco_g1_frontflip import (
    SIM_DT, DECIMATION,
    DeployConfig, MotionClip, build_body_indexes,
    build_observation, reset_to_frame, load_policy,
)

CTRL_DT = SIM_DT * DECIMATION
NPZ_SIZE = 37

# npz body index → MuJoCo body id (0-based, xpos[mj_id+1])
# 顺序来自 IsaacLab robot.body_names 实际打印（csv2npz 日志确认）：
#   0:pelvis 1:left_hip_pitch 2:pelvis_contour 3:right_hip_pitch
#   4:waist_yaw 5:left_hip_roll 6:right_hip_roll 7:waist_roll
#   8:left_hip_yaw 9:right_hip_yaw 10:torso 11:left_knee 12:right_knee
#   13:head 14:left_shoulder_pitch 15:logo 16:right_shoulder_pitch
#   17:left_ankle_pitch 18:right_ankle_pitch 19:left_shoulder_roll
#   20:right_shoulder_roll 21:left_ankle_roll 22:right_ankle_roll
#   23:left_shoulder_yaw 24:right_shoulder_yaw 25:LL_FOOT 26:LR_FOOT
#   27:left_elbow 28:right_elbow 29:left_wrist_roll 30:right_wrist_roll
#   31:left_wrist_pitch 32:right_wrist_pitch 33:left_wrist_yaw
#   34:right_wrist_yaw 35:left_rubber_hand 36:right_rubber_hand
#
# MuJoCo body id (0-based):
#   0:pelvis 1:left_hip_pitch 2:left_hip_roll 3:left_hip_yaw 4:left_knee
#   5:left_ankle_pitch 6:left_ankle_roll 7:right_hip_pitch 8:right_hip_roll
#   9:right_hip_yaw 10:right_knee 11:right_ankle_pitch 12:right_ankle_roll
#   13:waist_yaw 14:waist_roll 15:torso_link
#   16:left_shoulder_pitch 17:left_shoulder_roll 18:left_shoulder_yaw
#   19:left_elbow 20:left_wrist_roll 21:left_wrist_pitch 22:left_wrist_yaw
#   23:right_shoulder_pitch 24:right_shoulder_roll 25:right_shoulder_yaw
#   26:right_elbow 27:right_wrist_roll 28:right_wrist_pitch 29:right_wrist_yaw
#
# -1 = MuJoCo 无此 body，用 NPZ_FALLBACK 里的替代
NPZ_TO_MJ = [
     0,   # [0]  pelvis
     1,   # [1]  left_hip_pitch_link
    -1,   # [2]  pelvis_contour_link       → pelvis(0)
     7,   # [3]  right_hip_pitch_link
    13,   # [4]  waist_yaw_link
     2,   # [5]  left_hip_roll_link
     8,   # [6]  right_hip_roll_link
    14,   # [7]  waist_roll_link
     3,   # [8]  left_hip_yaw_link
     9,   # [9]  right_hip_yaw_link
    15,   # [10] torso_link
     4,   # [11] left_knee_link
    10,   # [12] right_knee_link
    -1,   # [13] head_link                 → torso_link(15)
    16,   # [14] left_shoulder_pitch_link
    -1,   # [15] logo_link                 → torso_link(15)
    23,   # [16] right_shoulder_pitch_link
     5,   # [17] left_ankle_pitch_link
    11,   # [18] right_ankle_pitch_link
    17,   # [19] left_shoulder_roll_link
    24,   # [20] right_shoulder_roll_link
     6,   # [21] left_ankle_roll_link
    12,   # [22] right_ankle_roll_link
    18,   # [23] left_shoulder_yaw_link
    25,   # [24] right_shoulder_yaw_link
    -1,   # [25] LL_FOOT                   → left_ankle_roll_link(6)
    -1,   # [26] LR_FOOT                   → right_ankle_roll_link(12)
    19,   # [27] left_elbow_link
    26,   # [28] right_elbow_link
    20,   # [29] left_wrist_roll_link
    27,   # [30] right_wrist_roll_link
    21,   # [31] left_wrist_pitch_link
    28,   # [32] right_wrist_pitch_link
    22,   # [33] left_wrist_yaw_link
    29,   # [34] right_wrist_yaw_link
    -1,   # [35] left_rubber_hand          → left_wrist_yaw_link(22)
    -1,   # [36] right_rubber_hand         → right_wrist_yaw_link(29)
]

NPZ_FALLBACK = {
     2:  0,   # pelvis_contour_link → pelvis
    13: 15,   # head_link           → torso_link
    15: 15,   # logo_link           → torso_link
    25:  6,   # LL_FOOT             → left_ankle_roll_link
    26: 12,   # LR_FOOT             → right_ankle_roll_link
    35: 22,   # left_rubber_hand    → left_wrist_yaw_link
    36: 29,   # right_rubber_hand   → right_wrist_yaw_link
}


def build_npz_frame(data):
    """从 MuJoCo 当前状态构建一帧 37-body npz 数据"""
    bpos  = np.zeros((NPZ_SIZE, 3), dtype=np.float32)
    bquat = np.zeros((NPZ_SIZE, 4), dtype=np.float32)
    blvel = np.zeros((NPZ_SIZE, 3), dtype=np.float32)
    bavel = np.zeros((NPZ_SIZE, 3), dtype=np.float32)

    for npz_i, mj_id in enumerate(NPZ_TO_MJ):
        if mj_id == -1:
            mj_id = NPZ_FALLBACK[npz_i]
        bid = mj_id + 1           # MuJoCo xpos index（0=world）
        bpos[npz_i]  = data.xpos[bid]
        bquat[npz_i] = data.xquat[bid]   # (w,x,y,z)
        cvel = data.cvel[bid]             # [ang(0:3), lin(3:6)]
        blvel[npz_i] = cvel[3:]
        bavel[npz_i] = cvel[:3]

    return bpos, bquat, blvel, bavel


def run(args: argparse.Namespace):
    deploy = DeployConfig(args.deploy)
    NUM_JOINTS = deploy.num_joints
    ids = deploy.joint_ids_map

    # ── MuJoCo 模型 ──────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(args.model)
    data  = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

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
        print("[Physics] IsaacLab 物理对齐已启用")

    # ── 参考动作（原始，用于 obs 和初始化）──────────────────────
    body_indexes = build_body_indexes(model)
    motion = MotionClip(args.motion, body_indexes)
    fps = motion.fps
    total_steps = motion.T
    print(f"[Record] fps={fps}, T={total_steps}, duration={total_steps*motion.motion_dt:.2f}s")

    # ── Policy ───────────────────────────────────────────────────
    policy = load_policy(args.policy, 286)

    # ── KP/KD ────────────────────────────────────────────────────
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
    KP_MAP = {'hip_pitch':40.2,'hip_yaw':40.2,'waist_yaw':40.2,
              'waist_roll':28.5,'waist_pitch':28.5,'hip_roll':99.1,'knee':99.1,
              'ankle':28.5,'shoulder':14.3,'elbow':14.3,'wrist':14.3}
    KD_MAP = {'hip_pitch':2.56,'hip_yaw':2.56,'waist_yaw':2.56,
              'waist_roll':1.81,'waist_pitch':1.81,'hip_roll':6.31,'knee':6.31,
              'ankle':1.81,'shoulder':0.907,'elbow':0.907,'wrist':0.907}
    def _lookup(name, table):
        for k,v in table.items():
            if k in name: return v
        return None
    KP = np.array([_lookup(XML_JOINT_NAMES[xi], KP_MAP) or deploy.stiffness[i]
                   for i,xi in enumerate(ids)], dtype=np.float32)
    KD = np.array([_lookup(XML_JOINT_NAMES[xi], KD_MAP) or deploy.damping[i]
                   for i,xi in enumerate(ids)], dtype=np.float32)
    _FRCRANGE_XML = np.array([
        88,139,88,139,50,50, 88,139,88,139,50,50, 88,50,50,
        25,25,25,25,25,5,5,  25,25,25,25,25,5,5,
    ], dtype=np.float32)
    EFFORT_LIMITS = _FRCRANGE_XML[ids]

    # ── Warmup ───────────────────────────────────────────────────
    ref0 = motion.get_frame(0.0)
    reset_to_frame(model, data, deploy, ref0)
    for _ in range(10):
        for _ in range(DECIMATION):
            q_tr  = data.qpos[7:7+NUM_JOINTS].astype(np.float32)[ids]
            dq_tr = data.qvel[6:6+NUM_JOINTS].astype(np.float32)[ids]
            tau   = np.clip(KP*(ref0["joint_pos"]-q_tr)-KD*dq_tr,
                            -EFFORT_LIMITS, EFFORT_LIMITS)
            torque_xml = np.zeros(NUM_JOINTS, dtype=np.float32)
            torque_xml[ids] = tau
            data.ctrl[:] = torque_xml
            mujoco.mj_step(model, data)
    # warmup 后 reset 回第0帧，保证录制起点和原始 motion 对齐
    ref0 = motion.get_frame(0.0)
    reset_to_frame(model, data, deploy, ref0)
    mujoco.mj_forward(model, data)
    print("[Record] warmup 完成，reset 回第0帧，开始录制...")

    # ── 录制主循环 ───────────────────────────────────────────────
    rec_joint_pos  = []
    rec_joint_vel  = []
    rec_bpos       = []
    rec_bquat      = []
    rec_blvel      = []
    rec_bavel      = []

    sim_time    = 0.0
    last_action = np.zeros(NUM_JOINTS, dtype=np.float32)

    for step in range(total_steps):
        # ── 录制当前帧（policy 执行前）──────────────────────────
        # joint_pos/vel 存训练顺序（和原始 npz 一致，即 joint_ids_map 映射后）
        q_xml  = data.qpos[7:7+NUM_JOINTS].astype(np.float32)
        dq_xml = data.qvel[6:6+NUM_JOINTS].astype(np.float32)
        rec_joint_pos.append(q_xml[ids].copy())    # 训练顺序 [29]
        rec_joint_vel.append(dq_xml[ids].copy())   # 训练顺序 [29]

        bpos, bquat, blvel, bavel = build_npz_frame(data)
        rec_bpos.append(bpos)
        rec_bquat.append(bquat)
        rec_blvel.append(blvel)
        rec_bavel.append(bavel)

        # ── policy 步进 ──────────────────────────────────────────
        obs = build_observation(model, data, motion, deploy, sim_time, last_action)
        with torch.no_grad():
            act = policy(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()
        target = deploy.action_offset + act * deploy.action_scale
        lo = model.jnt_range[1:1+NUM_JOINTS, 0][ids]
        hi = model.jnt_range[1:1+NUM_JOINTS, 1][ids]
        target = np.clip(target, lo, hi)

        for _ in range(DECIMATION):
            q_tr  = data.qpos[7:7+NUM_JOINTS].astype(np.float32)[ids]
            dq_tr = data.qvel[6:6+NUM_JOINTS].astype(np.float32)[ids]
            tau   = np.clip(KP*(target-q_tr)-KD*dq_tr, -EFFORT_LIMITS, EFFORT_LIMITS)
            torque_xml = np.zeros(NUM_JOINTS, dtype=np.float32)
            torque_xml[ids] = tau
            data.ctrl[:] = torque_xml
            mujoco.mj_step(model, data)
            sim_time += SIM_DT

        last_action = act.copy()
        if (step+1) % 50 == 0:
            print(f"[Record] {step+1}/{total_steps} ({sim_time:.2f}s)")

    print(f"[Record] 录制完成，共 {total_steps} 帧")

    # ── 保存 npz ─────────────────────────────────────────────────
    out = {
        "fps":            np.array([fps], dtype=np.int64),
        "joint_pos":      np.stack(rec_joint_pos).astype(np.float32),   # [T,29] 训练顺序
        "joint_vel":      np.stack(rec_joint_vel).astype(np.float32),   # [T,29] 训练顺序
        "body_pos_w":     np.stack(rec_bpos).astype(np.float32),        # [T,37,3]
        "body_quat_w":    np.stack(rec_bquat).astype(np.float32),       # [T,37,4]
        "body_lin_vel_w": np.stack(rec_blvel).astype(np.float32),       # [T,37,3]
        "body_ang_vel_w": np.stack(rec_bavel).astype(np.float32),       # [T,37,3]
    }

    assert out["joint_pos"].shape  == (total_steps, 29)
    assert out["body_pos_w"].shape == (total_steps, NPZ_SIZE, 3)
    assert out["body_quat_w"].shape== (total_steps, NPZ_SIZE, 4)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **out)
    print(f"[Record] 已保存: {output_path}")
    for k,v in out.items():
        print(f"  {k}: {v.shape} {v.dtype}")

    # ── 和原始 motion 对比 ────────────────────────────────────────
    orig = np.load(args.motion)
    rmse = float(np.sqrt(np.mean((orig["joint_pos"] - out["joint_pos"])**2)))
    print(f"\n[Verify] joint_pos RMSE vs 原始: {rmse:.4f} rad")
    print(f"  越小说明 policy 跟踪越好")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/user/Workspace/Noetix_GMT/source/general_motion_tracking/general_motion_tracking/assets/unitree_description/mjcf/g1.xml")
    p.add_argument("--policy",        required=True)
    p.add_argument("--deploy",        required=True)
    p.add_argument("--motion",        required=True, help="原始 npz（用于 obs 和初始化）")
    p.add_argument("--output",        required=True, help="输出物理 motion npz 路径")
    p.add_argument("--align-physics", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())