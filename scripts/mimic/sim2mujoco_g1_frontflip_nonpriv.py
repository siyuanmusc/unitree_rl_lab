"""
sim2mujoco_g1_frontflip_nonpriv.py — 非特权obs版本（对应新训练cfg）
====================================================================
PolicyCfg obs (154维):
  motion_command      [58]  cat(ref_joint_pos, ref_joint_vel)
  motion_anchor_ori_b  [6]  rot6d
  base_ang_vel         [3]
  joint_pos_rel       [29]
  joint_vel_rel       [29]
  last_action         [29]
total = 154

用法:
  python sim2mujoco_g1_frontflip_nonpriv.py \\
      --policy logs/.../exported/policy.pt \\
      --deploy logs/.../exported/deploy.yaml \\
      --motion source/.../front_flip_physical.npz \\
      --render

python scripts/mimic/sim2mujoco_g1_frontflip_nonpriv.py \
  --policy logs/rsl_rl/unitree_g1_29dof_mimic_front_flip_no_privilege/2026-03-11_08-39-27/exported/policy.pt \
  --deploy logs/rsl_rl/unitree_g1_29dof_mimic_front_flip_no_privilege/2026-03-11_08-39-27/params/deploy.yaml \
  --motion source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/front_flip/front_flip_physical.npz \
  --render
"""

from __future__ import annotations
import argparse, time
from pathlib import Path
import sys

import numpy as np
import torch
import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).parent))
from sim2mujoco_g1_frontflip import (
    SIM_DT, DECIMATION, CTRL_DT,
    TRACKED_BODY_NAMES, ANCHOR_BODY, ANCHOR_IDX_IN_TRACKED,
    quat_inv, quat_mul, quat_rotate_inverse, rot6d,
    DeployConfig, MotionClip,
    get_body_xpos_xquat, get_body_cvel, build_body_indexes,
    reset_to_frame, load_policy,
)

OBS_DIM = 154


def build_observation_nonpriv(model, data, motion: MotionClip,
                               deploy: DeployConfig, sim_time: float,
                               last_action: np.ndarray) -> np.ndarray:
    """
    非特权obs，对齐新训练cfg的PolicyCfg (154维):
      motion_command      [58]
      motion_anchor_ori_b  [6]
      base_ang_vel         [3]
      joint_pos_rel       [29]
      joint_vel_rel       [29]
      last_action         [29]
    """
    ref = motion.get_frame(sim_time)

    # robot anchor (torso_link)
    robot_anchor_pos, robot_anchor_quat = get_body_xpos_xquat(model, data, ANCHOR_BODY)

    # motion_anchor_ori_b: ref anchor 朝向相对 robot anchor frame
    ref_anchor_quat = ref["body_quat_w"][ANCHOR_IDX_IN_TRACKED]
    motion_anchor_ori_b = rot6d(quat_mul(quat_inv(robot_anchor_quat), ref_anchor_quat))  # [6]

    # base_ang_vel (pelvis frame)
    base_pos, base_quat = get_body_xpos_xquat(model, data, "pelvis")
    _, base_ang_w = get_body_cvel(model, data, "pelvis")
    base_ang_vel = quat_rotate_inverse(base_quat, base_ang_w)  # [3]

    # joint state: xml顺序 → 训练顺序
    ids = deploy.joint_ids_map
    q_xml  = data.qpos[7:7+deploy.num_joints].astype(np.float32)
    dq_xml = data.qvel[6:6+deploy.num_joints].astype(np.float32)
    joint_pos_rel = q_xml[ids] - deploy.default_joint_pos   # [29]
    joint_vel     = dq_xml[ids]                              # [29]

    # motion_command
    motion_command = np.concatenate([ref["joint_pos"], ref["joint_vel"]])  # [58]

    obs = np.concatenate([
        motion_command,       # 58
        motion_anchor_ori_b,  # 6
        base_ang_vel,         # 3
        joint_pos_rel,        # 29
        joint_vel,            # 29
        last_action,          # 29
    ])
    assert len(obs) == OBS_DIM, f"obs dim {len(obs)} != {OBS_DIM}"
    return obs.astype(np.float32)


def run(args):
    deploy = DeployConfig(args.deploy)
    NUM_JOINTS = deploy.num_joints
    ids = deploy.joint_ids_map

    model = mujoco.MjModel.from_xml_path(args.model)
    data  = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    body_indexes = build_body_indexes(model)
    motion = MotionClip(args.motion, body_indexes)

    policy = load_policy(args.policy, OBS_DIM)

    # KP/KD
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

    # 初始化
    ref0 = motion.get_frame(0.0)
    reset_to_frame(model, data, deploy, ref0)

    # warmup
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
    print("[Sim] warmup 完成")

    last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
    sim_time    = 0.0
    motion_duration = motion.T * motion.motion_dt

    def _step():
        nonlocal last_action, sim_time
        obs = build_observation_nonpriv(model, data, motion, deploy,
                                        sim_time, last_action)
        with torch.no_grad():
            act = policy(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()
        target = deploy.action_offset + act * deploy.action_scale
        lo = model.jnt_range[1:1+NUM_JOINTS, 0][ids]
        hi = model.jnt_range[1:1+NUM_JOINTS, 1][ids]
        target = np.clip(target, lo, hi)

        for _ in range(DECIMATION):
            q_xml  = data.qpos[7:7+NUM_JOINTS].astype(np.float32)
            dq_xml = data.qvel[6:6+NUM_JOINTS].astype(np.float32)
            tau = np.clip(KP*(target-q_xml[ids])-KD*dq_xml[ids],
                          -EFFORT_LIMITS, EFFORT_LIMITS)
            torque_xml = np.zeros(NUM_JOINTS, dtype=np.float32)
            torque_xml[ids] = tau
            data.ctrl[:] = torque_xml
            mujoco.mj_step(model, data)
            sim_time += SIM_DT

        last_action = act.copy()

    if args.render:
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ANCHOR_BODY)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = torso_id
            viewer.cam.distance    = 3.5
            viewer.cam.elevation   = -15.0
            print("[Sim] viewer 启动，ESC 退出")
            while viewer.is_running():
                if sim_time >= motion_duration:
                    print("[Sim] motion 播放完毕")
                    break
                t0 = time.time()
                _step()
                viewer.sync()
                sleep = CTRL_DT - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)
    else:
        total = int(motion_duration / CTRL_DT)
        for _ in range(total):
            _step()
        print(f"[Sim] headless 完成，sim_time={sim_time:.2f}s")


def parse_args():
    G1_XML = "/home/user/Workspace/Noetix_GMT/source/general_motion_tracking/general_motion_tracking/assets/unitree_description/mjcf/g1.xml"
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default=G1_XML)
    p.add_argument("--policy", required=True)
    p.add_argument("--deploy", required=True)
    p.add_argument("--motion", required=True)
    p.add_argument("--render", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())