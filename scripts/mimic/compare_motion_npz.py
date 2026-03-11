"""
compare_motion_npz.py — 单窗口并排对比两个 npz motion
=======================================================
左边(y=-0.8): 原始 motion（运动学，可能穿模）
右边(y=+0.8): 物理 motion（MuJoCo 录制，无穿模）

用法:
  python compare_motion_npz.py
  python compare_motion_npz.py --orig orig.npz --phys phys.npz
"""

from __future__ import annotations
import argparse, time
import numpy as np
import mujoco
import mujoco.viewer

# joint_ids_map: ids[train_i] = xml_i
IDS = [0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28]
MOTION_DT = 1.0 / 50


def make_two_robot_xml(xml_path: str, y_offset: float = 1.5) -> str:
    """在原始 xml 里再加一个偏移的机器人，所有 name 加 _B 后缀"""
    import re, tempfile

    with open(xml_path) as f:
        xml = f.read()

    # 找到 pelvis body 块（第一个 worldbody 下的顶级 body）
    # 用深度匹配找到完整的 robot body 块
    robot_start = xml.find('<body name="pelvis"')
    depth, i = 0, robot_start
    while i < len(xml):
        if xml[i:i+5] == '<body':
            depth += 1
        elif xml[i:i+7] == '</body>':
            depth -= 1
            if depth == 0:
                robot_end = i + 7
                break
        i += 1

    robot_block = xml[robot_start:robot_end]

    # 给第二个机器人所有 name 加 _B 后缀（避免冲突）
    robot_b = re.sub(r'((?:name|joint)=")([^"]+)(")',
                     lambda m: m.group(1) + m.group(2) + '_B' + m.group(3),
                     robot_block)

    # 修改 pelvis_B 的 y 位置
    robot_b = re.sub(
        r'(name="pelvis_B"\s+pos=")([^"]+)(")',
        lambda m: (
            m.group(1) +
            m.group(2).split()[0] + ' ' +
            str(float(m.group(2).split()[1]) + y_offset) + ' ' +
            m.group(2).split()[2] +
            m.group(3)
        ),
        robot_b
    )

    # 插入到 </worldbody> 之前
    insert_pos = xml.rfind('</worldbody>')
    new_xml = xml[:insert_pos] + '\n    <!-- Second robot for comparison -->\n' + robot_b + '\n' + xml[insert_pos:]

    import os as _os
    tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='w',
                                      dir=str(_os.path.dirname(_os.path.abspath(xml_path))), prefix='g1_two_')
    tmp.write(new_xml)
    tmp.close()
    return tmp.name


def run(args):
    orig_npz = np.load(args.orig)
    phys_npz = np.load(args.phys)
    T = min(int(orig_npz["joint_pos"].shape[0]),
            int(phys_npz["joint_pos"].shape[0]))
    print(f"[Compare] T={T} frames @ 50fps = {T/50:.1f}s")

    orig_jp   = orig_npz["joint_pos"].astype(np.float32)   # [T,29] 训练顺序
    phys_jp   = phys_npz["joint_pos"].astype(np.float32)
    orig_pos  = orig_npz["body_pos_w"][:, 0, :].astype(np.float32)   # pelvis pos
    orig_quat = orig_npz["body_quat_w"][:, 0, :].astype(np.float32)
    phys_pos  = phys_npz["body_pos_w"][:, 0, :].astype(np.float32)
    phys_quat = phys_npz["body_quat_w"][:, 0, :].astype(np.float32)

    # 构建双机器人 xml
    print("[Compare] 构建双机器人 xml...")
    tmp_xml = make_two_robot_xml(args.model, y_offset=1.5)
    print(f"[Compare] 临时 xml: {tmp_xml}")

    model = mujoco.MjModel.from_xml_path(tmp_xml)
    data  = mujoco.MjData(model)
    nj    = 29   # 每个机器人的关节数

    # 两个机器人的 qpos 布局:
    # robot A: qpos[0:7]   = free joint,  qpos[7:36]   = joints
    # robot B: qpos[36:43] = free joint,  qpos[43:72]  = joints
    QA_FREE  = 0
    QA_JOINT = 7
    QB_FREE  = 7 + nj
    QB_JOINT = 7 + nj + 7

    print("[Compare] 左=原始motion  右=物理motion  ESC退出")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance  = 5.0
        viewer.cam.elevation = -15.0
        viewer.cam.azimuth   = 90.0
        # 摄像机看向两个机器人中间
        viewer.cam.lookat[:] = [0.0, 0.75, 0.8]

        for frame in range(T):
            t0 = time.time()

            # Robot A: 原始 motion（左，y 不偏移）
            data.qpos[QA_FREE:QA_FREE+3]   = orig_pos[frame]
            data.qpos[QA_FREE+3:QA_FREE+7] = orig_quat[frame]
            q_xml = np.zeros(nj, dtype=np.float32)
            q_xml[IDS] = orig_jp[frame]
            data.qpos[QA_JOINT:QA_JOINT+nj] = q_xml

            # Robot B: 物理 motion（右，y+1.5）
            pos_b = phys_pos[frame].copy()
            pos_b[1] += 1.5
            data.qpos[QB_FREE:QB_FREE+3]   = pos_b
            data.qpos[QB_FREE+3:QB_FREE+7] = phys_quat[frame]
            q_xml = np.zeros(nj, dtype=np.float32)
            q_xml[IDS] = phys_jp[frame]
            data.qpos[QB_JOINT:QB_JOINT+nj] = q_xml

            mujoco.mj_forward(model, data)
            viewer.sync()

            if not viewer.is_running():
                break

            elapsed = time.time() - t0
            sleep = MOTION_DT - elapsed
            if sleep > 0:
                time.sleep(sleep)

        print(f"[Compare] 播放完成")

    import os
    os.unlink(tmp_xml)


def parse_args():
    G1_XML  = "/home/user/Workspace/Noetix_GMT/source/general_motion_tracking/general_motion_tracking/assets/unitree_description/mjcf/g1.xml"
    NPZ_DIR = "source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/front_flip"
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=G1_XML)
    p.add_argument("--orig",  default=f"{NPZ_DIR}/front_flip.npz")
    p.add_argument("--phys",  default=f"{NPZ_DIR}/front_flip_physical.npz")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())