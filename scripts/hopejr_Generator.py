import numpy as np
import os
import sys
import time
import keyboard
import mujoco
import mujoco.viewer
import pickle
import matplotlib.pyplot as plt
from collections import deque

from hopejr_Optimizer_SLSQP import Optimizer

class HopeJrArm:
    def __init__(self, model_path, q0=None):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        if q0 is None:
            q0 = np.array([0.0, 0.0, 0.0, 0.0, 1.6, 0.0, 0.0])
        self.q_init = q0
        self.data.qpos = self.q_init
        self.data.ctrl = self.q_init
        mujoco.mj_forward(self.model, self.data)

        self.target_pos = None
        self.target_pose = None
        self.obstacle_poses = None
        self.task_defined = False
        self.viewer = None
        self.trajectory_points = deque(maxlen=500)

        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        self.obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj")

    def fk_position(self, q):
        if self.model is None or self.data is None:
            raise ValueError("Mujoco model is not loaded")
        
        current_qpos = self.data.qpos.copy()
        
        self.data.qpos[:7] = q
        mujoco.mj_forward(self.model, self.data)
        
        ee_pos = self.data.xpos[self.ee_id].copy()
        
        self.data.qpos = current_qpos
        mujoco.mj_forward(self.model, self.data)
        
        return ee_pos

    def get_ee_pose(self, q):
        current_qpos = self.data.qpos.copy()
        
        self.data.qpos[:7] = q
        mujoco.mj_forward(self.model, self.data)
        
        ee_pos = self.data.xpos[self.ee_id].copy()
        ee_quat = self.data.xquat[self.ee_id].copy()  # [w, x, y, z]
        
        self.data.qpos = current_qpos
        mujoco.mj_forward(self.model, self.data)
        
        return np.concatenate([ee_pos, ee_quat])

    def get_obstacle_positions(self):
        positions = {}
        
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and 'obstacle' in body_name.lower():
                positions[body_name] = self.data.xpos[i].copy()
        
        return positions

    def initialize_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self.viewer
    
    def close_viewer(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def task_set(self, set_mode="object", with_orientation=False):
        viewer = self.initialize_viewer()
        def get_ee_pos(set_mode):
            match set_mode:
                case "object":
                    return self.data.xpos[self.obj_id].copy()
                case "joint":
                    return self.data.xpos[self.ee_id].copy()

        def record_reach():
            ee_pos = get_ee_pos(set_mode)
            self.obstacle_poses = self.get_obstacle_positions()
            self.target_pos = ee_pos.copy()

            if with_orientation:
                current_q = self.data.qpos[:7].copy()
                self.target_pose = self.get_ee_pose(current_q)
                print(f"Target position: {self.target_pos}")
                print(f"Target orientation: {self.target_pose[3:7]}")
            else:
                self.target_pose = None
                print(f"Target position: {self.target_pos}")

            self.task_defined = True

        print("-Please press 'r' to record target pose-")
        keyboard.on_press_key("r", lambda _: record_reach())

        while viewer.is_running() and not self.task_defined:
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(0.01)

        keyboard.unhook_all()
        print("Task setting complete!")

        return self.target_pos, self.obstacle_poses, self.task_defined

    def save_trajectory(self, q_trajectory, save_path=None):
        if save_path is None:
            save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Trajectory')
            os.makedirs(save_path, exist_ok=True)

        data = {'target_pos': self.target_pos, 'obstacle_pos': self.obstacle_poses, 'trajectory': q_trajectory}
        complete_save_path = os.path.join(save_path, "trajectory.pkl")
        with open(complete_save_path, 'wb') as f:
            pickle.dump(data, f)

    def show_trajectory(self, positions):
        if not positions:
            return
        
        self.viewer.user_scn.ngeom = 0

        for i, pos in enumerate(positions):
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0, 0],
                pos = pos,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 1]
            )
        self.viewer.user_scn.ngeom = i

    def simulate_trajectory(self, trajectory, dt=0.04):
        self.data.qpos = trajectory[0]
        self.data.ctrl = trajectory[0]
        mujoco.mj_forward(self.model, self.data)
        self.data.mocap_pos[0] = self.target_pos

        i = 1
        for obstacle_name, obstacle_pos in self.obstacle_poses.items():
            self.data.mocap_pos[i] = obstacle_pos
            i += 1

        self.trajectory_points.clear()
        time.sleep(0.5)

        viewer = self.viewer
        if viewer is None or not viewer.is_running():
            viewer = self.initialize_viewer()

        for t in range(len(trajectory)):
            if not viewer.is_running():
                break

            self.data.ctrl[:7] = trajectory[t]
            ee_pos = self.data.xpos[self.ee_id].copy()
            self.trajectory_points.append(ee_pos.copy())
            self.show_trajectory(self.trajectory_points)

            start_time = time.time()
            while time.time() - start_time < dt:
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.01)

        print(f"Trajectory simulate successfully!")
        start_time = time.time()
        while time.time() - start_time < 1:
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(0.01)
        
    def simulate_over(self):
        viewer = self.viewer
        print("-----------------------------")
        print("Trajectory runnning complete! Press 'esc' to exit...")
        keyboard.on_press_key("esc", lambda _: self.close_viewer())

        while viewer.is_running():
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(0.01)

    def visualize_trajectory(self, q_trajectory=None):
        if q_trajectory is None:
            raise ValueError("Trajectory can not be empty")
        
        mujoco.mj_resetData(self.model, self.data)
        qpos = q_trajectory

        plt.figure(figsize=(15, 10))

        #joint position
        plt.subplot(4, 1, 1)
        for i in range(7):
            plt.plot(range(len(qpos)), qpos[:, i], label=f'Joint {i+1}')
        plt.title('Joint Positions')
        plt.xlabel('Time Steps')
        plt.ylabel('Joint Angle (rad)')
        plt.legend()
        plt.grid(True)

        #joint velocity
        plt.subplot(4, 1, 2)
        qvel = np.zeros((len(qpos)-1, 7))
        for t in range(len(qpos)-1):
            qvel[t] = qpos[t+1] - qpos[t]
        for i in range(7):
            qvel_profile = qvel[:, i]
            plt.plot(range(len(qvel)), qvel_profile, label=f'Joint {i+1} velocity')
        plt.title('Velocity Profiles')
        plt.xlabel('Time Steps')
        plt.ylabel('Velocity')
        plt.legend()
        plt.grid(True)
        
        #joint acceleration
        plt.subplot(4, 1, 3)
        qacc = np.zeros((len(qvel)-1, 7))
        for t in range(len(qvel)-1):
            qacc[t] = qvel[t+1] - qvel[t]
        for i in range(7):
            qacc_profile = qacc[:, i]
            plt.plot(range(len(qacc)), qacc_profile, label=f'Joint {i+1} acceleration')
        plt.title('Acceleration Profiles')
        plt.xlabel('Time Steps')
        plt.ylabel('Acceleration')
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 4)
        ee_positions = np.zeros((len(qpos), 3))
        for t in range(len(qpos)):
            ee_positions[t] = np.asarray(self.fk_position(qpos[t]))
        ee_vel = np.zeros((len(qvel), 3))
        for t in range(len(qvel)):
            ee_vel[t] = ee_positions[t+1] - ee_positions[t]
        ee_vel_norm = np.linalg.norm(ee_vel, axis=1)
        plt.plot(range(len(ee_vel_norm)), ee_vel_norm, 'r-', linewidth=3, label='End-effector velocity')
        plt.title('End-Effector Velocity Profile')
        plt.xlabel('Time Steps')
        plt.ylabel('Vel')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"trajectory_visualization.png")
        print("Figure saved!")
        plt.show()

def main():
    project_root_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(project_root_path,'MujocoModel', 'scene.xml')

    N = 30
    dt = 0.3
    with_orientation = False
    use_synergies = False
    
    #Initialize mujoco
    hopejr = HopeJrArm(model_path)
    hopejr.task_set(set_mode="object", with_orientation=with_orientation)
    if hopejr.task_defined:
        print("Trajectory optimizing...")
        target_pos = hopejr.target_pos
        target_pose = hopejr.target_pose
        q_start = hopejr.q_init

        optimizer = Optimizer(
            timestep=N,
            q0=q_start,
            mujoco_model=hopejr.model,
            mujoco_data=hopejr.data,
            use_synergies=use_synergies,
            position_only=True)
        if hopejr.target_pose is not None:
            q_trajectory = optimizer.optimize(q0=q_start, EE_target=target_pos, target_pose=target_pose, init_method="cartesian_rrt")
        else:
            q_trajectory = optimizer.optimize(q0=q_start, EE_target=target_pos, init_method="cartesian_rrt")
    else:
        print("TASK DEFINED FAILED, TERMINATED")
        sys.exit()
 
    hopejr.save_trajectory(q_trajectory=q_trajectory)
    print("Trajectory saved!")
    hopejr.visualize_trajectory(q_trajectory=q_trajectory)
    print("Ready to simulate...")
    hopejr.simulate_trajectory(trajectory=q_trajectory, dt=dt)
    hopejr.simulate_over()
        
if __name__ == "__main__":
    main()