import numpy as np
import time
import os
import mujoco
import mujoco.viewer
import keyboard
import pickle
from collections import deque
import glob
from pathlib import Path

class ArmReplay:
    def __init__(self, model_path, q0, dt=0.04, target_pos=None, obstacle_poses=None):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        if q0 is None:
            q0 = np.array([0, 0, 0, 0, 1.6, 0, 0])
        self.q_init = q0
        self.dt = dt
        self.data.qpos = self.q_init
        self.data.ctrl = self.q_init
        mujoco.mj_forward(self.model, self.data)

        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")

        if target_pos is not None:
            self.data.mocap_pos[0] = target_pos
        if obstacle_poses is not None:
            i = 1
            for obstacle_name, obstacle_pos in obstacle_poses.items():
                self.data.mocap_pos[i] = obstacle_pos
                i += 1

        self.viewer = None
        self.replaying = False

    def get_position(self):
        ee_pos = self.data.xpos[self.ee_id].copy()
        return ee_pos

    def show_trajectory(self, viewer, positions):
        if not positions:
            return
        
        viewer.user_scn.ngeom = 0
        for i, pos in enumerate(positions):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.015, 0, 0],
                pos = pos,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 1]
            )
        viewer.user_scn.ngeom = i

    def replay(self, q_traj):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            trajectory_points = deque(maxlen=500)
            r_pressed = False
            while viewer.is_running():
                if keyboard.is_pressed('r'):
                    if not r_pressed and not self.replaying:
                        self.replaying = True
                        self.data.qpos = q_traj[0]
                        self.data.ctrl = q_traj[0]
                        self._replay_trajectories(viewer, q_traj, trajectory_points)
                        self.replaying = False
                        print("--------------------")
                    r_pressed = True
                else:
                    r_pressed = False
                
                if keyboard.is_pressed('q'):
                    break
                
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.01)

    def _replay_trajectories(self, viewer, q_traj, trajectory_points):
        trajectory_points.clear()
        
        for t in range(len(q_traj)):
            if not viewer.is_running():
                return
            
            ee_pos = self.get_position()
            trajectory_points.append(ee_pos.copy())
            self.show_trajectory(viewer, trajectory_points)
            viewer.sync()
            self.data.ctrl[:7] = q_traj[t]
            start_time = time.time()
            while time.time() - start_time < self.dt:
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.005)
        print(f"Trajectory replayed successfully!")

        start_time = time.time()
        while time.time() - start_time < 1:
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(0.01)


class DatasetTaskSelector:
    def __init__(self, dataset_root_path):
        self.dataset_root_path = Path(dataset_root_path)
        self.available_datasets = self._find_datasets()
        self.current_dataset = None
        self.trajectories = []
    
    def _find_datasets(self):
        """查找所有可用的数据集文件夹"""
        datasets = []
        for folder in self.dataset_root_path.glob("dataset_*"):
            if folder.is_dir():
                # 查找trajectories_final.pkl文件
                pkl_file = folder / "trajectories_final.pkl"
                if pkl_file.exists():
                    datasets.append(folder)
        return sorted(datasets)
    
    def list_datasets(self):
        """列出所有可用的数据集"""
        print("Available datasets:")
        print("-" * 50)
        for i, dataset_folder in enumerate(self.available_datasets):
            print(f"{i+1:2d}. {dataset_folder.name}")
            
            # 尝试读取metadata显示更多信息
            metadata_file = dataset_folder / "metadata.json"
            if metadata_file.exists():
                try:
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    total_trajs = metadata['generation_info']['total_trajectories']
                    success_rate = metadata['generation_info']['success_rate']
                    print(f"     轨迹数量: {total_trajs}, 成功率: {success_rate:.2%}")
                except:
                    pass
        print("-" * 50)
    
    def select_dataset(self, dataset_idx=None):
        """选择数据集"""
        if not self.available_datasets:
            print("No datasets found!")
            return False
            
        if dataset_idx is None:
            self.list_datasets()
            try:
                dataset_idx = int(input("Please select a dataset (number): ")) - 1
            except ValueError:
                print("Invalid input!")
                return False
        
        if 0 <= dataset_idx < len(self.available_datasets):
            selected_folder = self.available_datasets[dataset_idx]
            pkl_file = selected_folder / "trajectories_final.pkl"
            
            try:
                with open(pkl_file, 'rb') as f:
                    self.trajectories = pickle.load(f)
                self.current_dataset = selected_folder
                print(f"Dataset loaded: {selected_folder.name}")
                print(f"Total trajectories: {len(self.trajectories)}")
                return True
            except Exception as e:
                print(f"Failed to load dataset: {e}")
                return False
        else:
            print("Invalid dataset selection!")
            return False
    
    def list_trajectories(self, max_display=20):
        """列出当前数据集中的轨迹"""
        if not self.trajectories:
            print("No dataset selected!")
            return
            
        print(f"\nTrajectories in {self.current_dataset.name}:")
        print("-" * 80)
        print(f"{'ID':<4} {'Task ID':<15} {'Final Error':<12} {'Target Position':<20} {'Obstacle Position'}")
        print("-" * 80)
        
        display_count = min(len(self.trajectories), max_display)
        for i in range(display_count):
            traj = self.trajectories[i]
            task_id = traj.get('task_id', 'N/A')
            final_error = traj.get('final_error', 0.0)
            target_pos = traj.get('target_pos', [0, 0, 0])
            obstacle_pos = traj.get('obstacle_pos', [0, 0, 0])
            
            print(f"{i+1:<4} {task_id:<15} {final_error:<12.4f} "
                  f"[{target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}]    "
                  f"[{obstacle_pos[0]:.3f},{obstacle_pos[1]:.3f},{obstacle_pos[2]:.3f}]")
        
        if len(self.trajectories) > max_display:
            print(f"... and {len(self.trajectories) - max_display} more trajectories")
        print("-" * 80)
    
    def select_trajectory(self, traj_idx=None):
        if not self.trajectories:
            print("No dataset loaded!")
            return None
            
        if traj_idx is None:
            self.list_trajectories()
            try:
                traj_idx = int(input(f"Select a trajectory to replay (1-{len(self.trajectories)}): ")) - 1
            except ValueError:
                print("Invalid input!")
                return None
        
        if 0 <= traj_idx < len(self.trajectories):
            selected_traj = self.trajectories[traj_idx]
            print(f"\nSelected trajectory:")
            print(f"  Task ID: {selected_traj.get('task_id', 'N/A')}")
            print(f"  Final Error: {selected_traj.get('final_error', 0.0):.4f}")
            print(f"  Target Position: {selected_traj.get('target_pos', [0, 0, 0])}")
            print(f"  Obstacle Position: {selected_traj.get('obstacle_pos', [0, 0, 0])}")
            print(f"  Trajectory Length: {len(selected_traj.get('q_trajectory', []))}")
            
            return selected_traj
        else:
            print("Invalid trajectory selection!")
            return None


def interactive_main():
    project_root_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(project_root_path, 'MujocoModel', 'scene.xml')
    dataset_root_path = os.path.join(project_root_path, 'data', 'Dataset', 'Trajectory', 'dataset2','batch_001')
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    if not os.path.exists(dataset_root_path):
        print(f"Dataset directory not found: {dataset_root_path}")
        return
    
    selector = DatasetTaskSelector(dataset_root_path)
    
    if not selector.select_dataset():
        return
    
    selected_trajectory = selector.select_trajectory()
    if selected_trajectory is None:
        return
    
    q_trajectory = selected_trajectory['q_trajectory']
    q0 = q_trajectory[0]
    target_pos = selected_trajectory['target_pos']
    obstacle_pos = selected_trajectory['obstacle_pos']
    
    obstacle_poses = {"obstacle1": obstacle_pos}
    
    dt = 0.15
    panda = ArmReplay(model_path, q0, dt, target_pos, obstacle_poses)
    
    print("\n" + "="*50)
    print("Trajectory Replayer Ready!")
    print("Controls:")
    print("  Press 'r' - Replay the selected trajectory")
    print("  Press 'q' - Quit the viewer")
    print("="*50)
    
    panda.replay(q_trajectory)


def direct_replay(dataset_folder, trajectory_index):
    project_root_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(project_root_path, 'MujocoModel', 'scene.xml')
    
    pkl_file = Path(dataset_folder) / "trajectories_final.pkl"
    
    try:
        with open(pkl_file, 'rb') as f:
            trajectories = pickle.load(f)
        
        if 0 <= trajectory_index < len(trajectories):
            selected_trajectory = trajectories[trajectory_index]
            
            q_trajectory = selected_trajectory['q_trajectory']
            q0 = q_trajectory[0]
            target_pos = selected_trajectory['target_pos']
            obstacle_pos = selected_trajectory['obstacle_pos']
            obstacle_poses = {"obstacle1": obstacle_pos}
            
            dt = 0.15
            panda = ArmReplay(model_path, q0, dt, target_pos, obstacle_poses)
            
            print(f"Replaying trajectory {trajectory_index} from {dataset_folder}")
            panda.replay(q_trajectory)
        else:
            print(f"Invalid trajectory index: {trajectory_index}")
    except Exception as e:
        print(f"Error loading trajectory: {e}")


if __name__ == "__main__":
    interactive_main()