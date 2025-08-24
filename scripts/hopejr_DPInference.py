import os
import torch
import numpy as np
import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path

from hopejr_DPTrainer import DiffusionPolicy, DiffusionPolicyNetwork
from hopejr_DatasetProcessor import TrajectoryDataProcessor

class ArmController:
    def __init__(self, model_path, scene_xml_path, scalers_dir=None, device='cuda', action_horizon=10):
        self.device = device
        self.model_path = model_path
        self.scalers_dir = Path(scalers_dir) if scalers_dir else None

        #Mujoco scene
        self.model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        self.scalers = self._load_scalers()

        self.policy = self._load_model()
        self.n_joints = 7
        self.action_horizon = action_horizon
        self.obs_dim = 13
        self.action_dim = 7
        self._find_indices()

        self.action_buffer = deque(maxlen=self.action_horizon)
        self.action_step = 0

        #Trajectory recording
        self.trajectory = {
            'time': [],
            'joint_positions': [],
            'end_effector_pos': [],
            'target_pos': [],
            'obstacle_pos': [],
            'distance_to_target': [],
            'distance_to_obstacle': []
        }
        
        print(f"Controller initialized successfully!")
        print(f"Joints: {self.joint_names}")
        print(f"Target object: {self.target_name}")
        print(f"Obstacle: {self.obstacle_name}")
        print(f"Scalers loaded: {list(self.scalers.keys())}")

    def _load_scalers(self):
        scalers = {}
        
        if self.scalers_dir is None:
            raise ValueError("Scaler directory could not be empty")
            
        if not self.scalers_dir.exists():
            raise ValueError("Scaler directory not found")
        
        #Load observation scalers
        obs_joint_scaler_path = self.scalers_dir / "obs_joint_scaler.pkl"
        if obs_joint_scaler_path.exists():
            scalers['obs_joint_scaler'] = TrajectoryDataProcessor.load_scaler(obs_joint_scaler_path)
            print(f"Loaded obs_joint_scaler from {obs_joint_scaler_path}")
        else:
            print(f"Warning: obs_joint_scaler not found at {obs_joint_scaler_path}")
            
        obs_position_scaler_path = self.scalers_dir / "obs_position_scaler.pkl"
        if obs_position_scaler_path.exists():
            scalers['obs_position_scaler'] = TrajectoryDataProcessor.load_scaler(obs_position_scaler_path)
            print(f"Loaded obs_position_scaler from {obs_position_scaler_path}")
        else:
            print(f"Warning: obs_position_scaler not found at {obs_position_scaler_path}")
        
        #Load action scaler
        action_scaler_path = self.scalers_dir / "action_scaler.pkl"
        if action_scaler_path.exists():
            scalers['action_scaler'] = TrajectoryDataProcessor.load_scaler(action_scaler_path)
            print(f"Loaded action_scaler from {action_scaler_path}")
        else:
            print(f"Warning: action_scaler not found at {action_scaler_path}")
        
        return scalers

    def _find_indices(self):
        #Joint
        self.joint_names = [f'joint{i+1}' for i in range(self.n_joints)]
        self.joint_ids = []
        for joint_name in self.joint_names:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                self.joint_ids.append(joint_id)
            except:
                print(f"Warning: Joint {joint_name} not found")
        
        #Object
        try:
            self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'obj')
            self.target_name = 'obj'
        except:
            print("Warning: Target object 'obj' not found")
            self.target_id = None
            
        try:
            self.obstacle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'obstacle1')
            self.obstacle_name = 'obstacle1'
        except:
            print("Warning: Obstacle 'obstacle1' not found")
            self.obstacle_id = None
        
        #EE
        try:
            self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link7')
            self.ee_name = 'link7'
        except:
            print("Warning: End effector 'link7' not found")
            self.ee_id = None

    def _load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        config = checkpoint['config']
        
        network = DiffusionPolicyNetwork(
            obs_dim=config['obs_dim'],
            action_dim=config['action_dim'],
            action_horizon=config['action_horizon'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            time_emb_dim=config['time_emb_dim']
        ).to(self.device)
        
        policy = DiffusionPolicy(
            network,
            num_diffusion_steps=config['num_diffusion_steps'],
            beta_schedule=config['beta_schedule']
        ).to(self.device)
        
        policy.load_state_dict(checkpoint['model_state_dict'])
        policy.eval()
        
        print(f"Model loaded from {self.model_path}")
        print(f"Model config: {config}")
        
        return policy
    
    def get_observation(self, normalize=True):
        joint_angles = np.array([self.data.qpos[qpos_id] for qpos_id in self.joint_ids])
        
        if self.target_id is not None:
            target_pos = self.data.body(self.target_id).xpos.copy()
        else:
            target_pos = np.array([0.6, 0.0, 0.7])
        
        if self.obstacle_id is not None:
            obstacle_pos = self.data.body(self.obstacle_id).xpos.copy()
        else:
            obstacle_pos = np.array([0.3, 0.0, 0.7])
        
        obs = np.concatenate([joint_angles, target_pos, obstacle_pos])

        #Normalize obs using scaler
        if normalize and 'obs_joint_scaler' in self.scalers and 'obs_position_scaler' in self.scalers:
            obs = self._normalize_observation(obs)

        return obs.astype(np.float32)
    
    def _normalize_observation(self, obs):
        joint_angles = obs[:7].reshape(1, -1)
        positions = obs[7:].reshape(1, -1)
        
        joint_angles_normalized = self.scalers['obs_joint_scaler'].transform(joint_angles)
        positions_normalized = self.scalers['obs_position_scaler'].transform(positions)
        
        obs_normalized = np.concatenate([
            joint_angles_normalized.flatten(),
            positions_normalized.flatten()
        ])
        
        return obs_normalized

    def get_ee_position(self):
        if self.ee_id is not None:
            return self.data.body(self.ee_id).xpos.copy()
        else:
            return np.array([0.0, 0.0, 0.0])
        
    def predict_action(self, obs, num_inference_steps):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            action_tensor = self.policy.sample(obs_tensor, num_inference_steps=num_inference_steps)
            action = action_tensor.cpu().numpy()[0]  #(action_horizon, action_dim)

        if 'action_scaler' in self.scalers:
            action = self._denormalize_actions(action)
        return action
    
    def _denormalize_actions(self, scaled_actions):
        original_shape = scaled_actions.shape
        actions_reshaped = scaled_actions.reshape(-1, scaled_actions.shape[-1])
        actions_denormalized = self.scalers['action_scaler'].inverse_transform(actions_reshaped)
        return actions_denormalized.reshape(original_shape)

    def execute_action(self, action_delta):
        current_qpos = np.array([self.data.qpos[qpos_id] for qpos_id in self.joint_ids])
        
        #Try a small scale
        target_qpos = current_qpos + action_delta * 2
        
        #Check joint limits
        for i, joint_id in enumerate(self.joint_ids):
            joint_range = self.model.jnt_range[joint_id]
            target_qpos[i] = np.clip(target_qpos[i], joint_range[0], joint_range[1])
        
        self.data.ctrl[:self.n_joints] = target_qpos

    def reset_robot(self):
        initial_qpos = np.array([0.0, 0.5, -0.5, 0.8, -0.3, 0.0, 0.0])
        
        for i, joint_id in enumerate(self.joint_ids):
            self.data.qpos[joint_id] = initial_qpos[i]
        
        for joint_id in self.joint_ids:
            self.data.qvel[joint_id] = 0.0
        
        self.data.ctrl[:self.n_joints] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        
        self.action_buffer.clear()
        self.action_step = 0
        
        #Clear recoding
        self.trajectory = {
            'time': [],
            'joint_positions': [],
            'end_effector_pos': [],
            'target_pos': [],
            'obstacle_pos': [],
            'distance_to_target': [],
            'distance_to_obstacle': []
        }

    def record_trajectory(self, sim_time):
        obs = self.get_observation(normalize=False)
        ee_pos = self.get_ee_position()
        
        joint_pos = obs[:7]
        target_pos = obs[7:10]
        obstacle_pos = obs[10:13]
        
        dist_to_target = np.linalg.norm(ee_pos - target_pos)
        dist_to_obstacle = np.linalg.norm(ee_pos - obstacle_pos)
        
        self.trajectory['time'].append(sim_time)
        self.trajectory['joint_positions'].append(joint_pos.copy())
        self.trajectory['end_effector_pos'].append(ee_pos.copy())
        self.trajectory['target_pos'].append(target_pos.copy())
        self.trajectory['obstacle_pos'].append(obstacle_pos.copy())
        self.trajectory['distance_to_target'].append(dist_to_target)
        self.trajectory['distance_to_obstacle'].append(dist_to_obstacle)

    def run_simulation(self, duration=30.0, control_freq=10.0, execute_steps=1, num_inference_steps=50):
        print("Starting simulation...")
        print("Press 'esc' to exit")

        self.reset_robot()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.viewer = viewer

            dt = 1.0 / control_freq
            sim_time = 0.0
            last_control_time = 0.0

            while sim_time < duration and viewer.is_running():
                step_start = time.time()

                #Control freq
                if sim_time - last_control_time >= dt:
                    obs = self.get_observation(normalize=True)

                    #Different from typical sliding window, here we execute all predicted steps in buffer instead of just executing only first step, to lower the inference freq
                    #Action buffer stores all 8 steps
                    if len(self.action_buffer) == 0 or self.action_step >= execute_steps:
                        predicted_actions = self.predict_action(obs, num_inference_steps)

                        self.action_buffer.clear()
                        for i in range(self.action_horizon):
                            self.action_buffer.append(predicted_actions[i])
                        self.action_step = 0

                    #Execute step
                    if len(self.action_buffer) > 0:
                        current_action = self.action_buffer[self.action_step]
                        self.execute_action(current_action)
                        self.action_step += 1

                    self.record_trajectory(sim_time)

                    ee_pos = self.get_ee_position()
                    obs_raw = self.get_observation(normalize=False)
                    target_pos = obs_raw[7:10]
                    obstacle_pos = obs_raw[10:13]
                    dist_to_target = np.linalg.norm(ee_pos - target_pos)
                    dist_to_obstacle = np.linalg.norm(ee_pos - obstacle_pos)

                    print(f"Time: {sim_time:.2f}s | "
                          f"Target dist: {dist_to_target:.3f} | "
                          f"Obstacle dist: {dist_to_obstacle:.3f} | "
                          f"Action step: {self.action_step}/{self.action_horizon}")
                    
                    last_control_time = sim_time

                mujoco.mj_step(self.model, self.data)
                sim_time = self.data.time

                viewer.sync()

                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time.sleep(0.01)

        print("Simulation completed!")
        return self.trajectory

def main():
    project_root_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(project_root_path, 'data', 'DPModel', 'checkpoint_epoch_140.pth')
    scene_xml_path = os.path.join(project_root_path, 'MujocoModel', 'scene.xml')
    scalers_dir = os.path.join(project_root_path, 'data', 'Dataset')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    controller = ArmController(
        model_path=model_path,
        scene_xml_path=scene_xml_path,
        scalers_dir=scalers_dir,
        device=device,
        action_horizon=15
    )
    
    trajectory = controller.run_simulation(duration=60.0, control_freq=30.0, execute_steps=10, num_inference_steps=20)

if __name__ == "__main__":
    main()