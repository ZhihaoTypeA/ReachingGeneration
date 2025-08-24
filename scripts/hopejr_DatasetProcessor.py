import numpy as np
import pickle
import os
from pathlib import Path
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

class TrajectoryDataProcessor:
    def __init__(self, trajectories_dir, trajectories_filename="trajectories_final.pkl"):
        self.trajectories = []
        
        if isinstance(trajectories_dir, (str, Path)):
            path = Path(trajectories_dir)
            if path.is_dir():
                pkl_files = list(path.glob(f"**/{trajectories_filename}"))
                print("Filename not found, loaded pkl automatically")
                if not pkl_files:
                    pkl_files = list(path.glob("**/*.pkl"))
                trajectories_files = pkl_files
            else:
                trajectories_files = [path]
        else:
            trajectories_files = [Path(f) for f in trajectories_dir]
        
        total_trajectories = 0
        for file_path in trajectories_files:
            try:
                with open(file_path, 'rb') as f:
                    batch_trajectories = pickle.load(f)
                self.trajectories.extend(batch_trajectories)
                total_trajectories += len(batch_trajectories)
                print(f"{file_path}: {len(batch_trajectories)} trajectories are loaded")
            except Exception as e:
                print(f"{file_path} loading failed: {e}")
        
        print(f"Total {total_trajectories} trajectories from {len(trajectories_files)} files are loaded")

        self.action_scaler = None
        self.obs_joint_scaler = None
        self.obs_position_scaler = None
    
    def create_training_data(self, prediction_horizon=4, action_scaling_method='standard', obs_scaling_method='standard'):
        observations = []
        actions = []

        if len(self.trajectories) == 0:
            raise ValueError("No trajectories loaded")

        print(f"Action scaling method: {action_scaling_method}")
        print(f"Observation scaling method: {obs_scaling_method}")
        
        for traj in self.trajectories:
            q_traj = traj['q_trajectory']
            target_pos = traj['target_pos']
            obstacle_pos = traj['obstacle_pos']

            
            joint_actions = np.diff(q_traj, axis=0)
            
            for t in range(len(q_traj) - prediction_horizon):
                obs = np.concatenate([
                    q_traj[t],
                    target_pos,
                    obstacle_pos
                ]).astype(np.float32)
                
                action_seq = joint_actions[t:t+prediction_horizon].astype(np.float32)
                
                observations.append(obs)
                actions.append(action_seq)
        
        observations = np.array(observations)
        actions = np.array(actions)
        
        print(f"Raw data statistics:")
        print(f"  Observations shape: {observations.shape}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Joint angles (obs[:,:7]) - mean: {observations[:,:7].mean():.6f}, std: {observations[:,:7].std():.6f}")
        print(f"  Target positions (obs[:,7:10]) - mean: {observations[:,7:10].mean():.6f}, std: {observations[:,7:10].std():.6f}")
        print(f"  Obstacle positions (obs[:,10:13]) - mean: {observations[:,10:13].mean():.6f}, std: {observations[:,10:13].std():.6f}")
        print(f"  Actions - mean: {actions.mean():.6f}, std: {actions.std():.6f}, min: {actions.min():.6f}, max: {actions.max():.6f}")
        
        if obs_scaling_method is not None:
            observations = self._scale_obs(observations, method=obs_scaling_method)
            print(f"After observation scaling:")
            print(f"  Joint angles - mean: {observations[:,:7].mean():.6f}, std: {observations[:,:7].std():.6f}")
            print(f"  Positions - mean: {observations[:,7:].mean():.6f}, std: {observations[:,7:].std():.6f}")

        if action_scaling_method is not None:
            actions = self._scale_actions(actions, method=action_scaling_method)
            print(f"After action scaling:")
            print(f"  Actions - mean: {actions.mean():.6f}, std: {actions.std():.6f}, min: {actions.min():.6f}, max: {actions.max():.6f}")
        
        return observations, actions

    def _scale_obs(self, observations, method='standard'):
        #Scaling joint position and position of object seperately
        joint_angles = observations[:, :7]
        positions = observations[:, 7:]

        if method == 'standard':
            self.obs_joint_scaler = StandardScaler()
            self.obs_position_scaler = StandardScaler()
        elif method == 'minmax':
            self.obs_joint_scaler = MinMaxScaler(feature_range=(-1, 1))
            self.obs_position_scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        joint_angles_scaled = self.obs_joint_scaler.fit_transform(joint_angles)
        positions_scaled = self.obs_position_scaler.fit_transform(positions)

        observations_scaled = np.concatenate([joint_angles_scaled, positions_scaled], axis=1)

        return observations_scaled.astype(np.float32)

    def _scale_actions(self, actions, method='standard'):
        original_shape = actions.shape
        actions_reshaped = actions.reshape(-1, actions.shape[-1])
        
        if method == 'standard':
            self.action_scaler = StandardScaler()
        elif method == 'minmax':
            self.action_scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        actions_scaled = self.action_scaler.fit_transform(actions_reshaped)
        
        actions_scaled = actions_scaled.reshape(original_shape)
        
        return actions_scaled.astype(np.float32)

    def save_for_training(self, output_dir, test_split=0.2, prediction_horizon=4,  action_scaling_method='standard', obs_scaling_method='standard'):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        obs, actions = self.create_training_data(
            prediction_horizon=prediction_horizon, 
            action_scaling_method=action_scaling_method,
            obs_scaling_method=obs_scaling_method
        )
        
        n_total = len(obs)
        indices = np.random.permutation(n_total)
        n_test = int(n_total * test_split)
        
        train_obs = obs[indices[:-n_test]]
        train_actions = actions[indices[:-n_test]]
        
        test_obs = obs[indices[-n_test:]]
        test_actions = actions[indices[-n_test:]]
        
        train_file = output_dir / "train_data.npz"
        test_file = output_dir / "test_data.npz"
        
        np.savez(train_file, observations=train_obs, actions=train_actions)
        np.savez(test_file, observations=test_obs, actions=test_actions)
        
        if self.obs_joint_scaler is not None:
            obs_joint_scaler_file = output_dir / "obs_joint_scaler.pkl"
            joblib.dump(self.obs_joint_scaler, obs_joint_scaler_file)
            
        if self.obs_position_scaler is not None:
            obs_position_scaler_file = output_dir / "obs_position_scaler.pkl"
            joblib.dump(self.obs_position_scaler, obs_position_scaler_file)

        if self.action_scaler is not None:
            action_scaler_file = output_dir / "action_scaler.pkl"
            joblib.dump(self.action_scaler, action_scaler_file)

        metadata = {
            'dataset_info': {
                'total_trajectories': len(self.trajectories),
                'total_samples': n_total,
                'train_samples': len(train_obs),
                'test_samples': len(test_obs),
                'test_split': test_split,
                'prediction_horizon': prediction_horizon,
                'obs_dim': obs.shape[1],
                'action_dim': actions.shape[1:],
                'generation_time': datetime.now().isoformat()
            },
            'preprocessing_info': {
                'action_scaling_method': action_scaling_method,
                'obs_scaling_method': obs_scaling_method,
                'has_action_scaler': self.action_scaler is not None,
                'has_obs_joint_scaler': self.obs_joint_scaler is not None,
                'has_obs_position_scaler': self.obs_position_scaler is not None
            },
            'data_statistics': {
                'obs_joint_mean': float(obs[:, :7].mean()),
                'obs_joint_std': float(obs[:, :7].std()),
                'obs_position_mean': float(obs[:, 7:].mean()),
                'obs_position_std': float(obs[:, 7:].std()),
                'action_mean': float(actions.mean()),
                'action_std': float(actions.std()),
                'action_min': float(actions.min()),
                'action_max': float(actions.max())
            }
        }
        
        scaler_params = {}
        
        if self.action_scaler is not None:
            if hasattr(self.action_scaler, 'mean_'):
                scaler_params['action_scaler'] = {
                    'mean': self.action_scaler.mean_.tolist(),
                    'scale': self.action_scaler.scale_.tolist()
                }
            elif hasattr(self.action_scaler, 'data_min_'):
                scaler_params['action_scaler'] = {
                    'data_min': self.action_scaler.data_min_.tolist(),
                    'data_max': self.action_scaler.data_max_.tolist(),
                    'feature_range': self.action_scaler.feature_range
                }
        
        if self.obs_joint_scaler is not None:
            if hasattr(self.obs_joint_scaler, 'mean_'):
                scaler_params['obs_joint_scaler'] = {
                    'mean': self.obs_joint_scaler.mean_.tolist(),
                    'scale': self.obs_joint_scaler.scale_.tolist()
                }
            elif hasattr(self.obs_joint_scaler, 'data_min_'):
                scaler_params['obs_joint_scaler'] = {
                    'data_min': self.obs_joint_scaler.data_min_.tolist(),
                    'data_max': self.obs_joint_scaler.data_max_.tolist(),
                    'feature_range': self.obs_joint_scaler.feature_range
                }
        
        if self.obs_position_scaler is not None:
            if hasattr(self.obs_position_scaler, 'mean_'):
                scaler_params['obs_position_scaler'] = {
                    'mean': self.obs_position_scaler.mean_.tolist(),
                    'scale': self.obs_position_scaler.scale_.tolist()
                }
            elif hasattr(self.obs_position_scaler, 'data_min_'):
                scaler_params['obs_position_scaler'] = {
                    'data_min': self.obs_position_scaler.data_min_.tolist(),
                    'data_max': self.obs_position_scaler.data_max_.tolist(),
                    'feature_range': self.obs_position_scaler.feature_range
                }
        
        if scaler_params:
            metadata['scaler_params'] = scaler_params

        with open(output_dir / "dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset Saved Successfully!")
        print(f"train size: {len(train_obs)}")
        print(f"test size: {len(test_obs)}")
        print(f"obs dim: {obs.shape[1]}")
        print(f"action dim: {actions.shape[1:]}")
        print(f"observation scaling: {obs_scaling_method}")
        print(f"action scaling: {action_scaling_method}")
        print(f"output path: {output_dir}")
        
        return train_file, test_file
    
    @staticmethod
    def load_scaler(scaler_path):
        return joblib.load(scaler_path)
    
    def load_all_scaler(scaler_dir):
        scalers_dir = Path(scalers_dir)
        scalers = {}
        
        action_scaler_path = scalers_dir / "action_scaler.pkl"
        if action_scaler_path.exists():
            scalers['action_scaler'] = joblib.load(action_scaler_path)
        
        obs_joint_scaler_path = scalers_dir / "obs_joint_scaler.pkl"
        if obs_joint_scaler_path.exists():
            scalers['obs_joint_scaler'] = joblib.load(obs_joint_scaler_path)
            
        obs_position_scaler_path = scalers_dir / "obs_position_scaler.pkl"
        if obs_position_scaler_path.exists():
            scalers['obs_position_scaler'] = joblib.load(obs_position_scaler_path)
        
        return scalers

    @staticmethod
    def inverse_transform_actions(scaled_actions, scaler_path):
        scaler = TrajectoryDataProcessor.load_scaler(scaler_path)
        original_shape = scaled_actions.shape
        actions_reshaped = scaled_actions.reshape(-1, scaled_actions.shape[-1])
        actions_original = scaler.inverse_transform(actions_reshaped)
        return actions_original.reshape(original_shape)

    @staticmethod
    def inverse_transform_observations(scaled_observations, obs_joint_scaler_path, obs_position_scaler_path):
        obs_joint_scaler = TrajectoryDataProcessor.load_scaler(obs_joint_scaler_path)
        obs_position_scaler = TrajectoryDataProcessor.load_scaler(obs_position_scaler_path)
        
        joint_angles_scaled = scaled_observations[:, :7]
        positions_scaled = scaled_observations[:, 7:]
        
        joint_angles_original = obs_joint_scaler.inverse_transform(joint_angles_scaled)
        positions_original = obs_position_scaler.inverse_transform(positions_scaled)
        
        observations_original = np.concatenate([joint_angles_original, positions_original], axis=1)
        
        return observations_original
    
    @staticmethod
    def inverse_transform_all(scaled_observations, scaled_actions, scalers_dir):
        scalers_dir = Path(scalers_dir)
        
        obs_joint_scaler_path = scalers_dir / "obs_joint_scaler.pkl"
        obs_position_scaler_path = scalers_dir / "obs_position_scaler.pkl"
        
        if obs_joint_scaler_path.exists() and obs_position_scaler_path.exists():
            observations_original = TrajectoryDataProcessor.inverse_transform_observations(
                scaled_observations, obs_joint_scaler_path, obs_position_scaler_path
            )
        else:
            observations_original = scaled_observations
        
        action_scaler_path = scalers_dir / "action_scaler.pkl"
        if action_scaler_path.exists():
            actions_original = TrajectoryDataProcessor.inverse_transform_actions(
                scaled_actions, action_scaler_path
            )
        else:
            actions_original = scaled_actions
        
        return observations_original, actions_original



if __name__ == "__main__":
    project_root_path = os.path.dirname(os.path.dirname(__file__))
    pkl_dir = os.path.join(project_root_path, 'data', 'Dataset', 'Trajectory')
    pkl_filename = "trajectories_final.pkl"
    output_dir = os.path.join(project_root_path, 'data', 'Dataset')
    
    processor = TrajectoryDataProcessor(pkl_dir, pkl_filename)
    
    train_file, test_file = processor.save_for_training(
        output_dir=output_dir,
        test_split=0.2,
        prediction_horizon=15,
        action_scaling_method='standard',
        obs_scaling_method='standard'
    )
