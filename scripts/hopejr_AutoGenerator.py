import numpy as np
from datetime import datetime
import os
import gc
import mujoco
import pickle
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from hopejr_Optimizer_SLSQP import Optimizer
from hopejr_IKSolver import IKSolver

class TrajectoryGenerator:
    def __init__(self, scene_xml_path, dataset_root_path="./dataset", difficulty_scale=0.0):
        self.scene_xml_path = scene_xml_path
        self.dataset_root_path = Path(dataset_root_path)
        self.dataset_root_path.mkdir(exist_ok=True)
        
        self.model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.data = mujoco.MjData(self.model)

        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        self.obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj")
        self.obstacle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obstacle1")

        self.ik_solver = IKSolver(self.model, self.data)
        
        min_ratio = 0.2 + 0.8 * difficulty_scale
        self.workspace_full_bounds = {
            'x': [0.05, 0.5],
            'y': [-0.3, 0.1], 
            'z': [0.5, 1.0]
        }
        self.workspace_bounds = {}
        for k in self.workspace_full_bounds:
            lower, upper = self.workspace_full_bounds[k]
            center = (lower + upper) / 2
            half_range = (upper - lower) * min_ratio / 2
            self.workspace_bounds[k] = [center - half_range, center + half_range]
        
        self.obstacle_bounds = {
            'x': [0.15, 0.4],
            'y': [-0.2, 0],
            'z': [0.6, 0.9]
        }
        
        self.min_target_obstacle_dist = 0.12
        self.min_start_target_dist = 0.15
        
        self.joint_ranges = np.array([
            [-0.1, 0.1],
            [-0.1, 0.1],
            [-0.1, 0.1],
            [-0.1, 0.1],
            [-0.1, 0.1],
            [-0.1, 0.1],
            [-0.1, 0.1]
        ])
        
        self.N = 30
        
    def sample_configuration(self):
        max_attempts = 100
        
        for _ in range(max_attempts):
            q_start = np.array([
                np.random.uniform(low, high) 
                for low, high in self.joint_ranges
            ])
            
            obstacle_pos = np.array([
                np.random.uniform(*self.obstacle_bounds['x']),
                np.random.uniform(*self.obstacle_bounds['y']),
                np.random.uniform(*self.obstacle_bounds['z'])
            ])
            
            target_pos = np.array([
                np.random.uniform(*self.workspace_bounds['x']),
                np.random.uniform(*self.workspace_bounds['y']),
                np.random.uniform(*self.workspace_bounds['z'])
            ])
            
            target_obstacle_dist = np.linalg.norm(target_pos - obstacle_pos)
            if target_obstacle_dist < self.min_target_obstacle_dist:
                continue
                
            self.data.qpos[:7] = q_start
            mujoco.mj_forward(self.model, self.data)
            start_ee_pos = self.data.xpos[self.ee_id].copy()
            start_target_dist = np.linalg.norm(start_ee_pos - target_pos)
            
            if start_target_dist < self.min_start_target_dist:
                continue
                
            return q_start, target_pos, obstacle_pos
            
        return None, None, None
    
    def set_scene_configuration(self, target_pos, obstacle_pos):
        self.data.mocap_pos[0] = target_pos
        self.data.mocap_pos[1] = obstacle_pos
        mujoco.mj_forward(self.model, self.data)
    
    def check_reachability(self, q_start, target_pos):
        try:
            q_target, error = self.ik_solver.ik_position(target_position=target_pos, q_init=q_start)
            return error < 0.05
        except:
            return False
    
    def generate_single_trajectory(self, task_id, q_start, target_pos, obstacle_pos):
        try:
            self.set_scene_configuration(target_pos, obstacle_pos)
            
            if not self.check_reachability(q_start, target_pos):
                return None
            
            optimizer = Optimizer(
                timestep=self.N,
                q0=q_start,
                mujoco_model=self.model,
                mujoco_data=self.data,
                use_synergies=False,
                position_only=True
            )
            
            optimizer.target_pos = target_pos
            optimizer.obstacle_poses = {"obstacle1": obstacle_pos}
            
            q_trajectory = optimizer.optimize(
                q0=q_start, 
                EE_target=target_pos, 
                init_method="cartesian_rrt"
            )
            
            self.data.qpos[:7] = q_trajectory[-1]
            mujoco.mj_forward(self.model, self.data)
            final_ee_pos = self.data.xpos[self.ee_id].copy()
            final_error = np.linalg.norm(final_ee_pos - target_pos)
            
            if final_error > 0.08:
                return None
            
            return {
                'task_id': task_id,
                'q_start': q_start,
                'q_trajectory': q_trajectory,
                'target_pos': target_pos,
                'obstacle_pos': obstacle_pos,
                'final_error': final_error,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return None
    
    def generate_dataset(self, num_trajectories=50, save_interval=10, verbose=True):
        trajectories = []
        successful = 0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.dataset_root_path / f"dataset_{timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        if verbose:
            print(f"Start generating, trajectory number: {num_trajectories} ")
            print(f"Output path: {output_dir}")
            print("-" * 50)
        
        progress_bar = tqdm(total=num_trajectories, desc="generating") if verbose else None
        
        attempts = 0
        max_attempts = num_trajectories * 3
        
        while successful < num_trajectories and attempts < max_attempts:
            attempts += 1
            
            q_start, target_pos, obstacle_pos = self.sample_configuration()
            if q_start is None:
                continue
            
            trajectory_data = self.generate_single_trajectory(
                f"task_{attempts:06d}", q_start, target_pos, obstacle_pos
            )
            
            if trajectory_data is not None:
                trajectories.append(trajectory_data)
                successful += 1
                
                if progress_bar:
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'success_rate': f"{successful/attempts*100:.1f}%",
                        'last_error': f"{trajectory_data['final_error']:.4f}"
                    })
                
                if successful % save_interval == 0:
                    self._save_trajectories(trajectories, output_dir, f"partial_{successful}")
        
        if progress_bar:
            progress_bar.close()
        
        if trajectories:
            self._save_trajectories(trajectories, output_dir, "final")
            self._save_metadata(trajectories, output_dir, attempts)
            
            if verbose:
                print(f"\nDataset generation successful!")
                print(f"Successful trajectories: {len(trajectories)}")
        
        return trajectories, output_dir
    
    def generate_dataset_parallel(self, num_trajectories=50, save_interval=10, verbose=True, n_workers=None):
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)
        
        trajectories = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.dataset_root_path / f"dataset_{timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        if verbose:
            print(f"Start parallel generation with {n_workers} workers")
            print(f"Target trajectory number: {num_trajectories}")
            print(f"Output path: {output_dir}")
            print("-" * 50)
        
        tasks = []
        for i in range(num_trajectories * 3):
            seed = np.random.randint(0, 10000000)
            task_id = f"task_{i:06d}"
            tasks.append((task_id, seed, str(self.scene_xml_path)))
        
        successful = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {
                executor.submit(generate_single_trajectory_worker, task): task[0] 
                for task in tasks
            }

            for future in as_completed(future_to_task):
                if successful >= num_trajectories:
                    break
                    
                try:
                    result = future.result()
                    if result is not None:
                        trajectories.append(result)
                        successful += 1
                        
                        if verbose and successful % save_interval == 0:
                            print(f"Generated {successful}/{num_trajectories} trajectories")
                            self._save_trajectories(trajectories, output_dir, f"partial_{successful}")
                            
                except Exception as e:
                    if verbose:
                        print(f"Task failed with error: {e}")
                    continue
        
        if verbose:
            print(f"\nParallel dataset generation completed!")
            print(f"Successfully generated {len(trajectories)} trajectories")
        
        if trajectories:
            self._save_trajectories(trajectories, output_dir, "final")
            self._save_metadata(trajectories, output_dir, len(tasks))
        
        return trajectories, output_dir
    
    def _save_trajectories(self, trajectories, output_dir, suffix):
        filename = f"trajectories_{suffix}.pkl"
        with open(output_dir / filename, 'wb') as f:
            pickle.dump(trajectories, f)
    
    def _save_metadata(self, trajectories, output_dir, total_attempts):
        errors = [t['final_error'] for t in trajectories]
        
        metadata = {
            'generation_info': {
                'total_trajectories': len(trajectories),
                'total_attempts': total_attempts,
                'success_rate': len(trajectories) / total_attempts,
                'scene_file': str(self.scene_xml_path)
            },
            'workspace_bounds': self.workspace_bounds,
            'obstacle_bounds': self.obstacle_bounds,
            'statistics': {
                'avg_error': float(np.mean(errors)),
                'max_error': float(np.max(errors)),
                'min_error': float(np.min(errors)),
                'std_error': float(np.std(errors))
            },
            'generation_time': datetime.now().isoformat()
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


class TrajectoryDataProcessor:
    def __init__(self, trajectories_file):
        with open(trajectories_file, 'rb') as f:
            self.trajectories = pickle.load(f)
        print(f"{len(self.trajectories)} trajectories are loaded")
    
    def create_training_data(self, prediction_horizon=4):
        observations = []
        actions = []
        
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
        
        return np.array(observations), np.array(actions)
    
    def save_for_training(self, output_dir, test_split=0.2):
        obs, actions = self.create_training_data()
        
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
        
        print(f"Dataset saved!:")
        print(f"Train set size: {len(train_obs)}")
        print(f"Test set size: {len(test_obs)}")
        print(f"Obs dim: {obs.shape[1]}")
        print(f"Action dim: {actions.shape[1:]}")
        
        return train_file, test_file


def generate_trajectories(scene_xml_path, num_trajectories=30, output_dir=None, use_parallel=True, n_workers=None, difficulty_scale=0.0):
    if not os.path.exists(scene_xml_path):
        raise ValueError("Scene file does not exist!")
    
    generator = TrajectoryGenerator(
        scene_xml_path, 
        dataset_root_path=output_dir or "./data",
        difficulty_scale=difficulty_scale
    )

    if use_parallel:
        n_workwers = n_workers or max(1, mp.cpu_count() - 1)
        print(f"Parallel generation on, workers: {n_workers}")
        trajectories, dataset_dir = generator.generate_dataset_parallel(
            num_trajectories=num_trajectories,
            n_workers=n_workers
        )
    else:
        trajectories, dataset_dir = generator.generate_dataset(num_trajectories)
    
    if not trajectories:
        print("Generation failed!")
        return None, None
    
    trajectories_file = dataset_dir / "trajectories_final.pkl"
    processor = TrajectoryDataProcessor(trajectories_file)
    train_file, test_file = processor.save_for_training(dataset_dir)
    
    print(f"\nGENERATION COMPLETE!")
    
    return train_file, test_file

def generate_single_trajectory_worker(args):
    task_id, seed, scene_xml_path = args
    
    try:
        np.random.seed(seed)
        
        generator = TrajectoryGenerator(scene_xml_path)
        q_start, target_pos, obstacle_pos = generator.sample_configuration()
        if q_start is None:
            return None
        
        trajectory_data = generator.generate_single_trajectory(
            task_id, q_start, target_pos, obstacle_pos
        )
        
        return trajectory_data
        
    except Exception as e:
        print(f"Worker process error in {task_id}: {e}")
        return None


if __name__ == "__main__":
    project_root_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(project_root_path,'MujocoModel', 'scene.xml')
    output_base_path = os.path.join(project_root_path,'data', 'Trajectory')
    
    use_parallel = True
    n_workers = 10
    
    batch_size = 200
    total_trajectories = 1000
    num_batches = (total_trajectories + batch_size - 1) // batch_size
    
    all_train_files = []
    all_test_files = []
    total_generated = 0
    
    print(f"Starting batch generation, Batch number: {num_batches}, Batch size: {batch_size}")
    
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, total_trajectories - total_generated)
        if current_batch_size <= 0:
            break
            
        print(f"\n{'='*50}")
        print(f"Batch {batch_idx + 1}/{num_batches}")
        print(f"current batch size: {current_batch_size}")
        print(f"process: {total_generated}/{total_trajectories}")
        print(f"{'='*50}")
        
        difficulty_scale = min(1.0, batch_idx / (num_batches - 1))
        batch_output_path = os.path.join(output_base_path, f"batch_{batch_idx + 1:03d}")
        
        try:
            train_file, test_file = generate_trajectories(
                scene_xml_path=model_path,
                num_trajectories=current_batch_size,
                output_dir=batch_output_path,
                use_parallel=use_parallel,
                n_workers=n_workers,
                difficulty_scale=difficulty_scale
            )
            
            if train_file and test_file:
                all_train_files.append(train_file)
                all_test_files.append(test_file)
                total_generated += current_batch_size
                print(f"Batch {batch_idx + 1} success!")
            else:
                print(f"Batch {batch_idx + 1} fail")
                
        except Exception as e:
            print(f"Batch {batch_idx + 1} error: {e}")
            continue
        
        gc.collect()
        
        if batch_idx < num_batches - 1:
            print("... Inter-batch break :) ...")
            import time
            time.sleep(30)
    

