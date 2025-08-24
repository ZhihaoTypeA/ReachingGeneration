import numpy as np
import mujoco
import time

class JointRRTConnectPlanner:
    def __init__(self, mujoco_model, mujoco_data, step_size=0.1, max_iter=5000, goal_threshold=0.1):
        self.mj_model = mujoco_model
        self.mj_data = mujoco_data
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_threshold = goal_threshold

        self.q_upper_limit = np.array([1.5707, 2.0943, 1.2217, 1.7453, 2.0943, 0.1745, 0.1745])
        self.q_lower_limit = np.array([-1.5707, 0, -1.5707, -0.2617, -0.8726, -0.5235, -0.8726])

        self.link_geom_ids = self.get_link_geom_ids()

        self.obstacle_poses = self.get_obstacle_positions()
        self.obstacle_geom_ids = self.get_obstacle_geom_ids()

        self.safety_margin = 0.08
        self.safety_margin_mesh = 0.01
        self.link_names = ["link1", "link2", "link3", "link4", "link5", "link6", "link7"]
        self.link_ids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.link_names]

    def get_obstacle_positions(self):
        positions = {}
        
        for i in range(self.mj_model.nbody):
            body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and 'obstacle' in body_name.lower():
                positions[body_name] = self.mj_data.xpos[i].copy()
        
        return positions
    
    def get_obstacle_geom_ids(self):
        obstacle_geom_ids = []
        
        for i in range(self.mj_model.ngeom):
            geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and 'obstacle' in geom_name.lower() and 'geom' in geom_name.lower():
                obstacle_geom_ids.append(i)
        
        return obstacle_geom_ids

    def get_link_geom_ids(self):
        link_geom_ids = []
        geom_names_to_check = ["link5_geom", "hand_geom"]

        for geom_name in geom_names_to_check:
            geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                link_geom_ids.append(geom_id)
            else:
                print(f"Warning: Geom {geom_name} not found")

        return link_geom_ids

    #Overall collision check
    def check_collision(self, q):
        current_qpos = self.mj_data.qpos.copy()

        self.mj_data.qpos[:7] = q
        mujoco.mj_forward(self.mj_model, self.mj_data)
        # link_positions = [self.mj_data.xpos[link_id].copy() for link_id in self.link_ids]
        # for i, link_pos in enumerate(link_positions[2:]):
        #     for obstacle_name, obstacle_pos in self.obstacle_poses.items():
        #         distance = np.linalg.norm(link_pos-obstacle_pos)
        #         if distance < self.safety_margin:
        #             self.mj_data.qpos[:] = current_qpos
        #             mujoco.mj_forward(self.mj_model, self.mj_data)
        #             return True
                
        geom_distance = self.check_geom_collision(q)
        if geom_distance < self.safety_margin_mesh:
            self.mj_data.qpos[:] = current_qpos
            mujoco.mj_forward(self.mj_model, self.mj_data)
            return True
        
        self.mj_data.qpos[:] = current_qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return False

    #link geom collision check
    def check_geom_collision(self, q):
        mujoco.mj_collision(self.mj_model, self.mj_data)

        min_distance = np.inf
        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            for link_geom_id in self.link_geom_ids:
                if ((geom1_id == link_geom_id and geom2_id in self.obstacle_geom_ids) or (geom1_id in self.obstacle_geom_ids and geom2_id == link_geom_id)):
                    distance = contact.dist
                    min_distance = min(min_distance, distance)

        #If no collision, get the min distance between 2 geom
        if min_distance == np.inf:
            for link_geom_id in self.link_geom_ids:
                for obstacle_geom_id in self.obstacle_geom_ids:
                    try:
                        distance = mujoco.mj_geomDistance(self.mj_model, self.mj_data, link_geom_id, obstacle_geom_id, 10.0, None)

                        if distance >= 0:
                            min_distance = min(min_distance, distance)
                    except:
                        pass

        return min_distance if min_distance != np.inf else 1.0

    def check_config_validity(self, q):
        if np.any(q < self.q_lower_limit) or np.any(q > self.q_upper_limit):
            return False
        
        return not self.check_collision(q)
    
    def random_config(self):
        for _ in range(1000):
            q = np.random.uniform(self.q_lower_limit, self.q_upper_limit)
            if self.check_config_validity(q):
                return q
        return (self.q_lower_limit + self.q_upper_limit) / 2
    
    def extend_towards(self, q_from, q_to):
        direction = q_to - q_from
        distance = np.linalg.norm(direction)

        if distance < self.step_size:
            if self.check_config_validity(q_to):
                return q_to, True
            else:
                return q_from, False
            
        direction = direction / distance * self.step_size
        q_new = q_from + direction

        if self.check_config_validity(q_new):
            return q_new, True
        else:
            return q_from, False
        
    def connect_trees(self, tree1, tree2, q_new):
        distances = [np.linalg.norm(q_new - node['config']) for node in tree2]
        nearest_idx = np.argmin(distances)
        q_nearest = tree2[nearest_idx]['config']

        path = [q_new]
        current = q_new

        while np.linalg.norm(current - q_nearest) > self.step_size:
            current, success = self.extend_towards(current, q_nearest)
            if not success:
                return None
            path.append(current)

        #final step to connect
        if np.linalg.norm(current - q_nearest) > 1e-6:
            current, success = self.extend_towards(current, q_nearest)
            if success and np.linalg.norm(current - q_nearest) < self.goal_threshold:
                path.append(q_nearest)
                return path, nearest_idx
        else:
            return path, nearest_idx

        return None    
    
    def plan_path(self, q_start, q_goal):
        if not self.check_config_validity(q_start):
            print("q_start is not valid!")
            return None
        if not self.check_config_validity(q_goal):
            print("q_goal is not valid!")
            return None
        
        tree_start = [{'config': q_start, 'parent': None, 'index': 0}]
        tree_goal = [{'config': q_goal, 'parent': None, 'index': 0}]

        print("Start RRT-Connect planning...")
        start_time = time.time()

        for iteration in range(self.max_iter):
            #Expand 2 trees alternately
            if iteration % 2 == 0:
                tree_a, tree_b = tree_start, tree_goal
            else:
                tree_a, tree_b = tree_goal, tree_start

            q_rand = self.random_config()

            distances = [np.linalg.norm(q_rand - node['config']) for node in tree_a]
            nearest_idx = np.argmin(distances)
            q_nearest = tree_a[nearest_idx]['config']

            q_new, success = self.extend_towards(q_nearest, q_rand)
            
            if success and np.linalg.norm(q_new - q_nearest) > 1e-6:
                new_node = {
                    'config': q_new,
                    'parent': nearest_idx,
                    'index': len(tree_a)
                }
                tree_a.append(new_node)

            connection_result = self.connect_trees([new_node], tree_b, q_new)
            if connection_result is not None:
                connection_path, target_idx = connection_result #target_idx always belongs to tree_2, here is tree_b

                #from tree_start to connection point
                path_start = []
                if iteration % 2 == 0: #tree_a is tree_start, tree_b is tree_goal
                    #backtracking from new node to root node of tree_start
                    current_idx = len(tree_a) - 1
                    while current_idx is not None:
                        path_start.append(tree_a[current_idx]['config'])
                        current_idx = tree_a[current_idx]['parent']
                    path_start.reverse()
                else: #tree_a is tree_goal, tree_b is tree_start
                    current_idx = target_idx
                    while current_idx is not None:
                        path_start.append(tree_b[current_idx]['config'])
                        current_idx = tree_b[current_idx]['parent']
                    path_start.reverse()

                path_goal = []
                if iteration % 2 == 0: #tree_a is tree_start, tree_b is tree_goal
                    current_idx = target_idx
                    while current_idx is not None:
                        path_goal.append(tree_b[current_idx]['config'])
                        current_idx = tree_b[current_idx]['parent']
                else:
                    current_idx = len(tree_a) - 1
                    while current_idx is not None:
                        path_goal.append(tree_a[current_idx]['config'])
                        current_idx = tree_a[current_idx]['parent']

                full_path = path_start + connection_path[1:] + path_goal[1:]

                planning_time = time.time() - start_time
                print(f"Plann successfully! time: {planning_time}, iter: {iteration}, path length: {len(full_path)}")

                return full_path
        
        print("Plan failed! Exceeds max iteration setting")
        return None
    
    def smooth_path(self, path, max_iterations=100):
        if len(path) < 3:
            return path
        
        smoothed_path = path.copy()

        for _ in range(max_iterations):
            if len(smoothed_path) < 3:
                break

            i = np.random.randint(0, len(smoothed_path) - 2)
            j = np.random.randint(i + 2, len(smoothed_path))

            if self.check_connect_directly(smoothed_path[i], smoothed_path[j]):
                smoothed_path = smoothed_path[:i+1] + smoothed_path[j:]
        
        return smoothed_path
            
    def check_connect_directly(self, q1, q2, num_checks=10):
        for i in range(num_checks + 1):
            alpha = i / num_checks
            q_intermediate = (1 - alpha) * q1 + alpha * q2
            if not self.check_config_validity(q_intermediate):
                return False
            
        return True
    
class CartesianRRTConnector:
    def __init__(self, mujoco_model, mujoco_data, ik_solver, step_size=0.05, 
                 max_iter=5000, goal_threshold=0.02, orientation_weight=0.3, position_only=True):
        self.mj_model = mujoco_model
        self.mj_data = mujoco_data
        self.ik_solver = ik_solver
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_threshold = goal_threshold
        self.orientation_weight = orientation_weight
        self.position_only = position_only
        
        #Cartesian workspace
        self.workspace_min = np.array([0.018, -0.5, 0.509])
        self.workspace_max = np.array([0.4, 0.045, 1.257])
        
        self.q_upper_limit = np.array([1.5707, 2.0943, 1.2217, 1.7453, 2.0943, 0.1745, 0.1745])
        self.q_lower_limit = np.array([-1.5707, 0, -1.5707, -0.2617, -0.8726, -0.5235, -0.8726])
        
        self.link_geom_ids = self.get_link_geom_ids()
        self.obstacle_poses = self.get_obstacle_positions()
        self.obstacle_geom_ids = self.get_obstacle_geom_ids()
        self.safety_margin = 0.08
        self.safety_margin_mesh = 0.01
        self.link_names = ["link1", "link2", "link3", "link4", "link5", "link6", "link7"]
        self.link_ids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name) 
                        for name in self.link_names]
        
        self.ee_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "link7")

    def get_obstacle_positions(self):
        positions = {}
        for i in range(self.mj_model.nbody):
            body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and 'obstacle' in body_name.lower():
                positions[body_name] = self.mj_data.xpos[i].copy()
        return positions
    
    def get_obstacle_geom_ids(self):
        obstacle_geom_ids = []
        for i in range(self.mj_model.ngeom):
            geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and 'obstacle' in geom_name.lower() and 'geom' in geom_name.lower():
                obstacle_geom_ids.append(i)
        return obstacle_geom_ids

    def get_link_geom_ids(self):
        link_geom_ids = []
        geom_names_to_check = ["link5_geom", "hand_geom"]
        for geom_name in geom_names_to_check:
            geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                link_geom_ids.append(geom_id)
            else:
                print(f"Warning: Geom {geom_name} not found")
        return link_geom_ids
    
    def get_ee_position(self, q):
        current_qpos = self.mj_data.qpos.copy()
        self.mj_data.qpos[:7] = q
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        ee_pos = self.mj_data.xpos[self.ee_id].copy()
        
        self.mj_data.qpos[:] = current_qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        return ee_pos

    def get_ee_pose(self, q):
        current_qpos = self.mj_data.qpos.copy()
        self.mj_data.qpos[:7] = q
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        ee_pos = self.mj_data.xpos[self.ee_id].copy()
        ee_quat = self.mj_data.xquat[self.ee_id].copy()  # [w, x, y, z]
        
        self.mj_data.qpos[:] = current_qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        return np.concatenate([ee_pos, ee_quat])  # [x, y, z, w, x, y, z]

    def cartesian_distance(self, pose1, pose2):
        if self.position_only:
            return np.linalg.norm(pose1[:3] - pose2[:3])
        else:
            #position distance
            pos_dist = np.linalg.norm(pose1[:3] - pose2[:3])
            #orientation distance (considering double cover)
            q1 = pose1[3:7]  # [w, x, y, z]
            q2 = pose2[3:7]
            
            q1 = q1 / np.linalg.norm(q1)
            q2 = q2 / np.linalg.norm(q2)


            dot_product = np.abs(np.dot(q1, q2))
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_diff = 2 * np.arccos(dot_product)
            
            total_dist = pos_dist + self.orientation_weight * angle_diff
            return total_dist
    
    def random_cartesian_pose(self, reference_q=None):
        max_attempts = 100
        
        for _ in range(max_attempts):
            #random position
            pos = np.random.uniform(self.workspace_min, self.workspace_max)

            if self.position_only:
                if reference_q is not None:
                    q_solution, error = self.ik_solver.ik_position(pos, reference_q)
                else:
                    q_init = (self.q_lower_limit + self.q_upper_limit) / 2
                    q_solution, error = self.ik_solver.ik_position(pos, q_init)

                if error < 0.08 and self.check_config_validity(q_solution):
                    return pos, q_solution

            else:
                #random orientation
                quat = self.random_quaternion()
                target_pose = np.concatenate([pos, quat])
                
                if reference_q is not None:
                    q_solution, error = self.ik_solver.ik_pose(target_pose, reference_q)
                else:
                    q_init = (self.q_lower_limit + self.q_upper_limit) / 2
                    q_solution, error = self.ik_solver.ik_pose(target_pose, q_init)
                
                if error < 0.08 and self.check_config_validity(q_solution):
                    return target_pose, q_solution
        
        if reference_q is not None:
            if self.position_only:
                return self.get_ee_position(reference_q), reference_q
            else:
                return self.get_ee_pose(reference_q), reference_q
        else:
            q_mid = (self.q_lower_limit + self.q_upper_limit) / 2
            if self.position_only:
                return self.get_ee_position(q_mid), q_mid
            else:
                return self.get_ee_pose(q_mid), q_mid
    
    def random_quaternion(self):
        #Shepperd's method for uniform random quaternions
        u1, u2, u3 = np.random.uniform(0, 1, 3)
        
        sqrt_1_u1 = np.sqrt(1 - u1)
        sqrt_u1 = np.sqrt(u1)
        
        w = sqrt_1_u1 * np.cos(2 * np.pi * u2)
        x = sqrt_1_u1 * np.sin(2 * np.pi * u2)
        y = sqrt_u1 * np.cos(2 * np.pi * u3)
        z = sqrt_u1 * np.sin(2 * np.pi * u3)
        
        return np.array([w, x, y, z])
    
    def interpolate_poses(self, pose1, pose2, t):
        if self.position_only:
            return (1 - t) * pose1 + t * pose2
        else:
            #position interp
            pos = (1 - t) * pose1[:3] + t * pose2[:3]
            #orientation interp (SLERP)
            q1 = pose1[3:7]
            q2 = pose2[3:7]
            if np.dot(q1, q2) < 0:
                q2 = -q2
            
            #SLERP
            dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)
            theta = np.arccos(np.abs(dot_product))
            
            if theta < 1e-6:
                quat = q1
            else:
                sin_theta = np.sin(theta)
                quat = (np.sin((1-t)*theta)/sin_theta)*q1 + (np.sin(t*theta)/sin_theta)*q2
            
            return np.concatenate([pos, quat])
    
    def check_config_validity(self, q):
        if np.any(q < self.q_lower_limit) or np.any(q > self.q_upper_limit):
            return False
        return not self.check_collision(q)
    
    def check_collision(self, q):
        current_qpos = self.mj_data.qpos.copy()
        
        self.mj_data.qpos[:7] = q
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        # link_positions = [self.mj_data.xpos[link_id].copy() for link_id in self.link_ids]
        # for i, link_pos in enumerate(link_positions[2:]):
        #     for obstacle_name, obstacle_pos in self.obstacle_poses.items():
        #         distance = np.linalg.norm(link_pos - obstacle_pos)
        #         if distance < self.safety_margin:
        #             self.mj_data.qpos[:] = current_qpos
        #             mujoco.mj_forward(self.mj_model, self.mj_data)
        #             return True
        
        geom_distance = self.check_geom_collision(q)
        if geom_distance < self.safety_margin_mesh:
            self.mj_data.qpos[:] = current_qpos
            mujoco.mj_forward(self.mj_model, self.mj_data)
            return True
        
        self.mj_data.qpos[:] = current_qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return False
    
    def check_geom_collision(self, q):
        mujoco.mj_collision(self.mj_model, self.mj_data)
        
        min_distance = np.inf
        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            for link_geom_id in self.link_geom_ids:
                if ((geom1_id == link_geom_id and geom2_id in self.obstacle_geom_ids) or 
                    (geom1_id in self.obstacle_geom_ids and geom2_id == link_geom_id)):
                    distance = contact.dist
                    min_distance = min(min_distance, distance)
        
        if min_distance == np.inf:
            for link_geom_id in self.link_geom_ids:
                for obstacle_geom_id in self.obstacle_geom_ids:
                    try:
                        distance = mujoco.mj_geomDistance(
                            self.mj_model, self.mj_data, 
                            link_geom_id, obstacle_geom_id, 10.0, None
                        )
                        if distance >= 0:
                            min_distance = min(min_distance, distance)
                    except:
                        pass
        
        return min_distance if min_distance != np.inf else 1.0

    def extend_towards(self, pose_from, pose_to, q_from):
        distance = self.cartesian_distance(pose_from, pose_to)
        
        if distance < self.step_size:
            if self.position_only:
                target_pos = pose_to[:3]
                q_solution, error = self.ik_solver.ik_position(target_pos, q_from)
            else:
                q_solution, error = self.ik_solver.ik_pose(pose_to, q_from)
            if error < 0.08 and self.check_config_validity(q_solution):
                return pose_to, q_solution, True
            else:
                return pose_from, q_from, False
        
        t = self.step_size / distance
        pose_new = self.interpolate_poses(pose_from, pose_to, t)
        
        if self.position_only:
            target_pos = pose_new[:3]
            q_solution, error = self.ik_solver.ik_position(target_pos, q_from)
        else:
            q_solution, error = self.ik_solver.ik_pose(pose_new, q_from)
        
        if error < 0.08 and self.check_config_validity(q_solution):
            return pose_new, q_solution, True
        else:
            return pose_from, q_from, False
    
    def connect_trees(self, tree1, tree2, pose_new, q_new):
        distances = [self.cartesian_distance(pose_new, node['pose']) for node in tree2]
        nearest_idx = np.argmin(distances)
        pose_nearest = tree2[nearest_idx]['pose']
        q_nearest = tree2[nearest_idx]['config']
        
        path_poses = [pose_new]
        path_configs = [q_new]
        current_pose = pose_new
        current_q = q_new
        
        while self.cartesian_distance(current_pose, pose_nearest) > self.step_size:
            current_pose, current_q, success = self.extend_towards(
                current_pose, pose_nearest, current_q
            )
            if not success:
                return None
            path_poses.append(current_pose)
            path_configs.append(current_q)
        
        #final step to connect
        if self.cartesian_distance(current_pose, pose_nearest) > 1e-6:
            current_pose, current_q, success = self.extend_towards(
                current_pose, pose_nearest, current_q
            )
            if success and self.cartesian_distance(current_pose, pose_nearest) < self.goal_threshold:
                path_poses.append(pose_nearest)
                path_configs.append(q_nearest)
                return path_poses, path_configs, nearest_idx
        else:
            return path_poses, path_configs, nearest_idx
        
        return None
    
    def plan_path(self, q_start, target_pose):
        if self.position_only:
            pose_start = self.get_ee_position(q_start)
            pose_goal = np.array(target_pose[:3])
            q_target, error = self.ik_solver.ik_position(pose_goal, q_start)
            # print(f"Target position IK solution found with error: {error}")
            # if error > 0.08 or not self.check_config_validity(q_target):
            #     print(f"Target pose is not reachable!, IK error: {error}, config validity: {self.check_config_validity(q_target)}")
            #     return None
        else:
            pose_start = self.get_ee_pose(q_start)
            pose_goal = target_pose
            q_target, error = self.ik_solver.ik_pose(target_pose, q_start)
            print(f"Target position IK solution found with error: {error}")
            if error > 0.08 or not self.check_config_validity(q_target):
                print(f"Target pose is not reachable!, IK error: {error}")
                return None
        
        tree_start = [{'pose': pose_start, 'config': q_start, 'parent': None, 'index': 0}]
        tree_goal = [{'pose': pose_goal, 'config': q_target, 'parent': None, 'index': 0}]
        
        print(f"Starting {'Position-only' if self.position_only else 'Full'} Cartesian RRT-Connect planning...")
        start_time = time.time()
        
        for iteration in range(self.max_iter):
            if iteration % 2 == 0:
                tree_a, tree_b = tree_start, tree_goal
            else:
                tree_a, tree_b = tree_goal, tree_start

            if len(tree_a) > 1:
                reference_q = tree_a[-1]['config']
            else:
                reference_q = tree_a[0]['config']
            
            pose_rand, q_rand = self.random_cartesian_pose(reference_q)
            
            distances = [self.cartesian_distance(pose_rand, node['pose']) for node in tree_a]
            nearest_idx = np.argmin(distances)
            pose_nearest = tree_a[nearest_idx]['pose']
            q_nearest = tree_a[nearest_idx]['config']
            
            pose_new, q_new, success = self.extend_towards(
                pose_nearest, pose_rand, q_nearest
            )
            
            if success and self.cartesian_distance(pose_new, pose_nearest) > 1e-6:
                new_node = {
                    'pose': pose_new,
                    'config': q_new,
                    'parent': nearest_idx,
                    'index': len(tree_a)
                }
                tree_a.append(new_node)

                connection_result = self.connect_trees([new_node], tree_b, pose_new, q_new)
                if connection_result is not None:
                    _, path_configs, target_idx = connection_result
                    
                    path_start = []
                    if iteration % 2 == 0:  # tree_a is tree_start
                        current_idx = len(tree_a) - 1
                        while current_idx is not None:
                            path_start.append(tree_a[current_idx]['config'])
                            current_idx = tree_a[current_idx]['parent']
                        path_start.reverse()
                    else:  # tree_b is tree_start
                        current_idx = target_idx
                        while current_idx is not None:
                            path_start.append(tree_b[current_idx]['config'])
                            current_idx = tree_b[current_idx]['parent']
                        path_start.reverse()
                    
                    path_goal = []
                    if iteration % 2 == 0:  # tree_b is tree_goal
                        current_idx = target_idx
                        while current_idx is not None:
                            path_goal.append(tree_b[current_idx]['config'])
                            current_idx = tree_b[current_idx]['parent']
                    else:  # tree_a is tree_goal
                        current_idx = len(tree_a) - 1
                        while current_idx is not None:
                            path_goal.append(tree_a[current_idx]['config'])
                            current_idx = tree_a[current_idx]['parent']
                    
                    full_path = path_start + path_configs[1:] + path_goal[1:]
                    
                    planning_time = time.time() - start_time
                    print(f"Cartesian planning successful! Time: {planning_time:.2f}s, "
                          f"Iterations: {iteration}, Path length: {len(full_path)}")
                    
                    return full_path
        
        print("Cartesian planning failed! Exceeded max iterations")
        return None

    def smooth_path(self, path, max_iterations=50):
        if len(path) < 3:
            return path
        
        smoothed_path = path.copy()
        
        for _ in range(max_iterations):
            if len(smoothed_path) < 3:
                break
            
            i = np.random.randint(0, len(smoothed_path) - 2)
            j = np.random.randint(i + 2, len(smoothed_path))
            
            if self.check_connect_directly(smoothed_path[i], smoothed_path[j]):
                smoothed_path = smoothed_path[:i+1] + smoothed_path[j:]
        
        return smoothed_path
    
    def check_connect_directly(self, q1, q2, num_checks=10):
        if self.position_only:
            pose1 = self.get_ee_position(q1)
            pose2 = self.get_ee_position(q2)
        else:   
            pose1 = self.get_ee_pose(q1)
            pose2 = self.get_ee_pose(q2)
        
        for i in range(num_checks + 1):
            t = i / num_checks
            pose_intermediate = self.interpolate_poses(pose1, pose2, t)
            
            q_ref = (1 - t) * q1 + t * q2

            if self.position_only:
                q_intermediate, error = self.ik_solver.ik_position(pose_intermediate, q_ref)
            else:
                q_intermediate, error = self.ik_solver.ik_pose(pose_intermediate, q_ref)
            
            if error > 0.01 or not self.check_config_validity(q_intermediate):
                return False
        
        return True


def path_to_trajectory(path, timesteps):
    path = np.array(path)
    if len(path) < 2:
        raise ValueError("You need at least 2 point in the path")
    
    distances = np.zeros(len(path))
    for i in range(1, len(path)):
        distances[i] = distances[i-1] + np.linalg.norm(path[i] - path[i-1])

    total_distance = distances[-1]
    if total_distance == 0:
        return np.tile(path[0], (timesteps + 1, 1))
    
    distances = distances / total_distance
    q_trajectory = np.zeros((timesteps + 1, 7))
    time_points = np.linspace(0, 1, timesteps + 1)

    for joint in range(7):
        q_trajectory[:, joint] = np.interp(time_points, distances, path[:, joint])
    
    return q_trajectory

def path_to_u(path, timesteps, use_synergies=False, S_matrix=None):
    q_trajectory = path_to_trajectory(path, timesteps)

    u_trajectory = np.zeros((timesteps, 7))
    for t in range(timesteps):
        u_trajectory[t] = q_trajectory[t+1] - q_trajectory[t]

    if use_synergies and S_matrix is not None:
        S = np.transpose(S_matrix)
        u_synergy = np.zeros((timesteps, S_matrix.shape[0]))
        for t in range(timesteps):
            u_synergy[t] = np.matmul(np.linalg.pinv(S), u_trajectory[t])
        return u_synergy

    return u_trajectory    

                
                    
