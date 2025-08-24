import numpy as np
from scipy.optimize import minimize
import mujoco
import matplotlib.pyplot as plt
import os
import pickle

from hopejr_IKSolver import IKSolver
from hopejr_RRTConnector import JointRRTConnectPlanner, CartesianRRTConnector, path_to_u, path_to_trajectory

class Optimizer:
    def __init__(self, timestep, s_matrix=None, q0=None, mujoco_model=None, mujoco_data=None, use_synergies=False, position_only=True):
        self.use_synergies = use_synergies
        #Synergies Matrix
        if use_synergies:
            self.S = np.transpose(s_matrix)
            self.num_components = len(s_matrix)
        else:
            self.S = np.eye(7)
            self.num_components = 7
        self.T = timestep

        self.position_only= position_only
        self.q_upper_limit = np.array([1.5707, 2.0943, 1.2217, 1.7453, 2.0943, 0.1745, 0.1745])
        self.q_lower_limit = np.array([-1.5707, 0, -1.5707, -0.2617, -0.8726, -0.5235, -0.8726])

        self.mj_model = mujoco_model
        self.mj_data = mujoco_data
        if self.mj_model is not None:
            self.ee_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "link7")
            self.obj_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "obj")

            self.obj_pos = self.mj_data.xpos[self.obj_id]
            # self.obstacle_pos = self.mj_data.xpos[self.obstacle_id]
            self.obstacle_poses = self.get_obstacle_positions()
            self.link_geom_ids = self.get_link_geom_ids()
            self.obstacle_geom_ids = self.get_obstacle_geom_ids()

            self.ik_solver = IKSolver(self.mj_model, self.mj_data) if self.mj_model is not None else None

            #joint space RRT-connect planner for u_init
            self.joint_rrt_planner = JointRRTConnectPlanner(
                mujoco_model=self.mj_model,
                mujoco_data=self.mj_data,
                step_size=0.05,
                max_iter=5000,
                goal_threshold=0.1
            )
            #cartesian space RRT-connect planner for u_init
            self.cartesian_rrt_planner = CartesianRRTConnector(
                mujoco_model=self.mj_model,
                mujoco_data=self.mj_data,
                ik_solver=self.ik_solver,
                step_size=0.05,
                max_iter=5000,
                goal_threshold=0.05,
                orientation_weight=0.3,
                position_only=self.position_only
            )
        else:
            self.joint_rrt_planner = None
            self.cartesian_rrt_planner = None

        #Set default value for initial pose
        if q0 is None:
            q0 = np.array([0.0, 0.0, 0.0, 0.0, 1.6, 0.0, 0.0])  
        self.q0 = q0
        self.EE_init = None
        self.EE_target = None
        self.target_pos = None

        self.lambda_jerk = 10
        self.lambda_EE = 100
        self.lambda_acc = 10
        self.lambda_path = 5
        self.lambda_joint = 10

        self.safety_margin = 0.08
        self.safety_margin_mesh = 0.015
        self.joint_safety_margin = 0.05
        self.link_names = ["link1", "link2", "link3", "link4", "link5", "link6", "link7"]
        self.link_ids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.link_names]

        
    def fk_position(self, q):
        if self.mj_model is None or self.mj_data is None:
            raise ValueError("Mujoco model is not loaded")
        
        current_qpos = self.mj_data.qpos.copy()
        
        self.mj_data.qpos[:7] = q
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        ee_pos = self.mj_data.xpos[self.ee_id].copy()
        
        self.mj_data.qpos = current_qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        return ee_pos
    
    def get_ee_pose(self, q):
        current_qpos = self.mj_data.qpos.copy()
        self.mj_data.qpos[:7] = q
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        ee_pos = self.mj_data.xpos[self.ee_id].copy()
        ee_quat = self.mj_data.xquat[self.ee_id].copy()  # [w, x, y, z]
        
        self.mj_data.qpos = current_qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        return np.concatenate([ee_pos, ee_quat])

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

    def get_link_positions(self, q):
        if self.mj_model is None or self.mj_data is None:
            raise ValueError("Mujoco model is not loaded")

        current_qpos = self.mj_data.qpos.copy()

        self.mj_data.qpos[:7] = q
        mujoco.mj_forward(self.mj_model, self.mj_data)
        link_positions = [self.mj_data.xpos[link_id].copy() for link_id in self.link_ids]
        self.mj_data.qpos = current_qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)

        return np.array(link_positions)

    def compute_trajectory(self, u):
        q = np.zeros((self.T+1, 7))
        q[0] = self.q0
        u = u.reshape((self.T, self.num_components))
        for t in range(self.T):
            q[t+1] = q[t] + np.matmul(self.S, u[t])
        
        return q

    #Core cost function definition
    def cost_function(self, u):
        q = self.compute_trajectory(u)
        cost = 0.0

        #======joint acceleration cost======
        joint_acc = np.zeros((self.T-1, 7))
        for t in range(self.T-1):
            joint_acc[t] = q[t+2] - 2*q[t+1] + q[t]
        joint_acc_cost = np.sum(np.linalg.norm(joint_acc, axis=1)**2)
        cost += self.lambda_acc * joint_acc_cost

        #======joint jerk cost======
        joint_jerk = np.zeros((self.T-2, 7))
        for t in range(self.T-2):
            joint_jerk[t] = q[t+3] - 3*q[t+2] + 3*q[t+1] - q[t]
        joint_jerk_cost = np.sum(np.linalg.norm(joint_jerk, axis=1)**2)
        cost += self.lambda_jerk * joint_jerk_cost

        #======path length cost======
        path_cost = self.path_cost(q)
        cost += self.lambda_path * path_cost

        #======EE pose cost======
        EE_current = self.fk_position(q[-1])
        cost += self.lambda_EE * np.sum((EE_current - self.EE_target)**2)

        #======joint limit cost======
        joint_limit_cost = self.joint_limit_cost(q)
        cost += self.lambda_joint * joint_limit_cost

        return cost

    def path_cost(self, q):
        ee_positions = np.zeros((len(q), 3))
        for t in range(self.T+1):
            ee_positions[t] = np.asarray(self.fk_position(q[t]))

        actual_path = 0
        for t in range(len(q)-1):
            actual_path += np.linalg.norm(ee_positions[t+1] - ee_positions[t])
        
        return actual_path

    def joint_limit_cost(self, q):
        cost = 0.0
        for t in range(len(q)):
            for i in range(7):
                qpos = q[t, i]
                qmin = self.q_lower_limit[i]
                qmax = self.q_upper_limit[i]

                dist_to_lower = qpos - qmin
                dist_to_upper = qmax - qpos

                if dist_to_lower < self.joint_safety_margin:
                    cost += (self.joint_safety_margin - dist_to_lower) ** 2
                if dist_to_upper < self.joint_safety_margin:
                    cost += (self.joint_safety_margin - dist_to_upper) ** 2
        return cost

    def get_u_init(self, q_init, target_position=None, target_pose=None, method="random"):
        if method == "random":
            u_initial = np.random.normal(0, 0.001, (self.T, self.num_components))

        elif method == "linear":
            q_target, final_error = self.ik_solver.ik_position(target_position=target_position, q_init=q_init)
            print(f"Target IK solution found with final error: {final_error}")
            print(f"ik result: {q_target}")

            q_trajectory = np.zeros((self.T+1, 7))
            q_trajectory[0] = q_init
            for t in range(1, self.T+1):
                alpha = t / self.T
                q_trajectory[t] = (1 - alpha) * q_init + alpha * q_target

            u_initial = np.zeros((self.T, self.num_components))
            for t in range(self.T):
                u_joint = q_trajectory[t+1] - q_trajectory[t]

                if self.use_synergies:
                    u_initial[t] = np.matmul(np.linalg.pinv(self.S), u_joint)
                else:
                    u_initial[t] = u_joint

        elif method == "joint_rrt":
            if self.joint_rrt_planner is None:
                print("Joint RRT planner is not initialized, use 'linear' instead")
                return self.get_u_init(q_init, target_position, method="linear")
            
            q_target, final_error = self.ik_solver.ik_position(target_position=target_position, q_init=q_init)
            print(f"Target IK solution found with final error: {final_error}")
            print(f"ik result: {q_target}")

            print("Planning in Joint space...")
            path = self.joint_rrt_planner.plan_path(q_init, q_target)
            if path is None:
                print("Joint RRT planner failed, use 'linear' instead")
                return self.get_u_init(q_init, target_position, method="linear")
            
            print("Smoothing path...")
            smoothed_path = self.joint_rrt_planner.smooth_path(path, max_iterations=50)
            print(f"Smoothed path length: {len(smoothed_path)}")

            u_initial = path_to_u(
                smoothed_path,
                self.T,
                use_synergies=self.use_synergies,
                S_matrix=self.S if self.use_synergies else None
            )

        elif method == "cartesian_rrt":
            if self.cartesian_rrt_planner is None:
                print("Cartesian RRT planner is not initialized, using 'joint_rrt' instead")
                return self.get_u_init(q_init, target_position, target_pose, method="joint_rrt")
            
            if target_pose is not None:
                target_pose_full = target_pose
            elif target_position is not None:
                current_pose = self.get_ee_pose(q_init)
                target_pose_full = np.concatenate([target_position, current_pose[3:7]])
            else:
                raise ValueError("Either target_position or target_pose must be provided")
            
            print("Planning in Cartesian space...")
            path = self.cartesian_rrt_planner.plan_path(q_init, target_pose_full)
            with open('cartesian_path.pkl', 'wb') as f:
                pickle.dump(path, f)
            
            if path is None:
                print("Cartesian RRT planner failed, using 'joint_rrt' instead")
                return self.get_u_init(q_init, target_position, target_pose, method="joint_rrt")
            
            print("Smoothing Cartesian path...")
            smoothed_path = self.cartesian_rrt_planner.smooth_path(path, max_iterations=50)
            print(f"Smoothed path length: {len(smoothed_path)}")
            with open('smoothed_cartesian_path.pkl', 'wb') as f:
                pickle.dump(smoothed_path, f)
            
            smoothed_q = path_to_trajectory(path=smoothed_path, timesteps=self.T)
            with open('smoothed_q.pkl', 'wb') as f:
                pickle.dump(smoothed_q, f)

            u_initial = path_to_u(
                smoothed_path, self.T,
                use_synergies=self.use_synergies,
                S_matrix=self.S if self.use_synergies else None
            )

        else:
            u_initial = np.zeros((self.T, self.num_components))

        print("U_initial is given!")
        return u_initial
    
    def q_upper_limit_condition(self, u):
        #Get the difference between upper limit and real q
        q = self.compute_trajectory(u)
        return (self.q_upper_limit - q).flatten()
    
    def q_lower_limit_condition(self, u):
        #Ger the difference between lower limit and real q
        q = self.compute_trajectory(u)
        return (q - self.q_lower_limit).flatten()

    def collision_avoidance(self, u):
        q = self.compute_trajectory(u)
        constraints_values = []
      
        for t in range(len(q)):
            # link_positions = self.get_link_positions(q[t])
            # for i, link_pos in enumerate(link_positions[2:]): #Collision check from link2
            #     for obstacle_name, obstacle_pos in self.obstacle_poses.items():
            #         distance = np.linalg.norm(link_pos - obstacle_pos)
            #         constraints_values.append(distance - self.safety_margin)

            geom_distance = self.check_geom_collision(q[t])
            constraints_values.append(geom_distance - self.safety_margin_mesh)

        return np.array(constraints_values)

    def check_geom_collision(self, q):
        if not self.link_geom_ids:
            raise ValueError("No link geom detected!")
        current_qpos = self.mj_data.qpos.copy()

        self.mj_data.qpos[:7] = q
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mujoco.mj_collision(self.mj_model, self.mj_data)

        min_distance = np.inf
        collision_found = False
        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            for link_geom_id in self.link_geom_ids:
                if ((geom1_id == link_geom_id and geom2_id in self.obstacle_geom_ids) or (geom1_id in self.obstacle_geom_ids and geom2_id == link_geom_id)):
                    distance = contact.dist
                    min_distance = min(min_distance, distance)
                    collision_found = True

        #If no collision, get the min distance between 2 geom
        if not collision_found:
            min_distance = self.get_link_obstacle_min_distance(q)

        self.mj_data.qpos[:] = current_qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)

        return min_distance

    def get_link_obstacle_min_distance(self, q):
        min_distance = np.inf
        try:
            for link_geom_id in self.link_geom_ids:
                for obstacle_geom_id in self.obstacle_geom_ids:
                    distance = mujoco.mj_geomDistance(self.mj_model, self.mj_data, link_geom_id, obstacle_geom_id, 10.0, None)

                    if distance >= 0:
                        min_distance = min(min_distance, distance)
        except Exception as e:
            print(f"Get min distance error: {e}")

        return min_distance 

    def _check_collision(self, q_trajectory):
        violations = 0
        # min_distance_point = np.inf
        min_distance_mesh = np.inf

        for t in range(len(q_trajectory)):
            # link_positions = self.get_link_positions(q_trajectory[t])
            # for i, link_pos in enumerate(link_positions[2:]): #Collision check from link2
            #     for obstacle_name, obstacle_pos in self.obstacle_poses.items():
            #         distance = np.linalg.norm(link_pos - obstacle_pos)
            #         if distance < self.safety_margin:
            #             violations += 1
            #             print(f"Collision! ({self.link_names[i]})->({obstacle_name}): {distance:.4f}")
            #         min_distance_point = min(min_distance_point, distance)

            geom_distance = self.check_geom_collision(q_trajectory[t])
            if geom_distance < self.safety_margin_mesh:
                violations += 1
                print(f"Mesh Collision! dis: {geom_distance:.4f}")
            min_distance_mesh = min(min_distance_mesh, geom_distance)
            
        print(f"Collision violations: {violations}")
        # print(f"Min origin-obstacle distance: {min_distance_point:.4f}")
        print(f"Min mesh-obstacle distance: {min_distance_mesh:.4f}")

    #Core optimizer definition
    def optimize(self, q0, EE_target, init_method="random", target_pose=None):
        self.q0 = q0
        self.EE_init = self.fk_position(q0)

        if target_pose is not None:
            self.EE_target = target_pose[:3]
            target_pose_full = target_pose
            target_position = target_pose[:3]
        elif len(EE_target) == 7:
            self.EE_target = EE_target[:3]
            target_pose_full = EE_target
            target_position = EE_target[:3]
        else:
            self.EE_target = EE_target.copy()
            target_pose_full = None
            target_position = EE_target
        
        self.u_initial = self.get_u_init(
            q_init=self.q0,
            target_position=target_position,
            target_pose=target_pose_full,
            method=init_method)

        #Constrains
        self.constraints = [
            {'type': 'ineq', 'fun': self.q_upper_limit_condition},
            {'type': 'ineq', 'fun': self.q_lower_limit_condition},
            {'type': 'ineq', 'fun': self.collision_avoidance}
        ]

        result = minimize(self.cost_function, 
                          self.u_initial.flatten(), 
                          constraints=self.constraints, 
                          method="SLSQP",
                          options={
                              'ftol': 1e-7,
                              'maxiter': 1000
                          })
        self.u_opt = result.x.reshape(self.T, self.num_components)
        self.q_opt = self.compute_trajectory(self.u_opt)
        EE_final = self.EE_init = self.fk_position(self.q_opt[-1])
        EE_error_final = np.linalg.norm(EE_final - self.EE_target)
        print(f"EE error:{EE_error_final}")

        self._check_collision(self.q_opt)

        return self.q_opt
