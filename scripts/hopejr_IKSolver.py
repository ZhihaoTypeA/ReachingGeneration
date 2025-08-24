from scipy.optimize import minimize, differential_evolution
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

class IKSolver:
    def __init__(self, mujoco_model, mujoco_data):
        self.mj_model = mujoco_model
        self.mj_data = mujoco_data
        self.ee_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "link7")

        self.q_lower = np.array([-1.5707, 0.0, -1.5707, -0.2617, -0.8726, -0.5235, -0.8726])
        self.q_upper = np.array([1.5707, 2.0943, 1.2217, 1.7453, 2.0943, 0.1745, 0.1745])

        self.n_joints = 7
        self.position_weight = 10.0
        self.orientation_weight = 0.3
        self.joint_regularization_weight = 0.5

    def fk(self, q):
        current_qpos = self.mj_data.qpos.copy()
        self.mj_data.qpos[:self.n_joints] = q

        mujoco.mj_forward(self.mj_model, self.mj_data)
        position = self.mj_data.xpos[self.ee_id].copy()
        quaternion = self.mj_data.xquat[self.ee_id].copy()

        self.mj_data.qpos[:self.n_joints] = current_qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)

        return position, quaternion
    
    def get_ee_pose(self, q):
        position, quaternion = self.fk(q)
        return np.concatenate([position, quaternion])

    def objective_position(self, q, target_position):
        position, _ = self.fk(q)
        position_error = np.linalg.norm(position - target_position)
        joint_regularization = self.joint_regularization_weight * np.linalg.norm(q)**2
        cost = self.position_weight * position_error + joint_regularization
        return cost
    
    def objective_pose(self, q, target_pose):
        position, quaternion = self.fk(q)
        target_position = target_pose[:3]
        target_quaternion = target_pose[3:7]
        
        position_error = np.linalg.norm(position - target_position)
        orientation_error = self.quaternion_distance(quaternion, target_quaternion)
        joint_regularization = self.joint_regularization_weight * np.linalg.norm(q)**2
        cost = (self.position_weight * position_error + 
                self.orientation_weight * orientation_error + 
                joint_regularization)
        
        return cost
    
    def quaternion_distance(self, q1, q2):
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        dot_product = np.abs(np.dot(q1, q2))
        dot_product = np.clip(dot_product, -1.0, 1.0)

        angle_diff = 2 * np.arccos(dot_product)
        return angle_diff
    
    def matrix_to_quaternion(self, R):
        rotation = Rotation.from_matrix(R)
        quat_scipy = rotation.as_quat()  # [x, y, z, w]
        return np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])  # [w, x, y, z]
    
    def quaternion_to_matrix(self, q):
        #input: [w, x, y, z], scipy needs [x, y, z, w]
        quat_scipy = np.array([q[1], q[2], q[3], q[0]])
        rotation = Rotation.from_quat(quat_scipy)
        return rotation.as_matrix()

    def ik_position(self, target_position, q_init=None):
        target_position = np.array(target_position)
        bounds = [(self.q_lower[i], self.q_upper[i]) for i in range(self.n_joints)]

        if q_init is None:
            q_init = np.random.uniform(self.q_lower, self.q_upper)

        result = minimize(
            fun=lambda q: self.objective_position(q, target_position),
            x0=q_init,
            bounds=bounds,
            options={'ftol': 1e-6, 'maxiter': 2000}
        )
        final_position, _ = self.fk(result.x)
        final_error = np.linalg.norm(final_position - target_position)
        return result.x, final_error
    
    def ik_pose(self, target_pose, q_init=None):
        if len(target_pose) == 7:
            # [x, y, z, qw, qx, qy, qz]
            target_pose_quat = np.array(target_pose)
        elif len(target_pose) == 6 and hasattr(target_pose[3], '__len__'):
            # [x, y, z, R]
            target_position = target_pose[:3]
            target_rotation = target_pose[3]
            target_quaternion = self.matrix_to_quaternion(target_rotation)
            target_pose_quat = np.concatenate([target_position, target_quaternion])
        else:
            raise ValueError("target_pose should be [x, y, z, qw, qx, qy, qz] or [position, rotation_matrix]")
        
        target_pose_quat[3:7] = target_pose_quat[3:7] / np.linalg.norm(target_pose_quat[3:7])
        
        bounds = [(self.q_lower[i], self.q_upper[i]) for i in range(self.n_joints)]
        
        if q_init is None:
            q_init = np.random.uniform(self.q_lower, self.q_upper)
        
        result = minimize(
            fun=lambda q: self.objective_pose(q, target_pose_quat),
            x0=q_init,
            bounds=bounds,
            options={'ftol': 1e-6, 'maxiter': 2000}
        )
        q_solution = result.x

        final_position, final_quaternion = self.fk(q_solution)
        position_error = np.linalg.norm(final_position - target_pose_quat[:3])
        orientation_error = self.quaternion_distance(final_quaternion, target_pose_quat[3:7])
        final_error = position_error + orientation_error
        
        return q_solution, final_error
    
