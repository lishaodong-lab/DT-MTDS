import gym
import pybullet as p
import pybullet_data
import math
import numpy as np
from gym import spaces
import os
import time
import sys
from math import sqrt
from gym.utils import seeding
from pcrl.buffer import Buffer
import random


class ur5Env_1_search(gym.Env):
    metadata = {'render.modes': ['human']}


    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ur5Env_1_search, cls).__new__(cls)
        return cls._instance

    def __init__(self, is_render=True, is_good_view=True, move_step=0.003):
     if not hasattr(self, 'initialized'):  

    
        self.is_render = is_render
        self.is_good_view = is_good_view
        self.move_step = move_step
        self.step_counter = 0
        self.stage = 1
        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.initialized = True  

        self._max_episode_steps = 500
        self.hole_position = np.array([0.6, 0.1, 0.07])

        self.reward = 0
        self.done = False
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.x_low_obs = 0.48 * 100
        self.x_high_obs = 0.72 * 100
        self.y_low_obs = -0.12 * 100
        self.y_high_obs = 0.12 * 100
        self.z_low_obs = 0.0
        self.z_high_obs = 0.35 * 100

        self.target_x_low_obs = 0.48 * 100
        self.target_x_high_obs = 0.72 * 100
        self.target_y_low_obs = -0.15 * 100
        self.target_y_high_obs = 0.15 * 100
        self.target_z_low_obs = 0.0 * 100
        self.target_z_high_obs = 0.2 * 100

        self.x1_low_obs = 0
        self.x1_high_obs = 1
        self.y1_low_obs = -1
        self.y1_high_obs = 1

        self.x2_low_obs = 0
        self.x2_high_obs = 1
        self.y2_low_obs = -1
        self.y2_high_obs = 1

        self.x3_low_obs = 0
        self.x3_high_obs = 1
        self.y3_low_obs = -1
        self.y3_high_obs = 1

        # 状态空间
        self.observation_space = spaces.Box(
            low=np.array(
                [self.x_low_obs, self.y_low_obs, self.z_low_obs,
                 self.target_x_low_obs, self.target_y_low_obs, self.target_z_low_obs
                ],
                dtype=float),
            high=np.array(
                [self.x_low_obs, self.y_low_obs, self.z_low_obs,
                 self.target_x_high_obs, self.target_y_high_obs, self.target_z_high_obs
                 ]),
            dtype=float
        )

        self.z = 0
        self.x_low_action = -1
        self.x_high_action = 1
        self.y_low_action = -1
        self.y_high_action = 1
        self.z_low_action = -1
        self.z_high_action = 1


        self.action_space = spaces.Box(low=np.array(
            [self.z_low_action, self.y_low_action, self.x_low_action, -0.12]),
            high=np.array([self.z_high_action, self.y_high_action, self.x_high_action, 0.12]),
            dtype=np.float32)


        self.human = 0
        self.seed()



    def quaternion_rotation(self, q1, q2):
        r1 = q1[3]
        r2 = q2[3]
        v1 = np.array([q1[0], q1[1], q1[2]])
        v2 = np.array([q2[0], q2[1], q2[2]])

        r = r1 * r2 - np.dot(v1, v2)
        v = r1 * v2 + r2 * v1 + np.cross(v1, v2)
        q = np.array([v[0], v[1], v[2], r])

        return q

    def get_position_r_vary(self, position, posture, E, x):     
        a = [key for i in posture for key in i]      
        need = np.array(p.getQuaternionFromEuler(E)).reshape(4, 1).tolist()   
        b = [key for j in need for key in j]    
        pos = np.array(self.quaternion_rotation(a, b)).reshape(4, 1)  

        matrix = np.array(p.getMatrixFromQuaternion(posture), dtype=float).reshape(3, 3)   
        dy = x  
        res = np.array(position, dtype=float).reshape(3, 1)  
        res += matrix[:, 0].reshape(3, 1) * dy  

        matrix_new = np.array(p.getMatrixFromQuaternion(pos)).reshape(3, 3)   
        dyy = -x   
        position_need = np.array(res).reshape(3, 1)   
        position_need += matrix_new[:, 0].reshape(3, 1) * dyy  
        return position_need, pos, res   

    def reset(self):

        self.step_counter = 0
        self.stage = 1
        self.done = False
        self.done1 = False

        self.into = 0
        self.find = 0
        self.finish_task_one = 0
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.loadURDF("plane.urdf", basePosition=[0, 0, 0])


        self.UR5 = p.loadURDF("franka_panda/urpeg.urdf", basePosition=np.array([0, 0, 0]), useFixedBase=True)
        hole_urdf_path = "/home/hole/urdf/hole.urdf"
        self.hole = p.loadURDF(hole_urdf_path, basePosition = np.array([0.6, 0.1, 0.035]), useFixedBase=True)

        p.resetDebugVisualizerCamera(
            cameraDistance=0.2,
            cameraYaw=90,
            cameraPitch=-10,
            cameraTargetPosition=[0.62, 0.05, 0.2]
        )


        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356]

        for j in range(7):
            p.resetJointState(self.UR5, j, rest_poses[j])


        target_positions_1 = [
              (0.55, 0.05, 0.155),
              (0.55, 0.09, 0.155),
              (0.59, 0.05, 0.155),
              (0.59, 0.09, 0.155)
                           ]
        
        init_position = random.choice(target_positions_1)


        init_postrue_vary = [0.0, 0.0, 0.0]

        init_posture = np.array(p.getLinkState(self.UR5, 7)[5], dtype=float).reshape(4, 1)



        position_start, postrue_start, res = self.get_position_r_vary(init_position, init_posture, E=init_postrue_vary,
                                                                      x=0.036) 
        jointposes = p.calculateInverseKinematics(self.UR5, 7, position_start, postrue_start, maxNumIterations=100)  
        for j in range(7):
            p.resetJointState(self.UR5, j, jointposes[j])  





        self.peg_position = np.array(p.getLinkState(self.UR5, 7)[4], dtype=float).reshape(3, 1)
        self.peg_postrue = np.array(p.getLinkState(self.UR5, 7)[5], dtype=float).reshape(4, 1)  

        euler_angles = p.getEulerFromQuaternion(self.peg_postrue)
        roll, pitch, yaw = euler_angles 

        z_bottom = self.peg_position[2] - (0.08 / 2)  
        x_bottom = self.peg_position[0]               
        y_bottom = self.peg_position[1]              
        x_bottom += (0.08 / 2) * math.sin(roll)  

        peg_bottom_position = (x_bottom, y_bottom, z_bottom)
        self.hole_position = p.getBasePositionAndOrientation(self.hole)   

        


        self.postrue = np.array(p.getLinkState(self.UR5, 7)[5], dtype=float).reshape(4, 1)   
        self.position = np.array(p.getLinkState(self.UR5, 7)[4], dtype=float).reshape(3, 1)  
        axis_state = p.getLinkState(self.UR5, 7, computeForwardKinematics=True)
        axis_orn = axis_state[1] 
        axis_euler = p.getEulerFromQuaternion(axis_orn) 
        current_yaw = axis_euler[2] 

        link_pos = axis_state[4]  
        link_orn = axis_state[5]  
        axis_length = 0.1 
        bottom_pos_local = [0, 0, axis_length/2]  


        self.bottom_pos_world, _ = p.multiplyTransforms(
        link_pos, link_orn,          
        bottom_pos_local, [0,0,0,1]  
                  )
        peg_bottom_position = (self.bottom_pos_world[0], self.bottom_pos_world[1], self.bottom_pos_world[2])

        target_yaw = round(current_yaw / (math.pi/2)) * (math.pi/2) 
        angle_diff = target_yaw - current_yaw  


        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)   


        axis_link_idx_7 = 7  
        p.changeVisualShape(self.UR5, axis_link_idx_7, rgbaColor=[1.0, 0.0, 0.0, 1.0])  





        p.stepSimulation()   

        self.observation_init = np.array(
                [peg_bottom_position[0] * 100, peg_bottom_position[1] * 100, peg_bottom_position[2] * 100,
                     self.position[0][0] * 100, self.position[1][0] * 100, self.position[2][0] * 100],
                dtype=float)


        return self.observation_init




    def step(self, action):
        
        axis_state = p.getLinkState(self.UR5, 7, computeForwardKinematics=True)
        link_pos = axis_state[4]  
        link_orn = axis_state[5]  
        axis_length = 0.1  
        self.bottom_pos_local = [0, 0, axis_length/2] 

        self.bottom_pos_world, _ = p.multiplyTransforms(
        link_pos, link_orn,         
        self.bottom_pos_local, [0,0,0,1]  
                  )


        peg_bottom_position = (self.bottom_pos_world[0], self.bottom_pos_world[1], self.bottom_pos_world[2])
        self.hole_position = p.getBasePositionAndOrientation(self.hole)   

        self.hole_top_position = (self.hole_position[0][0], self.hole_position[0][1], self.hole_position[0][2] + 0.04)  
        self.position = np.array(p.getLinkState(self.UR5, 7)[4], dtype=float).reshape(3, 1)

        self.distance = sqrt((peg_bottom_position[0]-self.hole_top_position[0])**2\
                        +(peg_bottom_position[1]-self.hole_top_position[1])**2\
                        +(peg_bottom_position[2]-self.hole_top_position[2])**2)   

        self.distance1 = sqrt((peg_bottom_position[0] - self.hole_top_position[0]) ** 2 \
                             + (peg_bottom_position[1] - self.hole_top_position[1]) ** 2)   
        self.distancez = abs(peg_bottom_position[2] - (self.hole_position[0][2]-0.03))      



        x = peg_bottom_position[0] - self.hole_top_position[0]
        y = peg_bottom_position[1] - self.hole_top_position[1]
        z = peg_bottom_position[2] - self.hole_top_position[2]
        d = sqrt(x**2 + y**2 +z**2)
        self.current_joint6 = p.getJointState(self.UR5, 6)[0]
        equivalent_distance_D = self.get_joint6_equivalent_distance(self.current_joint6)

        dx = 0.005 * action[0]
        dy = 0.005 * action[1]
        dz = 0.005 * action[2]
        rz = 0.1 * action[3]

        

        self.position_target = self.position + np.array([dx, dy, dz]).reshape(3, 1)   


        self.current_joint6 = p.getJointState(self.UR5, 6)[0]

    
        self.target_joint6 = self.current_joint6 + rz
    
        self.target_joint6 = self.normalize_joint6(self.target_joint6)

        self.postrue = np.array(p.getLinkState(self.UR5, 7)[5], dtype=float).reshape(4, 1)
        self.move(target_position=self.position_target, target_posture=self.postrue, target_angle=self.target_joint6, mode='position')  
        self.move(target_position=self.position_target, target_posture=self.postrue, target_angle=self.target_joint6, mode='rotation')  
        joint6_angle = p.getJointState(self.UR5, 6)[0]

        for i in range(10):
            self.contact = p.getContactPoints(self.UR5, self.hole)

        if self.stage == 1:
            if self.contact:
             self.done = True
            if d <= 0.01 and  equivalent_distance_D < 0.1: 
              self.find = 1
              
        elif self.stage == 0:
          if self.distancez < 0.04 and self.distance1 < 0.001:
                self.into = 1  


        self.step_counter += 1

        state, reward, done, info = self._reward()

        return state, reward, done, info

    def normalize_yaw(self, yaw):
        while yaw > math.pi:
            yaw -= 2 * math.pi
        while yaw < -math.pi:
            yaw += 2 * math.pi
        return yaw
    
    def normalize_joint6(self, angle):
        lower = -2.96
        upper = 2.96
        range = upper - lower
        angle = (angle - lower) % range + lower
        return angle
    
    def get_joint6_reward(self, joint6_angle):
        TARGET_ANGLE = -2.96
    

        JOINT_MIN = -2.96
        JOINT_MAX = 2.96
        JOINT_RANGE = JOINT_MAX - JOINT_MIN  
    
        def normalize_angle(angle):
            return ((angle - JOINT_MIN) % JOINT_RANGE) + JOINT_MIN
    
        normalized_angle = normalize_angle(joint6_angle)
    
        direct_distance = abs(normalized_angle - TARGET_ANGLE)
        equivalent_distance = min(direct_distance, JOINT_RANGE - direct_distance)
    
        reward = -equivalent_distance
    
        return reward
    
    def get_joint6_equivalent_distance(self, joint6_angle):
        TARGET_ANGLE = -2.96
    
        JOINT_MIN = -2.96
        JOINT_MAX = 2.96
        JOINT_RANGE = JOINT_MAX - JOINT_MIN  
    
        def normalize_angle(angle):
            return ((angle - JOINT_MIN) % JOINT_RANGE) + JOINT_MIN
    
        normalized_angle = normalize_angle(joint6_angle)
    
        direct_distance = abs(normalized_angle - TARGET_ANGLE)
        self.equivalent_distance = min(direct_distance, JOINT_RANGE - direct_distance)
    
        return self.equivalent_distance



    
    def move(self, target_position, target_posture, target_angle, mode='both'):

        current_joint_angles = [p.getJointState(self.UR5, i)[0] for i in range(7)]
    
        if mode in ['position', 'both']:

            jointposes = p.calculateInverseKinematics(self.UR5, 7, target_position, target_posture, maxNumIterations=100)
            p.setJointMotorControlArray(self.UR5, list([0,1,2,3,4,5,6]), p.POSITION_CONTROL
                                    , list(jointposes))
        
    
        if mode in ['rotation', 'both']:
            p.setJointMotorControlArray(
                bodyUniqueId=self.UR5,
                jointIndices=[0, 1, 2, 3, 4, 5],
                controlMode=p.POSITION_CONTROL,
                targetPositions=[p.getJointState(self.UR5, i)[0] for i in range(6)],
                forces=[500] * 6
            )
        
            p.setJointMotorControl2(
                bodyUniqueId=self.UR5,
                jointIndex=6,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_angle,
                force=300
            )
            print(target_angle)


        n = 100
        while (n):
            p.stepSimulation()
            n = n - 1


    def _reward(self):

        self.hole_position = p.getBasePositionAndOrientation(self.hole)
        axis_state = p.getLinkState(self.UR5, 7, computeForwardKinematics=True)
        link_pos = axis_state[4]  
        link_orn = axis_state[5] 
        axis_length = 0.1  
        self.bottom_pos_local = [0, 0, axis_length/2]  
        
        current_orn = p.getLinkState(self.UR5, 7)[1]
        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        error = p.getDifferenceQuaternion(current_orn, target_orn)
        orientation_error = np.sum(np.abs(error))

        self.bottom_pos_world, _ = p.multiplyTransforms(
        link_pos, link_orn,         
        self.bottom_pos_local, [0,0,0,1]  
                  )

        peg_bottom_position = (self.bottom_pos_world[0], self.bottom_pos_world[1], self.bottom_pos_world[2])
        self.current_joint6 = p.getJointState(self.UR5, 6)[0]
        self.reward_1 = self.get_joint6_reward(self.current_joint6)





        terminated = bool(
            self.position[0][0] * 100 < self.x_low_obs
            or self.position[0][0] * 100 > self.x_high_obs
            or self.position[1][0] * 100 < self.y_low_obs
            or self.position[1][0] * 100 > self.y_high_obs
            or self.position[2][0] * 100 < self.z_low_obs
            or self.position[2][0] * 100 > self.z_high_obs
        )

        info = {}
        self.contact1 = p.getContactPoints(self.UR5, self.hole)
        if terminated:
            self.reward = float(-10)
            self.done = True

        elif self.step_counter >= self._max_episode_steps:
            self.reward = float(-10)
            self.done = True

        elif self.stage == 1:
            if self.contact1:   
                self.done = True
            
            elif  self.find == 0:
                  
                  self.reward = -self.distance - orientation_error * 0.1
                  self.done = False
             
            elif  self.find == 1:  
                  self.stage = 0
                  self.reward = 2500.0
        
        elif self.stage == 0 and self.into == 0:
                self.reward =  -self.distance1 - abs(self.distancez)- self.reward_1

                self.done = False      
        

        elif self.into == 1 and self.stage == 0:

            self.reward = 5000
            self.done = True
        





        if self.stage == 1:
            self.observation = np.array(
                [peg_bottom_position[0] * 100, peg_bottom_position[1] * 100, peg_bottom_position[2] * 100,
                     self.position[0][0] * 100, self.position[1][0] * 100, self.position[2][0] * 100],
                dtype=float)
        else:
            self.observation = np.array(
                [peg_bottom_position[0] * 100, peg_bottom_position[1] * 100, peg_bottom_position[2] * 100,
                     0, 0, 0],
                dtype=float)
        return self.observation, self.reward, self.done, info


    def infomation(self):
        return self.stage


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        p.disconnect()

    def collect_demo(self, buffer, env, state, action):


        next_state, reward, done, _ = env.step(action)

        if self.stage == 0 and self.done1:
            buffer.append(state, action, reward, self.done1, next_state)
            self.done1 = 0
        else:
            buffer.append(state, action, reward, done, next_state)
        state = next_state


        return state, done, reward

