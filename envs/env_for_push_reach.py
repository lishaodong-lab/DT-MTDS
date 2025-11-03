import gym
import pybullet as p
import pybullet_data
import math
import numpy as np
from gym import spaces
import os
import time
from math import sqrt
from gym.utils import seeding
import sys
from pcrl.buffer import Buffer
import random

class ur5Env_1(gym.Env):
    metadata = {'render.modes': ['human']}
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ur5Env_1, cls).__new__(cls)
        return cls._instance

    def __init__(self, is_render=True, is_good_view=True, move_step=0.002):
     if not hasattr(self, 'initialized'):  

        self.is_render = is_render
        self.is_good_view = is_good_view
        self.move_step = move_step
        self.step_counter = 0
        self.stage = 1
        self.G = 0
        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.initialized = True  

        self._max_episode_steps = 300

        self.reward = 0
        self.done = False
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.x_low_obs = 0.48
        self.x_high_obs = 0.85
        self.y_low_obs = -0.21
        self.y_high_obs = 0.21
        self.z_low_obs = -1
        self.z_high_obs = 1

        self.target_x_low_obs = -1
        self.target_x_high_obs = 1
        self.target_y_low_obs = -1
        self.target_y_high_obs = 1
        self.target_z_low_obs = -1
        self.target_z_high_obs = 1

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

        self.observation_space = spaces.Box(
            low=np.array(
                [self.x_low_obs, self.y_low_obs, self.z_low_obs,
                 self.target_x_low_obs, self.target_y_low_obs, self.target_z_low_obs 
               ],
                dtype=float),
            high=np.array(
                [self.x_high_obs, self.y_high_obs, self.z_high_obs,
                 self.target_x_high_obs,self.target_y_high_obs, self.target_z_high_obs
                 ]),
            dtype=float
        )

        self.x_low_action = -1
        self.x_high_action = 1
        self.y_low_action = -1
        self.y_high_action = 1
        self.z_low_action = -1
        self.z_high_action = 1

        self.action_space = spaces.Box(low=np.array(
            [self.z_low_action, self.y_low_action, self.x_low_action]),
            high=np.array([self.z_high_action, self.y_high_action, self.x_high_action]),
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
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.loadURDF("plane.urdf", basePosition=[0, 0, -0.65])
        p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])
        p.loadURDF("tray/tray.urdf", basePosition=[0.62, 0, 0.0])
        target_positions_1 = [
              (0.665, 0.0, 0.02),
              (0.665, 0.045, 0.02),
              (0.665, -0.045, 0.02),
              (0.71, 0.0, 0.02)
                           ]
        target_positions_2 = [
              (0.665, 0.0, 0.02),
              (0.665, 0.045, 0.02),
              (0.62, 0.0, 0.02),
              (0.71, 0.0, 0.02)
                           ]
        target_positions_3 = [
              (0.665, 0.0, 0.02),
              (0.665, -0.045, 0.02),
              (0.62, 0.0, 0.02),
              (0.71, 0.0, 0.02)
                           ]
        target_positions = [target_positions_1, target_positions_2, target_positions_3]
        selected_position = random.choice(target_positions)
        self.target_cube = p.loadURDF("cube.urdf", basePosition=selected_position[0])
        self.cube1 = p.loadURDF("cube2.urdf", basePosition=selected_position[1])
        self.cube2 = p.loadURDF("cube2.urdf", basePosition=selected_position[2])
        self.cube3 = p.loadURDF("cube2.urdf", basePosition=selected_position[3])
        self.UR5 = p.loadURDF("franka_panda/urpeg.urdf", basePosition=np.array([0, 0, 0]), useFixedBase=True)
        self.obstacle_ids = [self.cube1, self.cube2, self.cube3]
        self.target_half_extents = [0.04, 0.04, 0.04]
        p.resetDebugVisualizerCamera(
            cameraDistance=0.3,
            cameraYaw=90,
            cameraPitch=-50,
            cameraTargetPosition=[0.8, 0.0, 0.1]
        )

        p.changeVisualShape(self.target_cube, -1, rgbaColor = [1, 0, 0, 1])   # 红色

        init_position_1 = [0.6, 0.02, 0.07] 
        init_position_2 = [0.6, -0.1, 0.07]
        init_position_3 = [0.6, 0.1, 0.07]
        if selected_position == target_positions_1:
            init_position = init_position_1
        elif selected_position == target_positions_2:
            init_position = init_position_2
        elif selected_position == target_positions_3:
            init_position = init_position_3


        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356]
        for j in range(7):
            p.resetJointState(self.UR5, j, rest_poses[j])
        init_postrue_vary = [0.0, 0.0, 0.0]
        init_posture = np.array(p.getLinkState(self.UR5, 7)[5], dtype=float).reshape(4, 1)


        position_start, postrue_start, res = self.get_position_r_vary(init_position, init_posture, E=init_postrue_vary,
                                                                      x=0.036)  # 初始姿态偏置
        jointposes = p.calculateInverseKinematics(self.UR5, 7, position_start, postrue_start, maxNumIterations=100)  
        for j in range(7):
            p.resetJointState(self.UR5, j, jointposes[j])  

        self.postrue = np.array(p.getLinkState(self.UR5, 6)[5], dtype=float).reshape(4, 1)
        self.position = np.array(p.getLinkState(self.UR5, 6)[4], dtype=float).reshape(3, 1)


        self.position_target_cube = p.getBasePositionAndOrientation(self.target_cube)
        self.position_cube1 = p.getBasePositionAndOrientation(self.cube1)
        self.position_cube2 = p.getBasePositionAndOrientation(self.cube2)
        self.position_cube3 = p.getBasePositionAndOrientation(self.cube3)


        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        p.stepSimulation()
        result = self.check_obstacles_around_target(
                self.target_cube, self.obstacle_ids, self.target_half_extents, distance_threshold=0.02
            )

        self.observation_init = np.array(
            [self.position[0][0] * 10, self.position[1][0]* 100, result[0],
             result[1], result[2], 
             result[3]],
            dtype=float)
        

        return self.observation_init

    def reset2(self):


        init_position = [0.52, -0.08, 0.2]  
        init_postrue_vary = [0.0, 0.0, 0.0]

        init_posture = np.array(p.getQuaternionFromEuler([0.0, math.pi / 2, 0.0]), dtype=float).reshape(4, 1)

        self.move(init_position, self.postrue)
        
        

        self.postrue = np.array(p.getLinkState(self.UR5, 6)[5], dtype=float).reshape(4, 1)
        self.position = np.array(p.getLinkState(self.UR5, 6)[4], dtype=float).reshape(3, 1)


    def check_obstacles_around_target(self, target_id, obstacle_ids, target_half_extents, distance_threshold=0.02):
        target_pos, target_orn = p.getBasePositionAndOrientation(target_id)
    

        rot_matrix = p.getMatrixFromQuaternion(target_orn)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
    

        directions_local = {
            "front": np.array([1, 0, 0]),  
            "back": np.array([-1, 0, 0]),  
            "left": np.array([0, 1, 0]),   
            "right": np.array([0, -1, 0])  
        }
    

        face_centers = {}
        for direction, local_dir in directions_local.items():

            world_dir = rot_matrix @ local_dir

            face_center = np.array(target_pos) + world_dir * target_half_extents[0]
            face_centers[direction] = face_center
    

        result_vector = np.zeros(4)
    
        for i, direction in enumerate(["front", "back", "left", "right"]):
            face_center = face_centers[direction]
            min_distance = float('inf')

            for obstacle_id in obstacle_ids:
                obstacle_pos, _ = p.getBasePositionAndOrientation(obstacle_id)
                distance = np.linalg.norm(np.array(obstacle_pos) - np.array(face_center))
                if distance < min_distance:
                    min_distance = distance
        
            if min_distance < distance_threshold:
                result_vector[i] = 1
    
        return result_vector






    def step(self, action):



        self.position = np.array(p.getLinkState(self.UR5, 7)[4], dtype=float).reshape(3, 1)

        self.distancez = abs(self.position[2][0] - self.position_target_cube[0][2])

        self.distance2 = sqrt((self.position_cube1[0][0] - self.position_target_cube[0][0]) ** 2 + (
                    self.position_cube1[0][1] - self.position_target_cube[0][1]) ** 2)
        self.distance3 = sqrt((self.position_cube2[0][0] - self.position_target_cube[0][0]) ** 2 + (
                    self.position_cube2[0][1] - self.position_target_cube[0][1]) ** 2)
        self.distance4 = sqrt((self.position_cube3[0][0] - self.position_target_cube[0][0]) ** 2 + (
                    self.position_cube3[0][1] - self.position_target_cube[0][1]) ** 2)
        
        self.distance5 = sqrt((self.position[0][0]-self.position_target_cube[0][0])**2\
                                  +(self.position[1][0]-self.position_target_cube[0][1])**2\
                                  +(self.position[2][0]-self.position_target_cube[0][2])**2)
        
        self.distance1 = sqrt((self.position[0][0] - self.position_target_cube[0][0]) ** 2 + (
                    self.position[1][0] - self.position_target_cube[0][1]) ** 2)


        if self.stage == 1:

            dx = 0.005 * action[0]
            dy = 0.005 * action[1]
            dz = 0
            self.position_target = self.position + np.array([dx, dy, dz]).reshape(3, 1)
            self.position_target[2] = 0.07
            self.move(self.position_target, self.postrue)
        else:
            dx = 0.005 * action[0]
            dy = 0.005 * action[1]
            dz = 0.005 * action[2]
            self.position_target = self.position + np.array([dx, dy, dz]).reshape(3, 1)
            self.move(self.position_target, self.postrue)


        self.contact2 = p.getContactPoints(self.UR5, self.cube1)
        self.contact3 = p.getContactPoints(self.UR5, self.cube2)
        self.contact4 = p.getContactPoints(self.UR5, self.cube3)


        self.step_counter += 1
        print(self.distance5) 
        state, reward, done, info = self._reward()
        return state, reward, done, info



    def straight(self, position, postrue, dy):  
        position_need = self.get_position_p(position, postrue, dy)
        self.move(position_need, postrue)

        return position_need, postrue

    def straight_y(self, position, postrue, dy):  
        position_need = self.get_position_y(position, postrue, dy)
        self.move(position_need, postrue)

        return position_need, postrue

    def straight_z(self, position, postrue, dy): 
        position_need = self.get_position_z(position, postrue, dy)
        self.move(position_need, postrue)

        return position_need, postrue

    def move(self, position, postrue):
        
        jointposes = p.calculateInverseKinematics(self.UR5, 7, position, postrue, maxNumIterations=100)
        p.setJointMotorControlArray(self.UR5, list([0,1,2,3,4,5,6]), p.POSITION_CONTROL
                                    , list(jointposes))
        n = 100
        while (n):
            p.stepSimulation()
            n = n - 1


    def get_position_p(self, position, posture, dy):  

        matrix = np.array(p.getMatrixFromQuaternion(posture)).reshape(3, 3)
        res = np.array(position).reshape(3, 1)
        res += matrix[:, 0].reshape(3, 1) * dy
        return res

    def get_position_y(self, position, posture, dy):  

        matrix = np.array(p.getMatrixFromQuaternion(posture)).reshape(3, 3)
        res = np.array(position).reshape(3, 1)
        res += matrix[:, 1].reshape(3, 1) * dy
        return res

    def get_position_z(self, position, posture, dy):  

        matrix = np.array(p.getMatrixFromQuaternion(posture)).reshape(3, 3)
        res = np.array(position).reshape(3, 1)
        res += matrix[:, 2].reshape(3, 1) * dy
        return res

    def _reward(self):


        self.result = self.check_obstacles_around_target(
                self.target_cube, self.obstacle_ids, self.target_half_extents, distance_threshold=0.05
            )

        self.position_target_cube = p.getBasePositionAndOrientation(self.target_cube)
        self.position_cube1 = p.getBasePositionAndOrientation(self.cube1)
        self.position_cube2 = p.getBasePositionAndOrientation(self.cube2)
        self.position_cube3 = p.getBasePositionAndOrientation(self.cube3)


        self.distance_a = sqrt((self.position_cube1[0][0] - self.position_target_cube[0][0])**2 + (self.position_cube1[0][1] - self.position_target_cube[0][1])**2)
        self.distance_b = sqrt((self.position_cube2[0][0] - self.position_target_cube[0][0])**2 + (self.position_cube2[0][1] - self.position_target_cube[0][1])**2)
        self.distance_c = sqrt((self.position_cube3[0][0] - self.position_target_cube[0][0])**2 + (self.position_cube3[0][1] - self.position_target_cube[0][1])**2)


        self.num_obstacles = np.count_nonzero(self.result)




        self.c1 = (self.distance_a > 0.1 and self.distance_b > 0.1)
        self.c2 = (self.distance_a > 0.1 and self.distance_c > 0.1)
        self.c3 = (self.distance_b > 0.1 and self.distance_c > 0.1)

        self.distance5 = sqrt((self.position[0][0]-self.position_target_cube[0][0])**2\
                                  +(self.position[1][0]-self.position_target_cube[0][1])**2\
                                  +(self.position[2][0]-self.position_target_cube[0][2])**2)



        terminated = bool(
            self.position[0][0] < self.x_low_obs
            or self.position[0][0] > self.x_high_obs
            or self.position[1][0] < self.y_low_obs
            or self.position[1][0] > self.y_high_obs
            or self.position[2][0] < self.z_low_obs
            or self.position[2][0] > self.z_high_obs
            or self.position_target_cube[0][0] < self.target_x_low_obs
            or self.position_target_cube[0][0] > self.target_x_high_obs
            or self.position_target_cube[0][1] < self.target_y_low_obs
            or self.position_target_cube[0][1] > self.target_y_high_obs
        )



        if terminated:
            self.reward = float(-2000)
            self.done = True

        elif self.step_counter >= self._max_episode_steps:
            self.reward = float(-2000)
            self.done = True



        elif self.stage == 1:
            if self.num_obstacles == 2:

                self.reward = 500 - self.distance5

            if self.num_obstacles < 2:
                self.done1 = True
                self.stage = 0
                self.reward = float(2500)
                self.done = False
                for _ in range(10):
                  dx = 0.0
                  dy = 0.0
                  dz = 0.005
                  self.position = np.array(p.getLinkState(self.UR5, 6)[4], dtype=float).reshape(3, 1)
                  self.position_target = self.position + np.array([dx, dy, dz]).reshape(3, 1)
                  self.move(self.position_target, self.postrue)
                self.reset2()
            else:
                self.reward = -self.distance5  
                self.done = False
            



        elif self.stage == 0:
            for i in range(10):
                self.contact1 = p.getContactPoints(self.UR5, self.target_cube)

            self.reward = -self.distance5


            if self.contact1:
                self.reward = float(5000)
                self.done = True

        info = {}


        if self.stage == 1:

         self.observation = np.array(
            [self.position[0][0]* 10, self.position[1][0]* 100, self.result[0],
             self.result[1], self.result[2],
             self.result[3]],
            dtype=float)
         
        else:
        
         self.observation = np.array(
            [self.position[0][0]* 100, self.position[1][0]* 100, self.position[2][0]*100,
             self.position_target_cube[0][0]* 100, self.position_target_cube[0][1]* 100,
              self.position_target_cube[0][2]* 100],
            dtype=float)




        return self.observation, self.reward, self.done, info

    def infomation(self):
        return self.stage

    def information(self):
        return self.done1

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


        return state, done, self.stage
