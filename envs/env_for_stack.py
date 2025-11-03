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





class ur5Env_1(gym.Env):
    metadata = {'render.modes': ['human']}


    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ur5Env_1, cls).__new__(cls)
        return cls._instance

    def __init__(self, is_render=True, is_good_view=True, move_step=0.003):
     if not hasattr(self, 'initialized'): 

    
        self.is_render = is_render
        self.is_good_view = is_good_view
        self.move_step = move_step
        self.step_counter = 0
        self.stage = 0
        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.initialized = True  

        self._max_episode_steps = 500

        self.reward = 0
        self.done = False
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.x_low_obs = 0.48 * 100
        self.x_high_obs = 0.72 * 100
        self.y_low_obs = -0.15 * 100
        self.y_high_obs = 0.15 * 100
        self.z_low_obs = 0.0
        self.z_high_obs = 0.3 * 100
        self.task_high_obs = 4.0
        self.task_low_obs = 0.0

        self.target_x_low_obs = 0.48 * 100
        self.target_x_high_obs = 0.72 * 100
        self.target_y_low_obs = -0.15 * 100
        self.target_y_high_obs = 0.15 * 100
        self.target_z_low_obs = 0.0 * 100
        self.target_z_high_obs = 0.17 * 100

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
                 self.x_low_obs, self.y_low_obs, self.z_low_obs
                ],
                dtype=np.float64),
            high=np.array(
                [self.x_high_obs, self.y_high_obs, self.z_high_obs,
                 self.x_high_obs, self.y_high_obs, self.z_high_obs
                ],
            dtype=np.float64)
        )

        self.z = 0
        self.x_low_action = -1
        self.x_high_action = 1
        self.y_low_action = -1
        self.y_high_action = 1
        self.z_low_action = -1
        self.z_high_action = 1
        self.G = 0


        self.action_space = spaces.Box(low=np.array(
            [self.z_low_action, self.y_low_action, self.x_low_action],dtype=np.float64),
            high=np.array([self.z_high_action, self.y_high_action, self.x_high_action],
            dtype=np.float64)
            )


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
        self.stage = 0
        self.done1 = False
        self.done2 = False
        self.done3 = False
        self.done4 = False
        self.done5 = False
        self.done6 = False
        self.done7 = False
        self.done = False
        self.G = 0
        self.finish_task_one = 0
        self.target_position = [0.6, 0.0, 0.0]
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.loadURDF("plane.urdf", basePosition=[0, 0, 0])

        target_positions_1 = [
              (0.6, 0.11, 0.02),
              (0.6, 0.15, 0.02),
              (0.56, 0.11, 0.02),
              (0.56, 0.15, 0.02)
                           ]
        

        target_positions_2 = [
              (0.6, -0.1, 0.02),
              (0.6, -0.14, 0.02),
              (0.56, -0.14, 0.02),
              (0.56, -0.1, 0.02)
                           ]
        
        selected_position_1 = random.choice(target_positions_1)
        selected_position_2 = random.choice(target_positions_2)


        self.cube1 = p.loadURDF("cube_one.urdf", basePosition=selected_position_1)
        self.cube2 = p.loadURDF("cube_two.urdf", basePosition=[selected_position_1[0], selected_position_1[1], 0.06])
        self.cube3 = p.loadURDF("cube_three.urdf", basePosition=selected_position_2)
        self.cube4 = p.loadURDF("cube_four.urdf", basePosition=[selected_position_2[0], selected_position_2[1], 0.06])


        p.changeVisualShape(self.cube1, -1, rgbaColor = [1, 0, 0, 1])   
        p.changeVisualShape(self.cube2, -1, rgbaColor = [0, 0, 1, 1])   
        p.changeVisualShape(self.cube3, -1, rgbaColor = [1, 1, 0, 1])   
        p.changeVisualShape(self.cube4, -1, rgbaColor = [0, 1, 0, 1])   



        self.UR5 = p.loadURDF("franka_panda/panda.urdf", basePosition=np.array([0, 0, 0]), useFixedBase=True)
        p.resetDebugVisualizerCamera(
            cameraDistance=0.4,
            cameraYaw=75,
            cameraPitch=-30,
            cameraTargetPosition=[0.6, 0, 0.0]
        )


        joint_active_ids = [0, 1, 2, 3, 4, 5, 6, 9, 10]

        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356]
        for j in range(7):
            p.resetJointState(self.UR5, j, rest_poses[j])
        p.resetJointState(self.UR5, 9, 0.08)
        p.resetJointState(self.UR5, 10, 0.08)

        init_position = [0.55, -0.1, 0.12]  

        init_postrue_vary = [0.0, 0.0, 0.0]

        init_posture = np.array(p.getLinkState(self.UR5, 11)[5], dtype=float).reshape(4, 1)

        position_start, postrue_start, res = self.get_position_r_vary(init_position, init_posture, E=init_postrue_vary,
                                                                      x=0.036)  
        jointposes = p.calculateInverseKinematics(self.UR5, 11, position_start, postrue_start, maxNumIterations=100)
        for j in range(7):
            p.resetJointState(self.UR5, j, jointposes[j])





        self.position_cube1 = p.getBasePositionAndOrientation(self.cube1)
        self.position_cube2 = p.getBasePositionAndOrientation(self.cube2)
        self.position_cube3 = p.getBasePositionAndOrientation(self.cube3)
        self.position_cube4 = p.getBasePositionAndOrientation(self.cube4)


        self.postrue = np.array(p.getLinkState(self.UR5, 11)[5], dtype=float).reshape(4, 1)
        self.position = np.array(p.getLinkState(self.UR5, 11)[4], dtype=float).reshape(3, 1)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        p.stepSimulation()


        self.observation_init = np.array(
            [self.position[0][0] * 100, self.position[1][0] * 100, self.position[2][0] * 100,
             self.position_cube4[0][0] * 100,self.position_cube4[0][1] * 100, self.position_cube4[0][2] * 100, 0],
            dtype=float)


        return self.observation_init



    def step(self, action):



        self.position = np.array(p.getLinkState(self.UR5, 11)[4], dtype=float).reshape(3, 1)
        self.position_cube1 = p.getBasePositionAndOrientation(self.cube1)
        self.position_cube2 = p.getBasePositionAndOrientation(self.cube2)
        self.position_cube3 = p.getBasePositionAndOrientation(self.cube3)
        self.position_cube4 = p.getBasePositionAndOrientation(self.cube4)
        

        self.cube4_distance = sqrt((self.position[0][0]-self.position_cube4[0][0])**2\
                                  +(self.position[1][0]-self.position_cube4[0][1])**2\
                                  +(self.position[2][0]-self.position_cube4[0][2])**2)

        self.cube4_distance1 = sqrt((self.position_cube4[0][0] - self.target_position[0]) ** 2 \
                                   + (self.position_cube4[0][1] - self.target_position[1]) ** 2)
        self.cube4_distancez = sqrt((self.position_cube4[0][2] - self.target_position[2]) ** 2)


        self.cube2_distance = sqrt((self.position[0][0]-self.position_cube2[0][0])**2\
                                  +(self.position[1][0]-self.position_cube2[0][1])**2\
                                  +(self.position[2][0]-self.position_cube2[0][2])**2)
        
        self.cube2_distance1 = sqrt((self.position_cube2[0][0] - self.position_cube4[0][0]) ** 2 \
                                   + (self.position_cube2[0][1] - self.position_cube4[0][1]) ** 2)        
        self.cube2_distancez = sqrt((self.position_cube2[0][2] - self.position_cube4[0][2]) ** 2)


        self.cube1_distance = sqrt((self.position[0][0]-self.position_cube1[0][0])**2\
                                  +(self.position[1][0]-self.position_cube1[0][1])**2\
                                  +(self.position[2][0]-self.position_cube1[0][2])**2)
        
        self.cube1_distance1 = sqrt((self.position_cube1[0][0] - self.position_cube2[0][0]) ** 2 \
                                   + (self.position_cube1[0][1] - self.position_cube2[0][1]) ** 2)        
        self.cube1_distancez = sqrt((self.position_cube1[0][2] - self.position_cube2[0][2]) ** 2)


        self.cube3_distance = sqrt((self.position[0][0]-self.position_cube3[0][0])**2\
                                  +(self.position[1][0]-self.position_cube3[0][1])**2\
                                  +(self.position[2][0]-self.position_cube3[0][2])**2)
        
        self.cube3_distance1 = sqrt((self.position_cube3[0][0] - self.position_cube1[0][0]) ** 2 \
                                   + (self.position_cube3[0][1] - self.position_cube1[0][1]) ** 2)        
        self.cube3_distancez = sqrt((self.position_cube3[0][2] - self.position_cube1[0][2]) ** 2)



        dx = 0.005 * action[0]
        dy = 0.005 * action[1]
        dz = 0.005 * action[2]

        self.position_target = self.position + np.array([dx, dy, dz]).reshape(3, 1)

        self.move(self.position_target, self.postrue)

        for i in range(10):
            self.contact1 = p.getContactPoints(self.UR5, self.cube4)
            self.contact2 = p.getContactPoints(self.UR5, self.cube2)
            self.contact3 = p.getContactPoints(self.UR5, self.cube1)
            self.contact4 = p.getContactPoints(self.UR5, self.cube3)

            
        if self.stage == 0:
            if self.cube4_distance <= 0.0125 and self.G == 0:
                self.move_gripper(self.position, self.postrue, list([0.01, 0.01]))
                self.G = 1
                if not self.contact1:
                    self.G = 0
                    self.stage = 0


        elif self.stage == 1:
            if self.contact1:
                self.move_gripper(self.position, self.postrue, list([0.015, 0.015]))
                if self.cube4_distance1 <= 0.01:
                    if self.cube4_distancez <= 0.03:
                        self.move_gripper(self.position, self.postrue, list([0.03, 0.03]))
                        self.G = 2
                        p.changeDynamics(self.cube4, -1, mass=0) 

            else:
                if self.position_cube4[0][2] < 0.04:
                    self.move_gripper(self.position, self.postrue, list([0.03, 0.03]))
                    self.G = 0

        elif self.stage == 2:
            if self.cube2_distance <= 0.0125 and self.G == 2:
                self.move_gripper(self.position, self.postrue, list([0.01, 0.01]))
                self.G = 3
                if not self.contact2:
                    self.G = 2
                    self.stage = 2

        elif self.stage == 3:
            if self.contact2:
                self.move_gripper(self.position, self.postrue, list([0.015, 0.015]))
                if self.cube2_distance1 <= 0.01:
                    if self.cube2_distancez <= 0.06:
                        self.move_gripper(self.position, self.postrue, list([0.03, 0.03]))
                        self.G = 4
                        p.changeDynamics(self.cube2, -1, mass=0)  

        elif self.stage == 4:
            if self.cube1_distance <= 0.0125 and self.G == 4:
                self.move_gripper(self.position, self.postrue, list([0.01, 0.01]))
                self.G = 5
                if not self.contact3:
                    self.G = 4
                    self.stage = 4

        elif self.stage == 5:
            if self.contact3:
                self.move_gripper(self.position, self.postrue, list([0.015, 0.015]))
                if self.cube1_distance1 <= 0.01:
                    if self.cube1_distancez <= 0.06:
                        self.move_gripper(self.position, self.postrue, list([0.03, 0.03]))
                        self.G = 6
                        p.changeDynamics(self.cube1, -1, mass=0) 

        elif self.stage == 6:
            if self.cube3_distance <= 0.0125 and self.G == 6:
                self.move_gripper(self.position, self.postrue, list([0.01, 0.01]))
                self.G = 7
                if not self.contact4:
                    self.G = 6
                    self.stage = 6

        elif self.stage == 7:
            if self.contact4:
                self.move_gripper(self.position, self.postrue, list([0.015, 0.015]))
                if self.cube3_distance1 <= 0.01:
                    if self.cube3_distancez <= 0.06:
                        self.move_gripper(self.position, self.postrue, list([0.03, 0.03]))
                        self.G = 8
                        p.changeDynamics(self.cube3, -1, mass=0)  



        self.step_counter += 1


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
        jointposes = p.calculateInverseKinematics(self.UR5, 11, position, postrue, maxNumIterations=100)
        p.setJointMotorControlArray(self.UR5, list([0,1,2,3,4,5,6,9,10]), p.POSITION_CONTROL
                                    , list(jointposes))
        n = 100
        while (n):
            p.stepSimulation()
            n = n - 1

    def move_gripper(self, position, postrue, move):

        p.setJointMotorControlArray(self.UR5, list([9,10]), p.POSITION_CONTROL, move)
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

    def get_z(self):
        return self.z

    def _reward(self):
        self.position_cube1 = p.getBasePositionAndOrientation(self.cube1)
        self.position_cube2 = p.getBasePositionAndOrientation(self.cube2)
        self.position_cube3 = p.getBasePositionAndOrientation(self.cube3)
        self.position_cube4 = p.getBasePositionAndOrientation(self.cube4)
        



        terminated = bool(
            self.position[0][0] * 100 < self.x_low_obs
            or self.position[0][0] * 100 > self.x_high_obs
            or self.position[1][0] * 100 < self.y_low_obs
            or self.position[1][0] * 100 > self.y_high_obs
            or self.position[2][0] * 100 < self.z_low_obs
            or self.position[2][0] * 100 > self.z_high_obs
        )

        info = {}
        if terminated:
            self.reward = float(-10)
            self.done = True

        elif self.step_counter >= self._max_episode_steps:
            self.reward = float(-10)
            self.done = True

        elif self.G == 1 and self.stage == 1:
            self.reward = -self.cube4_distance1 - abs(self.cube4_distancez - 0.03)
            self.done2 = False
        elif self.G == 0 and self.stage == 1:
            self.stage = 0
            self.reward = -10
            self.done2 = True   

        elif self.stage == 0:
            if self.G == 1:
                self.stage = 1
                self.done1 = True
                self.done2 = False
                self.reward = 2500.0
            else:   
                self.reward = -self.cube4_distance
                self.done1 = False

        elif self.G == 2 and self.stage == 1:

            self.reward = 5000
            self.stage = 2
            self.done2 = True


        elif self.stage == 2:


            if self.G == 3:               
                self.done3 = True
                self.stage = 3
                self.reward  = 7500
            else:
                self.reward = -self.cube2_distance
                
                if self.contact1:
                    self.reward = -10
                    self.done = 0


        elif self.stage == 3:
            if self.G == 3:
                self.reward = -self.cube2_distance1 - abs(self.cube2_distancez - 0.05)
            elif self.G == 2:
                self.stage = 2
                self.reward = -10
                self.done4 = True 

            elif self.G == 4:
                self.stage = 4
                self.reward = 10000
                self.done4 = True



        elif self.stage == 4:


            if self.G == 5:               
                self.done5 = True
                self.stage = 5
                self.reward  = 12500
            else:
                self.reward = -self.cube1_distance

        elif self.stage == 5:

            if self.G == 5:
                self.reward = -self.cube1_distance1 - abs(self.cube1_distancez - 0.07)
            elif self.G == 4:
                self.stage = 4
                self.reward = -10
                self.done6 = True 
            
            elif self.G == 6:
                self.stage = 6
                self.reward = 15000
                self.done6 = True


        elif self.stage == 6:


            if self.G == 7:               
                self.done7 = True
                self.stage = 7
                self.reward  = 17500
            else:
                self.reward = -self.cube3_distance

        elif self.stage == 7:
            if self.G == 7:
                self.reward = -self.cube3_distance1 - abs(self.cube3_distancez -0.09)
            
            elif self.G == 8:
                self.stage = 8
                self.reward = 20000
                self.done = True
                for _ in range(20):
                  dx = 0.0
                  dy = 0.0
                  dz = 0.005
                  self.position = np.array(p.getLinkState(self.UR5, 11)[4], dtype=float).reshape(3, 1)
                  self.position_target = self.position + np.array([dx, dy, dz]).reshape(3, 1)
                  self.move(self.position_target, self.postrue)








        if self.stage == 0:
            self.observation = np.array(
                [self.position[0][0] * 100, self.position[1][0] * 100, self.position[2][0] * 100,
                 self.position_cube4[0][0] * 100, self.position_cube4[0][1] * 100,
                 self.position_cube4[0][2] * 100],
                dtype=float)
        elif self.stage == 1:
            self.observation = np.array(
                [self.position_cube4[0][0] * 100, self.position_cube4[0][1] * 100, self.position_cube4[0][2]  * 100,
                 0.6, 0.0,
                 0.0],
                dtype=float)
        elif self.stage == 2:
            self.observation = np.array(
                [self.position[0][0] * 100, self.position[1][0] * 100, self.position[2][0] * 100,
                 self.position_cube2[0][0] * 100, self.position_cube2[0][1] * 100,
                 self.position_cube2[0][2] * 100],
                dtype=float)
        elif self.stage == 3:
            self.observation = np.array(
                [self.position_cube2[0][0] * 100, self.position_cube2[0][1] * 100, self.position_cube2[0][2]  * 100,
                 self.position_cube4[0][0] * 100, self.position_cube4[0][1] * 100,
                 self.position_cube4[0][2] * 100],
                dtype=float)
        elif self.stage == 4:
            self.observation = np.array(
                [self.position[0][0] * 100, self.position[1][0] * 100, self.position[2][0] * 100,
                 self.position_cube1[0][0] * 100, self.position_cube1[0][1] * 100,
                 self.position_cube1[0][2] * 100],
                dtype=float)
        elif self.stage == 5:
            self.observation = np.array(
                [self.position_cube1[0][0] * 100, self.position_cube1[0][1] * 100, self.position_cube1[0][2]  * 100,
                 self.position_cube2[0][0] * 100, self.position_cube2[0][1] * 100,
                 self.position_cube2[0][2] * 100],
                dtype=float)
        elif self.stage == 6:
            self.observation = np.array(
                [self.position[0][0] * 100, self.position[1][0] * 100, self.position[2][0] * 100,
                 self.position_cube3[0][0] * 100, self.position_cube3[0][1] * 100,
                 self.position_cube3[0][2] * 100],
                dtype=float)
        elif self.stage == 7:
            self.observation = np.array(
                [self.position_cube3[0][0] * 100, self.position_cube3[0][1] * 100, self.position_cube3[0][2]  * 100,
                 self.position_cube1[0][0] * 100, self.position_cube1[0][1] * 100,
                 self.position_cube1[0][2] * 100],
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

        state = next_state


        return state, done


        
   