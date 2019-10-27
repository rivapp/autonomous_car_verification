'''Copyright (C) 2019 Radoslav Ivanov, Taylor J Carpenter, James
Weimer, Rajeev Alur, George J. Pappa, Insup Lee

This file is an F1/10 autonomomus car racing simulator.

Version 1 was written by James Weimer. The current version was written
by Radoslav Ivanov.

This simulator is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This simulator is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.  You should have received a
copy of the GNU General Public License along with Verisig.  If not,
see <https://www.gnu.org/licenses/>.

'''

import gym
from gym import spaces
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# car parameters
CAR_LENGTH = .45 # in m
CAR_CENTER_OF_MASS = .225 # from rear of car (m)
CAR_DECEL_CONST = .4
CAR_ACCEL_CONST = 1.633 # estimated from data
CAR_MOTOR_CONST = 0.2 # estimated from data
HYSTERESIS_CONSTANT = 4
MAX_TURNING_INPUT = 15 # in degrees

# lidar parameter
LIDAR_RANGE = 5 # in m

# safety parameter
SAFE_DISTANCE = 0.3 # in m

# default throttle if left unspecified
CONST_THROTTLE = 16

# training parameters
STEP_REWARD_GAIN = 10
INPUT_REWARD_GAIN = -0.05
CRASH_REWARD = -100

class World:

    def __init__(self, hallWidths, hallLengths, turns,\
                 car_dist_s, car_dist_f, car_heading,\
                 episode_length, time_step, lidar_field_of_view,\
                 lidar_num_rays, lidar_noise = 0, lidar_missing_rays = 0, lidar_missing_in_turn_only = False):

        # hallway parameters
        self.numHalls = len(hallWidths)
        self.hallWidths = hallWidths
        self.hallLengths = hallLengths
        self.turns = turns
        self.curHall = 0

        # car relative states
        self.car_dist_s = car_dist_s
        self.car_dist_f = car_dist_f
        self.car_V = 0
        self.car_heading = car_heading

        # car global states
        self.car_global_x = -self.hallWidths[0] / 2.0 + self.car_dist_s
        self.car_global_y = self.hallLengths[0] / 2.0 - car_dist_f
        self.car_global_heading = self.car_heading + np.pi / 2 #first hall goes "up" by default

        # step parameters
        self.time_step = time_step
        self.cur_step = 0
        self.episode_length = episode_length

        # storage
        self.allX = []
        self.allY = []
        self.allX.append(self.car_global_x)
        self.allY.append(self.car_global_y)

        # lidar setup
        self.lidar_field_of_view = lidar_field_of_view
        self.lidar_num_rays = lidar_num_rays

        self.lidar_noise = lidar_noise
        self.total_lidar_missing_rays = lidar_missing_rays

        self.lidar_missing_in_turn_only = lidar_missing_in_turn_only
        
        self.cur_num_missing_rays = lidar_missing_rays
        self.missing_indices = np.random.choice(self.lidar_num_rays, self.cur_num_missing_rays)

        # parameters needed for consistency with gym environments
        self.obs_low = np.zeros(self.lidar_num_rays, )
        self.obs_high = LIDAR_RANGE * np.ones(self.lidar_num_rays, )

        self.action_space = spaces.Box(low=-MAX_TURNING_INPUT, high=MAX_TURNING_INPUT, shape=(1,))
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high)

        self._max_episode_steps = episode_length

    def reset(self, pos = None):
        self.curHall = 0

        self.car_dist_s = self.hallWidths[0] / 2.0 + np.random.uniform(-0.2, 0.2)

        if not pos == None:
            self.car_dist_s = pos
        
        self.car_dist_f = self.hallLengths[0] / 2.0
        self.car_V = 0
        self.car_heading = 0 + np.random.uniform(-0.3, 0.3)
        
        self.car_global_x = -self.hallWidths[0] / 2.0 + self.car_dist_s
        self.car_global_y = 0
        self.car_global_heading = self.car_heading + np.pi / 2 #first hall goes "up" by default

        self.missing_indices = np.random.choice(self.lidar_num_rays, self.cur_num_missing_rays)
        
        self.cur_step = 0

        self.allX = []
        self.allY = []
        self.allX.append(self.car_global_x)
        self.allY.append(self.car_global_y)

        return self.scan_lidar()

    #NB: Mode switches are handled in the step function
    # x := [s, f, V, theta_local, x, y, theta_global]
    def bicycle_dynamics(self, x, t, u, delta, turn):

        if 'right' in turn:
            # -V * sin(theta_local + beta)
            dsdt = -x[2] * np.sin(x[3] + np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH))
        else:
            # V * sin(theta_local + beta)
            dsdt = x[2] * np.sin(x[3] + np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH))

        # -V * cos(theta_local + beta)
        dfdt = -x[2] * np.cos(x[3] + np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH)) 

        if u > HYSTERESIS_CONSTANT:
            # a * u - V
            dVdt = CAR_ACCEL_CONST * CAR_MOTOR_CONST * (u - HYSTERESIS_CONSTANT) - CAR_ACCEL_CONST * x[2]
        else:
            dVdt = - CAR_ACCEL_CONST * x[2]
            
        # V * cos(beta) * tan(delta) / l
        dtheta_ldt = x[2] * np.cos(np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH)) * np.tan(delta) / CAR_LENGTH 

        # V * cos(theta_global + beta)
        dxdt = x[2] * np.cos(x[6] + np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH)) 

        # V * sin(theta_global + beta)
        dydt = x[2] * np.sin(x[6] + np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH))

        # V * cos(beta) * tan(delta) / l
        dtheta_gdt = x[2] * np.cos(np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH)) * np.tan(delta) / CAR_LENGTH

        dXdt = [dsdt, dfdt, dVdt, dtheta_ldt, dxdt, dydt, dtheta_gdt]
        
        return dXdt

    #NB: Mode switches are handled in the step function
    # x := [s, f, V, theta_local, x, y, theta_global]
    def bicycle_dynamics_no_beta(self, x, t, u, delta, turn):

        if 'right' in turn:
            # -V * sin(theta_local)
            dsdt = -x[2] * np.sin(x[3])
        else:
            # V * sin(theta_local)
            dsdt = x[2] * np.sin(x[3])

        # -V * cos(theta_local)
        dfdt = -x[2] * np.cos(x[3]) 

        if u > HYSTERESIS_CONSTANT:
            # a * u - V
            dVdt = CAR_ACCEL_CONST * CAR_MOTOR_CONST * (u - HYSTERESIS_CONSTANT) - CAR_ACCEL_CONST * x[2]
        else:
            dVdt = - CAR_ACCEL_CONST * x[2]
            
        # V * tan(delta) / l
        dtheta_ldt = x[2] * np.tan(delta) / CAR_LENGTH 

        # V * cos(theta_global)
        dxdt = x[2] * np.cos(x[6]) 

        # V * sin(theta_global)
        dydt = x[2] * np.sin(x[6])

        # V * tan(delta) / l
        dtheta_gdt = x[2] * np.tan(delta) / CAR_LENGTH

        dXdt = [dsdt, dfdt, dVdt, dtheta_ldt, dxdt, dydt, dtheta_gdt]
        
        return dXdt    

    def step(self, delta, throttle = CONST_THROTTLE):
        self.cur_step += 1

        # Constrain turning input
        if delta > MAX_TURNING_INPUT:
            delta = MAX_TURNING_INPUT

        if delta < -MAX_TURNING_INPUT:
            delta = -MAX_TURNING_INPUT

        # simulate dynamics
        x0 = [self.car_dist_s, self.car_dist_f, self.car_V, self.car_heading, self.car_global_x, self.car_global_y, self.car_global_heading]
        t = [0, self.time_step]
        
        #new_x = odeint(self.bicycle_dynamics, x0, t, args=(throttle, delta * np.pi / 180, self.turns[self.curHall],))
        new_x = odeint(self.bicycle_dynamics_no_beta, x0, t, args=(throttle, delta * np.pi / 180, self.turns[self.curHall],))

        new_x = new_x[1]

        self.car_dist_s, self.car_dist_f, self.car_V, self.car_heading, self.car_global_x, self.car_global_y, self.car_global_heading =\
                    new_x[0], new_x[1], new_x[2], new_x[3], new_x[4], new_x[5], new_x[6]

        terminal = False

        # Compute reward
        reward = STEP_REWARD_GAIN

        # Region 1
        if self.car_dist_s > 0 and self.car_dist_s < self.hallWidths[self.curHall] and\
           self.car_dist_f > self.hallWidths[self.curHall]:

            reward += INPUT_REWARD_GAIN * delta * delta
            #pass

        # Region 2
        elif self.car_dist_s > 0 and self.car_dist_s < self.hallWidths[self.curHall] and\
           self.car_dist_f <= self.hallWidths[self.curHall]:

            #reward += INPUT_REWARD_GAIN * delta * delta
            pass

        # Region 3
        elif self.car_dist_s >  self.hallWidths[self.curHall] and\
             self.car_dist_f <= self.hallWidths[self.curHall]:

            pass

        # Set reward to maximum negative value if too close to a wall
        if self.car_dist_s < SAFE_DISTANCE or self.car_dist_f < SAFE_DISTANCE or\
           (self.car_dist_s > self.hallWidths[self.curHall] - SAFE_DISTANCE and self.car_dist_f > self.hallWidths[self.curHall] - SAFE_DISTANCE):
            terminal = True
            reward = CRASH_REWARD

        if self.cur_step == self.episode_length:
            terminal = True

        # Test if a mode switch in the world has changed
        if 'right' in self.turns[self.curHall]:
    
            if self.car_dist_s > LIDAR_RANGE:
                temp = self.car_dist_s
                self.car_heading = self.car_heading + np.pi / 2
                self.car_dist_s = self.car_dist_f # front wall is now the left wall
                self.curHall = self.curHall + 1 # next hallway

                #NB: this case deals with loops in the environment
                if self.curHall >= self.numHalls:
                    self.curHall = 0

                self.car_dist_f = self.hallLengths[self.curHall] - temp
    
        else: # left turn 
    
            if self.car_dist_s > self.hallWidths[self.curHall] + 2:
                temp = self.car_dist_s
                self.car_heading = self.car_heading - np.pi / 2
                self.car_dist_s = self.car_dist_f # front wall is now the left wall
                self.curHall = self.curHall + 1 # next hallway

                #NB: this case deals with loops in the environment
                if self.curHall >= self.numHalls:
                    self.curHall = 0                

                self.car_dist_f = self.hallLengths[self.curHall] - temp

        self.allX.append(self.car_global_x)
        self.allY.append(self.car_global_y)
        
        return self.scan_lidar(), reward, terminal, -1

    def scan_lidar(self):

        car_heading_deg = self.car_heading * 180 / np.pi

        alpha = int(np.floor(4 * car_heading_deg))

        theta_t = np.linspace(-self.lidar_field_of_view, self.lidar_field_of_view, self.lidar_num_rays)

        # lidar measurements
        data = np.zeros(len(theta_t))

        if 'right' in self.turns[self.curHall]:
            dist_l = self.car_dist_s
            dist_r = self.hallWidths[self.curHall] - self.car_dist_s
        else:
            dist_l = self.hallWidths[self.curHall] - self.car_dist_s
            dist_r = self.car_dist_s

        # Region 1 (before turn)
        if self.car_dist_s > 0 and self.car_dist_s < self.hallWidths[self.curHall] and\
           self.car_dist_f > self.hallWidths[(self.curHall + 1) % self.numHalls]:

            if 'right' in self.turns[self.curHall]:
            
                theta_l = np.arctan(self.car_dist_s / self.car_dist_f) * 180 / np.pi
                theta_r = -np.arctan((self.hallWidths[self.curHall] - self.car_dist_s) /\
                            (self.car_dist_f - self.hallWidths[(self.curHall + 1) % self.numHalls])) * 180 / np.pi

            else:
                theta_l = np.arctan((self.hallWidths[self.curHall] - self.car_dist_s) /\
                                    (self.car_dist_f - self.hallWidths[(self.curHall + 1) % self.numHalls])) * 180 / np.pi
                theta_r = -np.arctan(self.car_dist_s / self.car_dist_f) * 180 / np.pi

            index = 0

            for angle in theta_t:

                angle = angle + car_heading_deg
                if angle > 180:
                    angle = angle - 360
                elif angle < -180:
                    angle = angle + 360

                if angle <= theta_r:
                    data[index] = (dist_r) /\
                            (np.cos( (90 + angle) * np.pi / 180))

                elif angle > theta_r and angle <= theta_l:
                    data[index] = (self.car_dist_f) /\
                            (np.cos( (angle) * np.pi / 180))

                else:
                    data[index] = (dist_l) /\
                            (np.cos( (90 - angle) * np.pi / 180))

                #add noise
                data[index] += np.random.uniform(0, self.lidar_noise)

                if data[index] > LIDAR_RANGE or data[index] < 0:
                    data[index] = LIDAR_RANGE
                    
                index += 1

        # Region 2 (during turn)
        elif self.car_dist_s > 0 and self.car_dist_s < self.hallWidths[self.curHall] and\
           self.car_dist_f <= self.hallWidths[(self.curHall+1) % self.numHalls]:

            if 'right' in self.turns[self.curHall]:
                theta_l = np.arctan(self.car_dist_s / self.car_dist_f) * 180 / np.pi
                theta_r = -np.arctan((self.hallWidths[self.curHall] - self.car_dist_s) /\
                            (self.car_dist_f - self.hallWidths[(self.curHall + 1) % self.numHalls])) * 180 / np.pi - 180

            else:
                theta_l = 180 - np.arctan((self.hallWidths[self.curHall] - self.car_dist_s) /\
                            (self.hallWidths[(self.curHall + 1) % self.numHalls] - self.car_dist_f)) * 180 / np.pi
                theta_r = -np.arctan(self.car_dist_s / self.car_dist_f) * 180 / np.pi
                
            index = 0
            
            for angle in theta_t:

                angle = angle + car_heading_deg
                if angle > 180:
                    angle = angle - 360
                elif angle < -180:
                    angle = angle + 360

                if 'right' in self.turns[self.curHall]:
                    if angle <= theta_r:
                        data[index] = (dist_r) /\
                                (np.cos( (90 + angle) * np.pi / 180))

                    elif angle > theta_r and angle < -90:
                        data[index] = (self.hallWidths[(self.curHall + 1) % self.numHalls] - self.car_dist_f) /\
                                (np.cos( (180 + angle) * np.pi / 180))

                    elif angle > -90 and angle <= theta_l:
                        data[index] = (self.car_dist_f) /\
                                (np.cos( (angle) * np.pi / 180))
                    else:
                        data[index] = (dist_l) /\
                                (np.cos( (90 - angle) * np.pi / 180))

                else:
                    if angle <= theta_r:
                        data[index] = (dist_r) /\
                                (np.cos( (90 + angle) * np.pi / 180))

                    elif angle > theta_r and angle <= 90:
                        data[index] = (self.car_dist_f) /\
                                (np.cos( (angle) * np.pi / 180))

                    elif angle > 90 and angle <= theta_l:
                        data[index] = (self.hallWidth - self.car_dist_f) /\
                                (np.cos( (180 + angle) * np.pi / 180))
                    else:
                        data[index] = (dist_l) /\
                                (np.cos( (90 - angle) * np.pi / 180))

                #add noise
                data[index] += np.random.uniform(0, self.lidar_noise)

                if data[index] > LIDAR_RANGE or data[index] < 0:
                    data[index] = LIDAR_RANGE

                index += 1

        # Region 3 (after turn)
        elif self.car_dist_s > self.hallWidths[self.curHall] and\
             self.car_dist_f <= self.hallWidths[(self.curHall + 1) % self.numHalls]:

            if 'right' in self.turns[self.curHall]:            
            
                theta_l =  np.arctan(self.car_dist_s / self.car_dist_f) * 180 / np.pi
                theta_r =  180 - np.arctan(- (self.hallWidths[self.curHall] - self.car_dist_s) /\
                            (self.hallWidths[(self.curHall + 1) % self.numHalls] - self.car_dist_f)) * 180 / np.pi

            else:
                theta_l =  np.arctan(- (self.hallWidths[self.curHall] - self.car_dist_s) /\
                            (self.hallWidths[(self.curHall + 1) % self.numHalls] - self.car_dist_f)) * 180 / np.pi - 180
                theta_r =  -np.arctan(self.car_dist_s / self.car_dist_f) * 180 / np.pi

            index = 0

            for angle in theta_t:

                angle = angle + car_heading_deg
                if angle > 180:
                    angle = angle - 360
                elif angle < -180:
                    angle = angle + 360

                if 'right' in self.turns[self.curHall]:
                    if angle < -90:
                        data[index] = (self.hallWidths[(self.curHall + 1) % self.numHalls] - self.car_dist_f) /\
                                (np.cos( (180 + angle) * np.pi / 180))

                    elif angle == -90:
                          data[index] = LIDAR_RANGE

                    elif angle > -90 and angle <= theta_l:
                        data[index] = (self.car_dist_f) /\
                                (np.cos( (angle) * np.pi / 180))

                    elif angle > theta_l and angle <= theta_r:
                        data[index] = (dist_l) /\
                                (np.cos( (90 - angle) * np.pi / 180))

                    else:
                        data[index] = (self.hallWidths[(self.curHall + 1) % self.numHalls] - self.car_dist_f) /\
                                (np.cos( (180 - angle) * np.pi / 180))
                else:
                    if angle > 90:
                        data[index] = (self.hallWidths[(self.curHall + 1) % self.numHalls] - self.car_dist_f) /\
                                (np.cos( (180 + angle) * np.pi / 180))

                    elif angle < 90 and angle >= theta_r:
                        data[index] = (self.car_dist_f) /\
                                (np.cos( (angle) * np.pi / 180))

                    elif angle < theta_r and angle >= theta_l:
                        data[index] = (dist_r) /\
                                (np.cos( (90 - angle) * np.pi / 180))

                    else:
                        data[index] = (self.hallWidths[(self.curHall + 1) % self.numHalls] - self.car_dist_f) /\
                                (np.cos( (180 - angle) * np.pi / 180))

                # add noise
                data[index] += np.random.uniform(0, self.lidar_noise)


                if data[index] > LIDAR_RANGE or data[index] < 0:
                    data[index] = LIDAR_RANGE

                index += 1

        # add missing rays
        if self.lidar_missing_in_turn_only:
            
            # add missing rays only in Region 2 (plus an extra 1m before it)
            if self.car_dist_s > 0 and self.car_dist_s < self.hallWidth and\
               self.car_dist_f <= self.hallWidth + 1:

                for ray in self.missing_indices:
                    data[ray] = LIDAR_RANGE                
        else:
            # add missing rays in all regions
            for ray in self.missing_indices:
                data[ray] = LIDAR_RANGE
                
        return data

    def plot_trajectory(self):
        fig = plt.figure()

        self.plotHalls()

        plt.plot(self.allX, self.allY, 'r--')

        plt.show()

    def plot_lidar(self):
        fig = plt.figure()

        self.plotHalls()

        plt.scatter([self.car_global_x], [self.car_global_y], c = 'red')

        data = self.scan_lidar()

        lidX = []
        lidY = []

        theta_t = np.linspace(-self.lidar_field_of_view, self.lidar_field_of_view, self.lidar_num_rays)

        index = 0

        for curAngle in theta_t:    

            lidX.append(self.car_global_x + data[index] * np.cos(curAngle * np.pi / 180 + self.car_global_heading))
            lidY.append(self.car_global_y + data[index] * np.sin(curAngle * np.pi / 180 + self.car_global_heading))
                          
            index += 1

        plt.scatter(lidX, lidY, c = 'green')

        plt.show()

    def plotHalls(self):

        # 1st hall going up by default and centralized around origin
        midX = 0
        midY = 0
        going_up = True
        left = True
        
        for i in range(self.numHalls):

            # vertical hallway
            if i % 2 == 0:
                x1 = midX - self.hallWidths[i]/2.0
                x2 = midX - self.hallWidths[i]/2.0
                x3 = midX + self.hallWidths[i]/2.0
                x4 = midX + self.hallWidths[i]/2.0

                # L shape of bottom corner
                
                # Case 1: going down and about to turn left
                if 'left' in self.turns[i] and not going_up:
                    y1 = midY - self.hallLengths[i]/2.0 
                    y3 = midY - self.hallLengths[i]/2.0 + self.hallWidths[(i + 1) % self.numHalls]
                    
                # Case 2: going up and previous turn was right
                elif 'right' in self.turns[i-1] and going_up:
                    y1 = midY - self.hallLengths[i]/2.0 
                    y3 = midY - self.hallLengths[i]/2.0 + self.hallWidths[i-1]

                # _| shape of bottom corner
                # Case 1: going down and about to turn right
                elif 'right' in self.turns[i] and not going_up:
                    y1 = midY - self.hallLengths[i]/2.0 + self.hallWidths[(i + 1) % self.numHalls]
                    y3 = midY - self.hallLengths[i]/2.0

                # Case 2: going up and previous turn was left
                else:
                    y1 = midY - self.hallLengths[i]/2.0 + self.hallWidths[i-1]
                    y3 = midY - self.hallLengths[i]/2.0

                # Gamma shape of top corner
                # Case 1: going up and about to turn right
                if 'right' in self.turns[i] and going_up:
                    y2 = midY + self.hallLengths[i]/2.0 
                    y4 = midY + self.hallLengths[i]/2.0 - self.hallWidths[(i + 1) % self.numHalls]
                # Case 2: going down and previous turn was left 
                elif 'left' in self.turns[i-1] and not going_up:
                    y2 = midY + self.hallLengths[i]/2.0 
                    y4 = midY + self.hallLengths[i]/2.0 - self.hallWidths[i-1]

                # Reverse Gamma shape of top corner
                # Case 1: going up and about to turn left
                elif 'left' in self.turns[i] and going_up:
                    y2 = midY + self.hallLengths[i]/2.0 - self.hallWidths[(i + 1) % self.numHalls]
                    y4 = midY + self.hallLengths[i]/2.0
                # Case 2: going down and previous turn was right
                else:
                    y2 = midY + self.hallLengths[i]/2.0 - self.hallWidths[(i-1)]
                    y4 = midY + self.hallLengths[i]/2.0

                # update coordinates and directions
                if going_up:
                    if 'left' in self.turns[i]:
                        midX = midX - self.hallLengths[(i + 1) % self.numHalls]/2.0 + self.hallWidths[i]/2.0
                        left = True

                    else:
                        midX = midX + self.hallLengths[(i + 1) % self.numHalls]/2.0 - self.hallWidths[i]/2.0
                        left = False
                        
                    midY = midY + self.hallLengths[i]/2.0 - self.hallWidths[(i + 1) % self.numHalls]/2.0
                    

                else:
                    if 'left' in self.turns[i]:
                        midX = midX + self.hallLengths[(i + 1) % self.numHalls]/2.0 - self.hallWidths[i]/2.0
                        left = False

                    else:
                        midX = midX - self.hallLengths[(i + 1) % self.numHalls]/2.0 + self.hallWidths[i]/2.0
                        left = True
                        
                    midY = midY - self.hallLengths[i]/2.0 + self.hallWidths[(i + 1) % self.numHalls]/2.0
                    
            # horizontal hallway    
            else:

                # Gamma shape of left corner
                # Case 1: going right and previous turn was right
                if 'right' in self.turns[i-1] and not left:
                    x1 = midX - self.hallLengths[i]/2.0 
                    x3 = midX - self.hallLengths[i]/2.0 + self.hallWidths[i-1]
                # Case 2: going left and about to turn left
                elif left and 'left' in self.turns[i]:
                    x1 = midX - self.hallLengths[i]/2.0 
                    x3 = midX - self.hallLengths[i]/2.0 + self.hallWidths[(i + 1) % self.numHalls]
                    
                # L shape of left corner
                # Case 1: going right and previous turn was left
                elif 'left' in self.turns[i-1] and not left:
                    x1 = midX - self.hallLengths[i]/2.0 + self.hallWidths[i-1]
                    x3 = midX - self.hallLengths[i]/2.0
                # Case 2: going left and about to turn right
                else:
                    x1 = midX - self.hallLengths[i]/2.0 + self.hallWidths[(i + 1) % self.numHalls]
                    x3 = midX - self.hallLengths[i]/2.0

                    
                # Reverse Gamma shape of right corner
                # Case 1: going right and about to turn right
                if 'right' in self.turns[i] and not left:
                    x2 = midX + self.hallLengths[i]/2.0 
                    x4 = midX + self.hallLengths[i]/2.0 - self.hallWidths[(i + 1) % self.numHalls]
                # Case 2: going left and previous turn was left
                elif 'left' in self.turns[i-1] and left:
                    x2 = midX + self.hallLengths[i]/2.0 
                    x4 = midX + self.hallLengths[i]/2.0 - self.hallWidths[i-1]

                # _| shape of right corner
                # Case 1: going right and about to turn left
                elif 'left' in self.turns[i] and not left:
                    x2 = midX + self.hallLengths[i]/2.0 - self.hallWidths[(i + 1) % self.numHalls]
                    x4 = midX + self.hallLengths[i]/2.0
                # Case 2: going left and previous turn was right
                else:
                    x2 = midX + self.hallLengths[i]/2.0 - self.hallWidths[i-1]
                    x4 = midX + self.hallLengths[i]/2.0

                
                y1 = midY + self.hallWidths[i]/2.0
                y2 = midY + self.hallWidths[i]/2.0
                y3 = midY - self.hallWidths[i]/2.0
                y4 = midY - self.hallWidths[i]/2.0

                # update coordinates and directions
                if left:
                    if 'left' in self.turns[i]:
                        midY = midY - self.hallLengths[(i + 1) % self.numHalls]/2.0 + self.hallWidths[i]/2.0
                        going_up = False
                    else:
                        midY = midY + self.hallLengths[(i + 1) % self.numHalls]/2.0 - self.hallWidths[i]/2.0
                        going_up = True
                        
                    midX = midX - self.hallLengths[i]/2.0 + self.hallWidths[(i + 1) % self.numHalls]/2.0

                else:
                    if 'left' in self.turns[i]:
                        midY = midY + self.hallLengths[(i + 1) % self.numHalls]/2.0 - self.hallWidths[i]/2.0
                        going_up = True

                    else:
                        midY = midY - self.hallLengths[(i + 1) % self.numHalls]/2.0 + self.hallWidths[i]/2.0
                        going_up = False
                    midX = midX + self.hallLengths[i]/2.0 - self.hallWidths[(i + 1) % self.numHalls]/2.0

                    

            l1x = np.array([x1, x2])
            l1y = np.array([y1, y2])
            l2x = np.array([x3, x4])
            l2y = np.array([y3, y4])
            plt.plot(l1x, l1y, 'b', linewidth=3)
            plt.plot(l2x, l2y, 'b', linewidth=3)
