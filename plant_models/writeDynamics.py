import numpy as np
from six.moves import cPickle as pickle

#k is the current step
#u is the NN's input
#y1 is distance to side wall
#y2 is distance to front wall
#y3 is linear velocity
#y4 is car heading in radians (relative to hallway)

#angle is the rotating lidar angle
#theta_l and theta_r are angles w.r.t. left and right wall, respectively
#temp1 and temp2 are used in various computations (denoted in each one)
#f_i are lidar rays (assuming lidar rays are -LIDAR_RANGE:LIDAR_OFFSET:LIDAR_RANGE)

MAX_TURNING_INPUT = 15 # in degrees
CONST_THROTTLE = 16 # constant throttle input for this case study

CAR_LENGTH = .45 # in m
CAR_LENGTH_INV = 1 / CAR_LENGTH # in m
CAR_CENTER_OF_MASS = 0.225 # from rear of car (m)
CAR_ACCEL_CONST = 1.633
CAR_MOTOR_CONST = 0.2 # 45 MPH top speed (20 m/s) at 100 throttle

LIDAR_MAX_DISTANCE = 5 # in m
LIDAR_RANGE = 120 * np.pi / 180 # in radians
LIDAR_OFFSET = 8 * np.pi / 180 # in radians
NUM_RAYS = int(round((2 *  LIDAR_RANGE) / LIDAR_OFFSET))  + 1

print(NUM_RAYS)
print("FUSCKSCASD")

HALLWAY_WIDTH = 1.5
HALLWAY_LENGTH = 20
MODE_SWITCH_OFFSET = 2

TIME_STEP = 0.1 # in s

PIBY2 = np.pi / 2
PIBY180 = np.pi / 180.0
ONE80BYPI = 180.0 / np.pi

HYSTERESIS_CONSTANT = 4

plant = {}

plant[1] = {}
plant[1]['name'] = 'cont_'
plant[1]['states'] = ['angle', 'theta_l', 'theta_r', 'temp1', 'temp2', 'y1', 'y2', 'y3', 'y4', 'u', 'k']
plant[1]['odetype'] = 'nonpoly ode'
plant[1]['dynamics'] = {}
plant[1]['dynamics']['y1'] = 'y1\' = -y3 * sin(y4)\n'
plant[1]['dynamics']['y2'] = 'y2\' = -y3 * cos(y4)\n'
plant[1]['dynamics']['y3'] = 'y3\' = ' + str(CAR_ACCEL_CONST) +\
                  ' * ' + str(CAR_MOTOR_CONST) + ' * (' + str(CONST_THROTTLE) +\
                  ' - ' + str(HYSTERESIS_CONSTANT) + ') - ' + str(CAR_ACCEL_CONST) + ' * y3\n'
plant[1]['dynamics']['y4'] = 'y4\' = ' + str(CAR_LENGTH_INV) + ' * y3 * sin(u) / cos(u)\n'
plant[1]['dynamics']['k'] = 'k\' = 0\n'
plant[1]['dynamics']['u'] = 'u\' = 0\n'
plant[1]['dynamics']['angle'] = 'angle\' = 0\n'
plant[1]['dynamics']['clock'] = 'clock\' = 0\n'
plant[1]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[1]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[1]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[1]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[1]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'

plant[1]['invariants'] = ['clock <= ' + str(TIME_STEP)]
plant[1]['transitions'] = {}
plant[1]['transitions'][(1,2)] = {}
plant[1]['transitions'][(1,2)]['guards1'] =\
                        ['clock = ' + str(TIME_STEP), 'y1 >= ' + str(HALLWAY_WIDTH + MODE_SWITCH_OFFSET)]
plant[1]['transitions'][(1,2)]['reset1'] =\
                        ['clock\' := 0', 'k\' := k + 1', 'y4\' := y4 + ' + str(PIBY2), 'y1\' := y2',\
                         'y2\' := ' + str(HALLWAY_LENGTH) + ' - y1']
plant[1]['transitions'][(1,2)]['guards2'] =\
                        ['clock = ' + str(TIME_STEP), 'y1 <= ' + str(HALLWAY_WIDTH + MODE_SWITCH_OFFSET)]
plant[1]['transitions'][(1,2)]['reset2'] = ['clock\' := 0', 'k\' := k + 1']

# end of plant dynanmics

# beginning of lidar model

# this mode is used to prepare the temp variables for the div mode
plant[2] = {}
plant[2]['name'] = ''
plant[2]['odetype'] = 'lti ode'
plant[2]['dynamics'] = {}
plant[2]['dynamics']['y1'] = 'y1\' = 0\n'
plant[2]['dynamics']['y2'] = 'y2\' = 0\n'
plant[2]['dynamics']['y3'] = 'y3\' = 0\n'
plant[2]['dynamics']['y4'] = 'y4\' = 0\n'
plant[2]['dynamics']['k'] = 'k\' = 0\n'
plant[2]['dynamics']['u'] = 'u\' = 0\n'
plant[2]['dynamics']['angle'] = 'angle\' = 0\n'
plant[2]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[2]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[2]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[2]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[2]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[2]['invariants'] = ['clock <= 0']
plant[2]['transitions'] = {}
plant[2]['transitions'][(2,3)] = {}
plant[2]['transitions'][(2,3)]['guards1'] = ['clock = 0']
plant[2]['transitions'][(2,3)]['reset1'] = ['clock\' := 0', 'temp1\' := 10 * y2',\
                        'temp2\' := 10 * (y2 - ' + str(HALLWAY_WIDTH) + ')', 'theta_l\' := 0',\
                        'theta_r\' := 0']

# need an empty mode to first perform the temp1 and temp2 resets
plant[3] = {}
plant[3]['name'] = ''
plant[3]['odetype'] = 'lti ode'
plant[3]['dynamics'] = {}
plant[3]['dynamics']['y1'] = 'y1\' = 0\n'
plant[3]['dynamics']['y2'] = 'y2\' = 0\n'
plant[3]['dynamics']['y3'] = 'y3\' = 0\n'
plant[3]['dynamics']['y4'] = 'y4\' = 0\n'
plant[3]['dynamics']['k'] = 'k\' = 0\n'
plant[3]['dynamics']['u'] = 'u\' = 0\n'
plant[3]['dynamics']['angle'] = 'angle\' = 0\n'
plant[3]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[3]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[3]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[3]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[3]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[3]['invariants'] = ['clock <= 0']
plant[3]['transitions'] = {}
plant[3]['transitions'][(3, 4)] = {}
plant[3]['transitions'][(3, 4)]['guards1'] = ['clock = 0']
plant[3]['transitions'][(3, 4)]['reset1'] = ['clock\' := 0']

#modes reg1, reg2, reg3 correspond to the car being in Region 1, 2, 3, respectively
reg1 = 6
reg2 = 7
reg3 = 8

#longest path is currently through reg2 modes (currently 10 jumps from plant to Lidar modes)

plant[4] = {}
plant[4]['name'] = 'div_1_3_'
plant[4]['odetype'] = 'lti ode'
plant[4]['dynamics'] = {}
plant[4]['dynamics']['y1'] = 'y1\' = 0\n'
plant[4]['dynamics']['y2'] = 'y2\' = 0\n'
plant[4]['dynamics']['y3'] = 'y3\' = 0\n'
plant[4]['dynamics']['y4'] = 'y4\' = 0\n'
plant[4]['dynamics']['k'] = 'k\' = 0\n'
plant[4]['dynamics']['u'] = 'u\' = 0\n'
plant[4]['dynamics']['angle'] = 'angle\' = 0\n'
plant[4]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[4]['dynamics']['temp2'] = 'temp2\' = 0\n'
plant[4]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[4]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
for i in range(NUM_RAYS):
    plant[4]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[4]['invariants'] = ['clock <= 0']
plant[4]['transitions'] = {}
plant[4]['transitions'][(4, 5)] = {}
plant[4]['transitions'][(4, 5)]['guards1'] = ['clock = 0']
plant[4]['transitions'][(4, 5)]['reset1'] = ['clock\' := 0', 'theta_l\' := theta_l * y1 * 10']

plant[5] = {}
plant[5]['name'] = 'div_2_4_'
plant[5]['odetype'] = 'lti ode'
plant[5]['dynamics'] = {}
plant[5]['dynamics']['y1'] = 'y1\' = 0\n'
plant[5]['dynamics']['y2'] = 'y2\' = 0\n'
plant[5]['dynamics']['y3'] = 'y3\' = 0\n'
plant[5]['dynamics']['y4'] = 'y4\' = 0\n'
plant[5]['dynamics']['k'] = 'k\' = 0\n'
plant[5]['dynamics']['u'] = 'u\' = 0\n'
plant[5]['dynamics']['angle'] = 'angle\' = 0\n'
plant[5]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[5]['dynamics']['temp2'] = 'temp2\' = 0\n'
plant[5]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[5]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
for i in range(NUM_RAYS):
    plant[5]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[5]['invariants'] = ['clock <= 0']
plant[5]['transitions'] = {}
plant[5]['transitions'][(5,reg1)] = {}
plant[5]['transitions'][(5,reg1)]['guards1'] =\
                        ['clock = 0', 'y1 <= ' + str(HALLWAY_WIDTH), 'y2 >= ' + str(HALLWAY_WIDTH)]
plant[5]['transitions'][(5,reg1)]['reset1'] = ['clock\' := 0',\
                        'theta_r\' := 10 * theta_r * (' + str(HALLWAY_WIDTH) + ' - y1)']
plant[5]['transitions'][(5,reg2)] = {}
plant[5]['transitions'][(5,reg2)]['guards1'] =\
                        ['clock = 0', 'y1 <= ' + str(HALLWAY_WIDTH), 'y2 <= ' + str(HALLWAY_WIDTH)]
plant[5]['transitions'][(5,reg2)]['reset1'] = ['clock\' := 0',\
                        'theta_r\' := 10 * theta_r * (' + str(HALLWAY_WIDTH) + ' - y1)']
plant[5]['transitions'][(5,reg3)] = {}
plant[5]['transitions'][(5,reg3)]['guards1'] =\
                        ['clock = 0', 'y1 >= ' + str(HALLWAY_WIDTH), 'y2 <= ' + str(HALLWAY_WIDTH)]
plant[5]['transitions'][(5,reg3)]['reset1'] = ['clock\' := 0',\
                        'theta_r\' := 10 * theta_r * (' + str(HALLWAY_WIDTH) + ' - y1)']

mode1_reg1 = reg3 + 16 #24 currently
mode1_reg2 = mode1_reg1 + NUM_RAYS * 8 # 8 modes per ray in region 1 currently
mode1_reg3 = mode1_reg2 + NUM_RAYS * 10 # 10 modes per ray in region 2 currenlty

#temp1 is angle - theta_l (after all the reg1 modes)
#temp2 is angle - theta_r (after all the reg1 modes)

plant[reg1] = {}
plant[reg1]['name'] = 'reg1'
plant[reg1]['odetype'] = 'lti ode'
plant[reg1]['dynamics'] = {}
plant[reg1]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg1]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg1]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg1]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg1]['dynamics']['k'] = 'k\' = 0\n'
plant[reg1]['dynamics']['u'] = 'u\' = 0\n'
plant[reg1]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg1]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg1]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg1]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg1]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg1]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg1]['invariants'] = ['clock <= 0']
plant[reg1]['transitions'] = {}
plant[reg1]['transitions'][(reg1, reg1 + 3)] = {}
plant[reg1]['transitions'][(reg1, reg1 + 3)]['guards1'] = ['clock = 0']
plant[reg1]['transitions'][(reg1, reg1 + 3)]['reset1'] = ['clock\' := 0']

# temp1 = arctan(theta_l) // theta_l \in [-1,1]
plant[reg1 + 3] = {}
plant[reg1 + 3]['name'] = 'arc_3_1_'
plant[reg1 + 3]['odetype'] = 'lti ode'
plant[reg1 + 3]['dynamics'] = {}
plant[reg1 + 3]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg1 + 3]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg1 + 3]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg1 + 3]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg1 + 3]['dynamics']['k'] = 'k\' = 0\n'
plant[reg1 + 3]['dynamics']['u'] = 'u\' = 0\n'
plant[reg1 + 3]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg1 + 3]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg1 + 3]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg1 + 3]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg1 + 3]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg1 + 3]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg1 + 3]['invariants'] = ['clock <= 0']
plant[reg1 + 3]['transitions'] = {}
plant[reg1 + 3]['transitions'][(reg1 + 3, reg1 + 4)] = {}
plant[reg1 + 3]['transitions'][(reg1 + 3, reg1 + 4)]['guards1'] =\
                                    ['clock = 0', 'theta_r >= -1', 'theta_r <= 1']
plant[reg1 + 3]['transitions'][(reg1 + 3, reg1 + 4)]['reset1'] = ['clock\' := 0', 'temp2\' := 0']
plant[reg1 + 3]['transitions'][(reg1 + 3, reg1 + 5)] = {}
plant[reg1 + 3]['transitions'][(reg1 + 3, reg1 + 5)]['guards1'] = ['clock = 0', 'theta_r >= 1']
plant[reg1 + 3]['transitions'][(reg1 + 3, reg1 + 5)]['reset1'] = ['clock\' := 0', 'temp2\' := 0']
plant[reg1 + 3]['transitions'][(reg1 + 3, reg1 + 5)]['guards2'] = ['clock = 0', 'theta_r <= -1']
plant[reg1 + 3]['transitions'][(reg1 + 3, reg1 + 5)]['reset2'] = ['clock\' := 0', 'temp2\' := 0']

# temp2 = arctan(theta_r) // theta_r \in [-1,1]
plant[reg1 + 4] = {}
plant[reg1 + 4]['name'] = 'arc_4_2_'
plant[reg1 + 4]['odetype'] = 'lti ode'
plant[reg1 + 4]['dynamics'] = {}
plant[reg1 + 4]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg1 + 4]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg1 + 4]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg1 + 4]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg1 + 4]['dynamics']['k'] = 'k\' = 0\n'
plant[reg1 + 4]['dynamics']['u'] = 'u\' = 0\n'
plant[reg1 + 4]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg1 + 4]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg1 + 4]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg1 + 4]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg1 + 4]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg1 + 4]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg1 + 4]['invariants'] = ['clock <= 0']
plant[reg1 + 4]['transitions'] = {}
plant[reg1 + 4]['transitions'][(reg1 + 4, mode1_reg1)] = {}
plant[reg1 + 4]['transitions'][(reg1 + 4, mode1_reg1)]['guards1'] = ['clock = 0']
plant[reg1 + 4]['transitions'][(reg1 + 4, mode1_reg1)]['reset1'] = ['clock\' := 0',\
                            'theta_l\' := temp1',\
                            'theta_r\' := - temp2',\
                            'angle\' := y4 + ' + str(-LIDAR_RANGE), \
                            'temp1\' := y4 + ' + str(-LIDAR_RANGE) + ' - temp1', \
                            'temp2\' := y4 + ' + str(-LIDAR_RANGE) + ' + temp2']

# theta_r =  (1 / theta_r) // theta_r \in [-infty, -1] \cap [1, \infty]
# This mode should not be reachable since theta_r \in [0, 1] in Region 1
plant[reg1 + 5] = {}
plant[reg1 + 5]['name'] = 'div_2_2_'
plant[reg1 + 5]['odetype'] = 'lti ode'
plant[reg1 + 5]['dynamics'] = {}
plant[reg1 + 5]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg1 + 5]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg1 + 5]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg1 + 5]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg1 + 5]['dynamics']['k'] = 'k\' = 0\n'
plant[reg1 + 5]['dynamics']['u'] = 'u\' = 0\n'
plant[reg1 + 5]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg1 + 5]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg1 + 5]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg1 + 5]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg1 + 5]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg1 + 5]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg1 + 5]['invariants'] = ['clock <= 0']
plant[reg1 + 5]['transitions'] = {}
plant[reg1 + 5]['transitions'][(reg1 + 5, reg1 + 6)] = {}
plant[reg1 + 5]['transitions'][(reg1 + 5, reg1 + 6)]['guards1'] = ['clock = 0']
plant[reg1 + 5]['transitions'][(reg1 + 5, reg1 + 6)]['reset1'] = ['clock\' := 0']

# temp2 = arctan(theta_r) // theta_r \in [-infty, -1] \cap [1, \infty]
# Ditto as for the previous mode
plant[reg1 + 6] = {}
plant[reg1 + 6]['name'] = 'arc_4_2_'
plant[reg1 + 6]['odetype'] = 'lti ode'
plant[reg1 + 6]['dynamics'] = {}
plant[reg1 + 6]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg1 + 6]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg1 + 6]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg1 + 6]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg1 + 6]['dynamics']['k'] = 'k\' = 0\n'
plant[reg1 + 6]['dynamics']['u'] = 'u\' = 0\n'
plant[reg1 + 6]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg1 + 6]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg1 + 6]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg1 + 6]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg1 + 6]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg1 + 6]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg1 + 6]['invariants'] = ['clock <= 0']
plant[reg1 + 6]['transitions'] = {}
plant[reg1 + 6]['transitions'][(reg1 + 6,mode1_reg1)] = {}
plant[reg1 + 6]['transitions'][(reg1 + 6,mode1_reg1)]['guards1'] = ['clock = 0', 'theta_r >= 0']
plant[reg1 + 6]['transitions'][(reg1 + 6,mode1_reg1)]['reset1'] = ['clock\' := 0',\
                                'theta_l\' := temp1',\
                                'theta_r\' := - (' + str(np.pi/2.0) + ' - temp2)',\
                                'angle\' := y4 + ' + str(-LIDAR_RANGE), \
                                'temp1\' := y4 + ' + str(-LIDAR_RANGE) + ' - temp1',\
                                'temp2\' := y4 + ' + str(-LIDAR_RANGE) + ' + (' + str(np.pi/2.0) + ' - temp2)']
plant[reg1 + 6]['transitions'][(reg1 + 6,mode1_reg1)]['guards2'] = ['clock = 0', 'theta_r <= 0']
plant[reg1 + 6]['transitions'][(reg1 + 6,mode1_reg1)]['reset2'] = ['clock\' := 0',\
                                'theta_l\' := temp1',\
                                'theta_r\' := - (-' + str(np.pi/2.0) + ' - temp2)',\
                                'angle\' := y4 + ' + str(-LIDAR_RANGE),\
                                'temp1\' := y4 + ' + str(-LIDAR_RANGE) + ' - temp1',\
                                'temp2\' := y4 + ' + str(-LIDAR_RANGE) + ' + (-' + str(np.pi/2.0) + ' - temp2)']

#temp1 is angle - theta_l (after all the reg2 modes)
#temp2 is angle - theta_r (after all the reg2 modes)


plant[reg2] = {}
plant[reg2]['name'] = 'reg2'
plant[reg2]['odetype'] = 'lti ode'
plant[reg2]['dynamics'] = {}
plant[reg2]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg2]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg2]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg2]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg2]['dynamics']['k'] = 'k\' = 0\n'
plant[reg2]['dynamics']['u'] = 'u\' = 0\n'
plant[reg2]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg2]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg2]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg2]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg2]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg2]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg2]['invariants'] = ['clock <= 0']
plant[reg2]['transitions'] = {}
plant[reg2]['transitions'][(reg2, reg2 + 6)] = {}
plant[reg2]['transitions'][(reg2, reg2 + 6)]['guards1'] = ['clock = 0', 'theta_l >= -1', 'theta_l <= 1']
plant[reg2]['transitions'][(reg2, reg2 + 6)]['reset1'] = ['clock\' := 0', 'temp1\' := 0', 'temp2\' := 0']
plant[reg2]['transitions'][(reg2, reg2 + 7)] = {}
plant[reg2]['transitions'][(reg2, reg2 + 7)]['guards1'] = ['clock = 0', 'theta_l >= 1']
plant[reg2]['transitions'][(reg2, reg2 + 7)]['reset1'] = ['clock\' := 0', 'temp1\' := 0', 'temp2\' := 0']
plant[reg2]['transitions'][(reg2, reg2 + 7)]['guards2'] = ['clock = 0', 'theta_l <= -1']
plant[reg2]['transitions'][(reg2, reg2 + 7)]['reset2'] = ['clock\' := 0', 'temp1\' := 0', 'temp2\' := 0']

# temp1 = arctan(theta_l) // theta_l \in [-1, 1]
plant[reg2 + 6] = {}
plant[reg2 + 6]['name'] = 'arc_3_1_'
plant[reg2 + 6]['odetype'] = 'lti ode'
plant[reg2 + 6]['dynamics'] = {}
plant[reg2 + 6]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg2 + 6]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg2 + 6]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg2 + 6]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg2 + 6]['dynamics']['k'] = 'k\' = 0\n'
plant[reg2 + 6]['dynamics']['u'] = 'u\' = 0\n'
plant[reg2 + 6]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg2 + 6]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg2 + 6]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg2 + 6]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg2 + 6]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg2 + 6]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg2 + 6]['invariants'] = ['clock <= 0']
plant[reg2 + 6]['transitions'] = {}
plant[reg2 + 6]['transitions'][(reg2 + 6, reg2 + 9)] = {}
plant[reg2 + 6]['transitions'][(reg2 + 6, reg2 + 9)]['guards1'] = ['clock = 0', 'theta_r <= 1', 'theta_r >= -1'] 
plant[reg2 + 6]['transitions'][(reg2 + 6, reg2 + 9)]['reset1'] = ['clock\' := 0', 'theta_l\' := temp1']
plant[reg2 + 6]['transitions'][(reg2 + 6, reg2 + 10)] = {}
plant[reg2 + 6]['transitions'][(reg2 + 6, reg2 + 10)]['guards1'] = ['clock = 0', 'theta_r <= -1'] 
plant[reg2 + 6]['transitions'][(reg2 + 6, reg2 + 10)]['reset1'] = ['clock\' := 0', 'theta_l\' := temp1']
plant[reg2 + 6]['transitions'][(reg2 + 6, reg2 + 10)]['guards2'] = ['clock = 0', 'theta_r >= 1'] 
plant[reg2 + 6]['transitions'][(reg2 + 6, reg2 + 10)]['reset2'] = ['clock\' := 0', 'theta_l\' := temp1']

# theta_l = (1 / theta_l)  // theta_l \in [-\infty, -1] or [1, \infty]
plant[reg2 + 7] = {}
plant[reg2 + 7]['name'] = 'div_1_1_'
plant[reg2 + 7]['odetype'] = 'lti ode'
plant[reg2 + 7]['dynamics'] = {}
plant[reg2 + 7]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg2 + 7]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg2 + 7]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg2 + 7]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg2 + 7]['dynamics']['k'] = 'k\' = 0\n'
plant[reg2 + 7]['dynamics']['u'] = 'u\' = 0\n'
plant[reg2 + 7]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg2 + 7]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg2 + 7]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg2 + 7]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg2 + 7]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg2 + 7]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg2 + 7]['invariants'] = ['clock <= 0']
plant[reg2 + 7]['transitions'] = {}
plant[reg2 + 7]['transitions'][(reg2 + 7, reg2 + 8)] = {}
plant[reg2 + 7]['transitions'][(reg2 + 7, reg2 + 8)]['guards1'] = ['clock = 0'] 
plant[reg2 + 7]['transitions'][(reg2 + 7, reg2 + 8)]['reset1'] = ['clock\' := 0']

# temp1 = arctan(theta_l) // theta_l \in [-\infty, -1] or [1, \infty]
plant[reg2 + 8] = {}
plant[reg2 + 8]['name'] = 'arc_3_1_'
plant[reg2 + 8]['odetype'] = 'lti ode'
plant[reg2 + 8]['dynamics'] = {}
plant[reg2 + 8]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg2 + 8]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg2 + 8]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg2 + 8]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg2 + 8]['dynamics']['k'] = 'k\' = 0\n'
plant[reg2 + 8]['dynamics']['u'] = 'u\' = 0\n'
plant[reg2 + 8]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg2 + 8]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg2 + 8]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg2 + 8]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg2 + 8]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg2 + 8]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg2 + 8]['invariants'] = ['clock <= 0']
plant[reg2 + 8]['transitions'] = {}
plant[reg2 + 8]['transitions'][(reg2 + 8, reg2 + 9)] = {}
plant[reg2 + 8]['transitions'][(reg2 + 8, reg2 + 9)]['guards1'] =\
                                    ['clock = 0', 'theta_r <= 1', 'theta_r >= -1', 'theta_l >= 0'] 
plant[reg2 + 8]['transitions'][(reg2 + 8, reg2 + 9)]['reset1'] =\
                                    ['clock\' := 0', 'theta_l\' := ' + str(np.pi / 2) + ' - temp1']
plant[reg2 + 8]['transitions'][(reg2 + 8, reg2 + 9)]['guards2'] =\
                                    ['clock = 0', 'theta_r <= 1', 'theta_r >= -1', 'theta_l <= 0'] 
plant[reg2 + 8]['transitions'][(reg2 + 8, reg2 + 9)]['reset2'] =\
                                    ['clock\' := 0', 'theta_l\' := -' + str(np.pi / 2) + ' - temp1']

plant[reg2 + 8]['transitions'][(reg2 + 8, reg2 + 10)] = {}
plant[reg2 + 8]['transitions'][(reg2 + 8, reg2 + 10)]['guards1'] =\
                                    ['clock = 0', 'theta_r <= -1', 'theta_l >= 0'] 
plant[reg2 + 8]['transitions'][(reg2 + 8, reg2 + 10)]['reset1'] =\
                                    ['clock\' := 0', 'theta_l\' := ' + str(np.pi / 2) + ' - temp1']
plant[reg2 + 8]['transitions'][(reg2 + 8, reg2 + 10)]['guards2'] =\
                                    ['clock = 0', 'theta_r <= -1', 'theta_l <= 0'] 
plant[reg2 + 8]['transitions'][(reg2 + 8, reg2 + 10)]['reset2'] =\
                                    ['clock\' := 0', 'theta_l\' := -' + str(np.pi / 2) + ' - temp1']
plant[reg2 + 8]['transitions'][(reg2 + 8, reg2 + 10)]['guards3'] =\
                                    ['clock = 0', 'theta_r >= 1', 'theta_l >= 0'] 
plant[reg2 + 8]['transitions'][(reg2 + 8, reg2 + 10)]['reset3'] =\
                                    ['clock\' := 0', 'theta_l\' := ' + str(np.pi / 2) + ' - temp1']
plant[reg2 + 8]['transitions'][(reg2 + 8, reg2 + 10)]['guards4'] =\
                                    ['clock = 0', 'theta_r >= 1', 'theta_l <= 0'] 
plant[reg2 + 8]['transitions'][(reg2 + 8, reg2 + 10)]['reset4'] =\
                                    ['clock\' := 0', 'theta_l\' := -' + str(np.pi / 2) + ' - temp1']

# temp2 = arctan(theta_r) // theta_r \in [-1, 1]
plant[reg2 + 9] = {}
plant[reg2 + 9]['name'] = 'arc_4_2_'
plant[reg2 + 9]['odetype'] = 'lti ode'
plant[reg2 + 9]['dynamics'] = {}
plant[reg2 + 9]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg2 + 9]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg2 + 9]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg2 + 9]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg2 + 9]['dynamics']['k'] = 'k\' = 0\n'
plant[reg2 + 9]['dynamics']['u'] = 'u\' = 0\n'
plant[reg2 + 9]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg2 + 9]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg2 + 9]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg2 + 9]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg2 + 9]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg2 + 9]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg2 + 9]['invariants'] = ['clock <= 0']
plant[reg2 + 9]['transitions'] = {}
plant[reg2 + 9]['transitions'][(reg2 + 9, mode1_reg2)] = {}
plant[reg2 + 9]['transitions'][(reg2 + 9, mode1_reg2)]['guards1'] = ['clock = 0'] 
plant[reg2 + 9]['transitions'][(reg2 + 9, mode1_reg2)]['reset1'] =\
                                ['clock\' := 0',\
                                 'theta_r\' := -temp2 - ' + str(np.pi),\
                                 'angle\' := y4 + ' + str(-LIDAR_RANGE),\
                                 'temp1\' := y4 + ' + str(-LIDAR_RANGE) + ' - theta_l',\
                                 'temp2\' := y4 + ' + str(-LIDAR_RANGE) + ' + temp2 + ' + str(np.pi)]

# theta_r = (1 / theta_r)  // theta_r \in [-\infty, -1] or [1, \infty]
plant[reg2 + 10] = {}
plant[reg2 + 10]['name'] = 'div_2_2_'
plant[reg2 + 10]['odetype'] = 'lti ode'
plant[reg2 + 10]['dynamics'] = {}
plant[reg2 + 10]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg2 + 10]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg2 + 10]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg2 + 10]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg2 + 10]['dynamics']['k'] = 'k\' = 0\n'
plant[reg2 + 10]['dynamics']['u'] = 'u\' = 0\n'
plant[reg2 + 10]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg2 + 10]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg2 + 10]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg2 + 10]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg2 + 10]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg2 + 10]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg2 + 10]['invariants'] = ['clock <= 0']
plant[reg2 + 10]['transitions'] = {}
plant[reg2 + 10]['transitions'][(reg2 + 10, reg2 + 11)] = {}
plant[reg2 + 10]['transitions'][(reg2 + 10, reg2 + 11)]['guards1'] = ['clock = 0'] 
plant[reg2 + 10]['transitions'][(reg2 + 10, reg2 + 11)]['reset1'] = ['clock\' := 0']

# temp2 = arctan(theta_r) // theta_r \in [-\infty, -1] or [1, \infty]
plant[reg2 + 11] = {}
plant[reg2 + 11]['name'] = 'arc_4_2_'
plant[reg2 + 11]['odetype'] = 'lti ode'
plant[reg2 + 11]['dynamics'] = {}
plant[reg2 + 11]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg2 + 11]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg2 + 11]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg2 + 11]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg2 + 11]['dynamics']['k'] = 'k\' = 0\n'
plant[reg2 + 11]['dynamics']['u'] = 'u\' = 0\n'
plant[reg2 + 11]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg2 + 11]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg2 + 11]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg2 + 11]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg2 + 11]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg2 + 11]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg2 + 11]['invariants'] = ['clock <= 0']
plant[reg2 + 11]['transitions'] = {}
plant[reg2 + 11]['transitions'][(reg2 + 11, mode1_reg2)] = {}
plant[reg2 + 11]['transitions'][(reg2 + 11, mode1_reg2)]['guards1'] = ['clock = 0', 'theta_r >= 0'] 
plant[reg2 + 11]['transitions'][(reg2 + 11, mode1_reg2)]['reset1'] =\
                                ['clock\' := 0',\
                                 'theta_r\' := - (' + str(np.pi / 2) + ' - temp2) - ' + str(np.pi),\
                                 'angle\' := y4 + ' + str(-LIDAR_RANGE),\
                                 'temp1\' := y4 + ' + str(-LIDAR_RANGE) + ' - theta_l',\
                                 'temp2\' := y4 + ' + str(-LIDAR_RANGE) +\
                                 ' + (' + str(np.pi / 2) + ' - temp2) + ' + str(np.pi)]
plant[reg2 + 11]['transitions'][(reg2 + 11, mode1_reg2)]['guards2'] = ['clock = 0', 'theta_r <= 0'] 
plant[reg2 + 11]['transitions'][(reg2 + 11, mode1_reg2)]['reset2'] =\
                                ['clock\' := 0',\
                                 'theta_r\' := - (-' + str(np.pi / 2) + ' - temp2) - ' + str(np.pi),\
                                 'angle\' := y4 + ' + str(-LIDAR_RANGE),\
                                 'temp1\' := y4 + ' + str(-LIDAR_RANGE) + ' - theta_l',\
                                 'temp2\' := y4 + ' + str(-LIDAR_RANGE) +\
                                 ' + (-' + str(np.pi / 2) + ' - temp2) + ' + str(np.pi)]

plant[reg3] = {}
plant[reg3]['name'] = 'reg3'
plant[reg3]['odetype'] = 'lti ode'
plant[reg3]['dynamics'] = {}
plant[reg3]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg3]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg3]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg3]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg3]['dynamics']['k'] = 'k\' = 0\n'
plant[reg3]['dynamics']['u'] = 'u\' = 0\n'
plant[reg3]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg3]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg3]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg3]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg3]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg3]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg3]['invariants'] = ['clock <= 0']
plant[reg3]['transitions'] = {}
plant[reg3]['transitions'][(reg3, reg3 + 11)] = {}
plant[reg3]['transitions'][(reg3, reg3 + 11)]['guards1'] = ['clock = 0']
plant[reg3]['transitions'][(reg3, reg3 + 11)]['reset1'] = ['temp1\' := 0', 'temp2\' := 0']

# theta_l = (1 / theta_l) // theta_l \in [1, infty]
plant[reg3 + 11] = {}
plant[reg3 + 11]['name'] = 'div_1_1_'
plant[reg3 + 11]['odetype'] = 'lti ode'
plant[reg3 + 11]['dynamics'] = {}
plant[reg3 + 11]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg3 + 11]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg3 + 11]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg3 + 11]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg3 + 11]['dynamics']['k'] = 'k\' = 0\n'
plant[reg3 + 11]['dynamics']['u'] = 'u\' = 0\n'
plant[reg3 + 11]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg3 + 11]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg3 + 11]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg3 + 11]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg3 + 11]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg3 + 11]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg3 + 11]['invariants'] = ['clock <= 0']
plant[reg3 + 11]['transitions'] = {}
plant[reg3 + 11]['transitions'][(reg3 + 11, reg3 + 12)] = {}
plant[reg3 + 11]['transitions'][(reg3 + 11, reg3 + 12)]['guards1'] = ['clock = 0']
plant[reg3 + 11]['transitions'][(reg3 + 11, reg3 + 12)]['reset1'] = ['temp1\' := 0', 'temp2\' := 0']

# temp1 = arctan(theta_l) // theta_l \in [1, infty]
plant[reg3 + 12] = {}
plant[reg3 + 12]['name'] = 'arc_3_1_'
plant[reg3 + 12]['odetype'] = 'lti ode'
plant[reg3 + 12]['dynamics'] = {}
plant[reg3 + 12]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg3 + 12]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg3 + 12]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg3 + 12]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg3 + 12]['dynamics']['k'] = 'k\' = 0\n'
plant[reg3 + 12]['dynamics']['u'] = 'u\' = 0\n'
plant[reg3 + 12]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg3 + 12]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg3 + 12]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg3 + 12]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg3 + 12]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg3 + 12]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg3 + 12]['invariants'] = ['clock <= 0']
plant[reg3 + 12]['transitions'] = {}
plant[reg3 + 12]['transitions'][(reg3 + 12, reg3 + 13)] = {}
plant[reg3 + 12]['transitions'][(reg3 + 12, reg3 + 13)]['guards1'] =\
                                ['clock = 0', 'theta_r >= -1', 'theta_r <= 1']
plant[reg3 + 12]['transitions'][(reg3 + 12, reg3 + 13)]['reset1'] =\
                                ['theta_l\' := (' + str(np.pi/2.0) + ' - temp1)']

plant[reg3 + 12]['transitions'][(reg3 + 12, reg3 + 14)] = {}
plant[reg3 + 12]['transitions'][(reg3 + 12, reg3 + 14)]['guards1'] =\
                                ['clock = 0', 'theta_r >= 1']
plant[reg3 + 12]['transitions'][(reg3 + 12, reg3 + 14)]['reset1'] =\
                                ['theta_l\' := (' + str(np.pi/2.0) + ' - temp1)']
plant[reg3 + 12]['transitions'][(reg3 + 12, reg3 + 14)]['guards2'] =\
                                ['clock = 0', 'theta_r <= -1']
plant[reg3 + 12]['transitions'][(reg3 + 12, reg3 + 14)]['reset2'] =\
                                ['theta_l\' := (' + str(np.pi/2.0) + ' - temp1)']


# temp2 = arctan(theta_r) // theta_r \in [-1, 1]
plant[reg3 + 13] = {}
plant[reg3 + 13]['name'] = 'arc_4_2_'
plant[reg3 + 13]['odetype'] = 'lti ode'
plant[reg3 + 13]['dynamics'] = {}
plant[reg3 + 13]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg3 + 13]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg3 + 13]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg3 + 13]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg3 + 13]['dynamics']['k'] = 'k\' = 0\n'
plant[reg3 + 13]['dynamics']['u'] = 'u\' = 0\n'
plant[reg3 + 13]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg3 + 13]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg3 + 13]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg3 + 13]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg3 + 13]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg3 + 13]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg3 + 13]['invariants'] = ['clock <= 0']
plant[reg3 + 13]['transitions'] = {}
plant[reg3 + 13]['transitions'][(reg3 + 13, mode1_reg3)] = {}
plant[reg3 + 13]['transitions'][(reg3 + 13, mode1_reg3)]['guards1'] = ['clock = 0']
plant[reg3 + 13]['transitions'][(reg3 + 13, mode1_reg3)]['reset1'] = ['clock\' := 0',\
                            'theta_r\' := ' + str(np.pi) + ' - temp2',\
                            'angle\' := y4 + ' + str(-LIDAR_RANGE),\
                            'temp1\' := y4 + ' + str(-LIDAR_RANGE) + ' - theta_l',\
                            'temp2\' := y4 + ' + str(-LIDAR_RANGE) + ' - ' + str(np.pi) + ' + temp2']

# theta_r = (1 / theta_r) // theta_r \in [-infty, -1] \cap [1, infty]
plant[reg3 + 14] = {}
plant[reg3 + 14]['name'] = 'div_2_2_'
plant[reg3 + 14]['odetype'] = 'lti ode'
plant[reg3 + 14]['dynamics'] = {}
plant[reg3 + 14]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg3 + 14]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg3 + 14]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg3 + 14]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg3 + 14]['dynamics']['k'] = 'k\' = 0\n'
plant[reg3 + 14]['dynamics']['u'] = 'u\' = 0\n'
plant[reg3 + 14]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg3 + 14]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg3 + 14]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg3 + 14]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg3 + 14]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg3 + 14]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg3 + 14]['invariants'] = ['clock <= 0']
plant[reg3 + 14]['transitions'] = {}
plant[reg3 + 14]['transitions'][(reg3 + 14, reg3 + 15)] = {}
plant[reg3 + 14]['transitions'][(reg3 + 14, reg3 + 15)]['guards1'] = ['clock = 0']
plant[reg3 + 14]['transitions'][(reg3 + 14, reg3 + 15)]['reset1'] = ['clock\' := 0']

# temp2 = arctan(theta_r) // theta_r \in [-infty, -1] \cap [1, infty]
plant[reg3 + 15] = {}
plant[reg3 + 15]['name'] = 'arc_4_2_'
plant[reg3 + 15]['odetype'] = 'lti ode'
plant[reg3 + 15]['dynamics'] = {}
plant[reg3 + 15]['dynamics']['y1'] = 'y1\' = 0\n'
plant[reg3 + 15]['dynamics']['y2'] = 'y2\' = 0\n'
plant[reg3 + 15]['dynamics']['y3'] = 'y3\' = 0\n'
plant[reg3 + 15]['dynamics']['y4'] = 'y4\' = 0\n'
plant[reg3 + 15]['dynamics']['k'] = 'k\' = 0\n'
plant[reg3 + 15]['dynamics']['u'] = 'u\' = 0\n'
plant[reg3 + 15]['dynamics']['angle'] = 'angle\' = 0\n'
plant[reg3 + 15]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
plant[reg3 + 15]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
plant[reg3 + 15]['dynamics']['temp1'] = 'temp1\' = 0\n'
plant[reg3 + 15]['dynamics']['temp2'] = 'temp2\' = 0\n'
for i in range(NUM_RAYS):
    plant[reg3 + 15]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
plant[reg3 + 15]['invariants'] = ['clock <= 0']
plant[reg3 + 15]['transitions'] = {}
plant[reg3 + 15]['transitions'][(reg3 + 15, mode1_reg3)] = {}
plant[reg3 + 15]['transitions'][(reg3 + 15, mode1_reg3)]['guards1'] = ['clock = 0', 'theta_r >= 0']
plant[reg3 + 15]['transitions'][(reg3 + 15, mode1_reg3)]['reset1'] = ['clock\' := 0',\
                        'theta_r\' := ' + str(np.pi) + ' - (' + str(np.pi/2.0) + ' - temp2)',\
                        'angle\' := y4 + ' + str(-LIDAR_RANGE),\
                        'temp1\' := y4 + ' + str(-LIDAR_RANGE) + ' - theta_l',\
                        'temp2\' := y4 + ' + str(-LIDAR_RANGE) +\
                                ' - ' + str(np.pi) + ' + (' + str(np.pi/2.0) + ' - temp2)']
plant[reg3 + 15]['transitions'][(reg3 + 15, mode1_reg3)]['guards2'] = ['clock = 0', 'theta_r <= 0']
plant[reg3 + 15]['transitions'][(reg3 + 15, mode1_reg3)]['reset2'] = ['clock\' := 0',\
                        'theta_r\' := ' + str(np.pi) + ' - (-' + str(np.pi/2.0) + ' - temp2)',\
                        'angle\' := y4 + ' + str(-LIDAR_RANGE),\
                        'temp1\' := y4 + ' + str(-LIDAR_RANGE) + ' - theta_l',\
                        'temp2\' := y4 + ' + str(-LIDAR_RANGE) +\
                                ' - ' + str(np.pi) + ' + (-' + str(np.pi/2.0) + ' - temp2)']

#Region 1
nextAngle = -LIDAR_RANGE + LIDAR_OFFSET
index = 0
curRay = 1

#while nextAngle <= LIDAR_RANGE + LIDAR_OFFSET:
while curRay <= NUM_RAYS:

    #name computation
    namePre = ''
    if nextAngle - LIDAR_OFFSET < 0:
        namePre = 'm'
    
    # first mode
    plant[mode1_reg1 + index] = {}
    plant[mode1_reg1 + index]['name'] = 'computing_ray_for_' + str(curRay)
    plant[mode1_reg1 + index]['odetype'] = 'lti ode'
    plant[mode1_reg1 + index]['dynamics'] = {}
    plant[mode1_reg1 + index]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg1 + index]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg1 + index]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg1 + index]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg1 + index]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg1 + index]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg1 + index]['dynamics']['angle'] = 'angle\' = 0\n'
    plant[mode1_reg1 + index]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg1 + index]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg1 + index]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg1 + index]['dynamics']['temp2'] = 'temp2\' = 0\n'
    for i in range(NUM_RAYS):
        plant[mode1_reg1 + index]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
        
    plant[mode1_reg1 + index]['invariants'] = ['clock <= 0']
    plant[mode1_reg1 + index]['transitions'] = {}

    # self transitions to convert angle to (-180, 180), i.e., (-pi, pi)
    plant[mode1_reg1 + index]['transitions'][(mode1_reg1 + index,mode1_reg1 + index)] = {}
    plant[mode1_reg1 + index]['transitions'][(mode1_reg1 + index,mode1_reg1 + index)]['guards1'] =\
                                    ['clock = 0', 'angle >= ' + str(np.pi)]
    plant[mode1_reg1 + index]['transitions'][(mode1_reg1 + index,mode1_reg1 + index)]['reset1'] =\
                                    ['angle\' := angle - ' + str(2 * np.pi),\
                                    'temp1\' := temp1 - ' + str(2 * np.pi), 'temp2\' := temp2 - ' + str(2 * np.pi)]
    plant[mode1_reg1 + index]['transitions'][(mode1_reg1 + index,mode1_reg1 + index)]['guards2'] =\
                                    ['clock = 0', 'angle <= ' + str(-np.pi)]
    plant[mode1_reg1 + index]['transitions'][(mode1_reg1 + index,mode1_reg1 + index)]['reset2'] =\
                                    ['angle\' := angle + ' + str(2 * np.pi), 'temp1\' := temp1 + ' + str(2 * np.pi),\
                                    'temp2\' := temp2 + ' + str(2 * np.pi)]


    # transition to correct wall

    #NB: some epsilons here
    # [-LIDAR_RANGE, theta_r]
    plant[mode1_reg1 + index]['transitions'][(mode1_reg1 + index,mode1_reg1 + index + 1)] = {}
    plant[mode1_reg1 + index]['transitions'][(mode1_reg1 + index,mode1_reg1 + index + 1)]['guards1'] =\
                                    ['clock = 0', 'angle >= ' + str(-np.pi), 'angle <= ' + str(np.pi), 'temp2 <= 0.01']
    plant[mode1_reg1 + index]['transitions'][(mode1_reg1 + index,mode1_reg1 + index + 1)]['reset1'] =\
                                    ['clock\' := 0', 'f' + str((index + 8)/8) + '\' := 0', 'angle\' := (' + str(np.pi/2) + ' + angle)']

    # (theta_r, theta_l]
    plant[mode1_reg1 + index]['transitions'][(mode1_reg1 + index,mode1_reg1 + index + 2)] = {}
    plant[mode1_reg1 + index]['transitions'][(mode1_reg1 + index,mode1_reg1 + index + 2)]['guards1'] =\
                                    ['clock = 0', 'angle >= ' + str(-np.pi), 'angle <= ' + str(np.pi),\
                                    'temp2 >= 0.01', 'temp1 <= 0.01']
    plant[mode1_reg1 + index]['transitions'][(mode1_reg1 + index,mode1_reg1 + index + 2)]['reset1'] =\
                                    ['clock\' := 0', 'f' + str((index + 8)/8) + '\' := 0', 'angle\' := angle']

    # (theta_l, LIDAR_RANGE]
    plant[mode1_reg1 + index]['transitions'][(mode1_reg1 + index,mode1_reg1 + index + 3)] = {}
    plant[mode1_reg1 + index]['transitions'][(mode1_reg1 + index,mode1_reg1 + index + 3)]['guards1'] =\
                                    ['clock = 0', 'angle >= ' + str(-np.pi), 'angle <= ' + str(np.pi), 'temp1 >= 0.01']
    plant[mode1_reg1 + index]['transitions'][(mode1_reg1 + index,mode1_reg1 + index + 3)]['reset1'] =\
                                    ['clock\' := 0', 'f' + str((index + 8)/8) + '\' := 0',\
                                     'angle\' := (' + str(np.pi/2) + ' - angle)']
    
    # compute cos(angle) [-LIDAR_RANGE, theta_r]
    plant[mode1_reg1 + index + 1] = {}
    plant[mode1_reg1 + index + 1]['name'] = 'right_wall_'
    plant[mode1_reg1 + index + 1]['odetype'] = 'lti ode'
    plant[mode1_reg1 + index + 1]['dynamics'] = {}
    plant[mode1_reg1 + index + 1]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg1 + index + 1]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg1 + index + 1]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg1 + index + 1]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg1 + index + 1]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg1 + index + 1]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg1 + index + 1]['dynamics']['angle'] = 'angle\' = 0\n'
    plant[mode1_reg1 + index + 1]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg1 + index + 1]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg1 + index + 1]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg1 + index + 1]['dynamics']['temp2'] = 'temp2\' = 0\n'
    for i in range(NUM_RAYS):
        plant[mode1_reg1 + index + 1]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'

    plant[mode1_reg1 + index + 1]['invariants'] = ['clock <= 0']
    plant[mode1_reg1 + index + 1]['transitions'] = {}
    plant[mode1_reg1 + index + 1]['transitions'][(mode1_reg1 + index + 1,mode1_reg1 + index + 4)] = {}
    plant[mode1_reg1 + index + 1]['transitions'][(mode1_reg1 + index + 1,mode1_reg1 + index + 4)]['guards1'] = ['clock = 0']
    plant[mode1_reg1 + index + 1]['transitions'][(mode1_reg1 + index + 1,mode1_reg1 + index + 4)]['reset1'] = ['clock\' := 0']
    
    # compute sec(angle)
    plant[mode1_reg1 + index + 4] = {}
    plant[mode1_reg1 + index + 4]['name'] = 'sec_0_0_'
    plant[mode1_reg1 + index + 4]['odetype'] = 'lti ode'
    plant[mode1_reg1 + index + 4]['dynamics'] = {}
    plant[mode1_reg1 + index + 4]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg1 + index + 4]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg1 + index + 4]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg1 + index + 4]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg1 + index + 4]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg1 + index + 4]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg1 + index + 4]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg1 + index + 4]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg1 + index + 4]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg1 + index + 4]['dynamics']['temp2'] = 'temp2\' = 0\n'
    plant[mode1_reg1 + index + 4]['dynamics']['angle'] = 'angle\' = 0\n'
    for i in range(NUM_RAYS):
        plant[mode1_reg1 + index + 4]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
        
    plant[mode1_reg1 + index + 4]['invariants'] = ['clock <= 0']
    plant[mode1_reg1 + index + 4]['transitions'] = {}
    plant[mode1_reg1 + index + 4]['transitions'][(mode1_reg1 + index + 4,mode1_reg1 + index + 7)] = {}
    plant[mode1_reg1 + index + 4]['transitions'][(mode1_reg1 + index + 4,mode1_reg1 + index + 7)]['guards1'] = ['clock = 0']
    plant[mode1_reg1 + index + 4]['transitions'][(mode1_reg1 + index + 4,mode1_reg1 + index + 7)]['reset1'] = ['clock\' := 0', 'f' + str((index + 8)/8) + '\' := angle * (' + str(HALLWAY_WIDTH) + ' - y1)']

    # compute cos(angle) (theta_r, theta_l]
    plant[mode1_reg1 + index + 2] = {}
    plant[mode1_reg1 + index + 2]['name'] = 'front_wall_'
    plant[mode1_reg1 + index + 2]['odetype'] = 'lti ode'
    plant[mode1_reg1 + index + 2]['dynamics'] = {}
    plant[mode1_reg1 + index + 2]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg1 + index + 2]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg1 + index + 2]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg1 + index + 2]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg1 + index + 2]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg1 + index + 2]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg1 + index + 2]['dynamics']['angle'] = 'angle\' = 0\n'
    plant[mode1_reg1 + index + 2]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg1 + index + 2]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg1 + index + 2]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg1 + index + 2]['dynamics']['temp2'] = 'temp2\' = 0\n'    
    
    for i in range(NUM_RAYS):
        plant[mode1_reg1 + index + 2]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'        

    plant[mode1_reg1 + index + 2]['invariants'] = ['clock <= 0']
    plant[mode1_reg1 + index + 2]['transitions'] = {}            
    plant[mode1_reg1 + index + 2]['transitions'][(mode1_reg1 + index + 2,mode1_reg1 + index + 5)] = {}
    plant[mode1_reg1 + index + 2]['transitions'][(mode1_reg1 + index + 2,mode1_reg1 + index + 5)]['guards1'] = ['clock = 0']
    plant[mode1_reg1 + index + 2]['transitions'][(mode1_reg1 + index + 2,mode1_reg1 + index + 5)]['reset1'] = ['clock\' := 0']

    # 
    plant[mode1_reg1 + index + 5] = {}
    plant[mode1_reg1 + index + 5]['name'] = 'sec_0_0_'
    plant[mode1_reg1 + index + 5]['odetype'] = 'lti ode'
    plant[mode1_reg1 + index + 5]['dynamics'] = {}
    plant[mode1_reg1 + index + 5]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg1 + index + 5]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg1 + index + 5]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg1 + index + 5]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg1 + index + 5]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg1 + index + 5]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg1 + index + 5]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg1 + index + 5]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg1 + index + 5]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg1 + index + 5]['dynamics']['temp2'] = 'temp2\' = 0\n'
    plant[mode1_reg1 + index + 5]['dynamics']['angle'] = 'angle\' = 0\n'
    for i in range(NUM_RAYS):
        plant[mode1_reg1 + index + 5]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'

    plant[mode1_reg1 + index + 5]['invariants'] = ['clock <= 0']
    plant[mode1_reg1 + index + 5]['transitions'] = {}
    plant[mode1_reg1 + index + 5]['transitions'][(mode1_reg1 + index + 5,mode1_reg1 + index + 7)] = {}
    plant[mode1_reg1 + index + 5]['transitions'][(mode1_reg1 + index + 5,mode1_reg1 + index + 7)]['guards1'] = ['clock = 0']
    plant[mode1_reg1 + index + 5]['transitions'][(mode1_reg1 + index + 5,mode1_reg1 + index + 7)]['reset1'] = ['clock\' := 0', 'f' + str((index + 8)/8) + '\' := angle * y2']

    # compute cos(angle) (theta_l, LIDAR_RANGE]
    plant[mode1_reg1 + index + 3] = {}
    plant[mode1_reg1 + index + 3]['name'] = 'left_wall_'
    plant[mode1_reg1 + index + 3]['odetype'] = 'lti ode'
    plant[mode1_reg1 + index + 3]['dynamics'] = {}
    plant[mode1_reg1 + index + 3]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg1 + index + 3]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg1 + index + 3]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg1 + index + 3]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg1 + index + 3]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg1 + index + 3]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg1 + index + 3]['dynamics']['angle'] = 'angle\' = 0\n'
    plant[mode1_reg1 + index + 3]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg1 + index + 3]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg1 + index + 3]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg1 + index + 3]['dynamics']['temp2'] = 'temp2\' = 0\n'    
    for i in range(NUM_RAYS):
        plant[mode1_reg1 + index + 3]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'

    plant[mode1_reg1 + index + 3]['invariants'] = ['clock <= 0']
    plant[mode1_reg1 + index + 3]['transitions'] = {}            
    plant[mode1_reg1 + index + 3]['transitions'][(mode1_reg1 + index + 3,mode1_reg1 + index + 6)] = {}
    plant[mode1_reg1 + index + 3]['transitions'][(mode1_reg1 + index + 3,mode1_reg1 + index + 6)]['guards1'] = ['clock = 0']
    plant[mode1_reg1 + index + 3]['transitions'][(mode1_reg1 + index + 3,mode1_reg1 + index + 6)]['reset1'] = ['clock\' := 0']

    # 
    plant[mode1_reg1 + index + 6] = {}
    plant[mode1_reg1 + index + 6]['name'] = 'sec_0_0_'
    plant[mode1_reg1 + index + 6]['odetype'] = 'lti ode'
    plant[mode1_reg1 + index + 6]['dynamics'] = {}
    plant[mode1_reg1 + index + 6]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg1 + index + 6]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg1 + index + 6]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg1 + index + 6]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg1 + index + 6]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg1 + index + 6]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg1 + index + 6]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg1 + index + 6]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg1 + index + 6]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg1 + index + 6]['dynamics']['temp2'] = 'temp2\' = 0\n'    
    plant[mode1_reg1 + index + 6]['dynamics']['angle'] = 'angle\' = 0\n'
    for i in range(NUM_RAYS):
        plant[mode1_reg1 + index + 6]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
        
    plant[mode1_reg1 + index + 6]['invariants'] = ['clock <= 0']
    plant[mode1_reg1 + index + 6]['transitions'] = {}
    plant[mode1_reg1 + index + 6]['transitions'][(mode1_reg1 + index + 6,mode1_reg1 + index + 7)] = {}
    plant[mode1_reg1 + index + 6]['transitions'][(mode1_reg1 + index + 6,mode1_reg1 + index + 7)]['guards1'] = ['clock = 0']
    plant[mode1_reg1 + index + 6]['transitions'][(mode1_reg1 + index + 6,mode1_reg1 + index + 7)]['reset1'] = ['clock\' := 0', 'f' + str((index + 8)/8) + '\' := angle * y1']

    # last mode
    plant[mode1_reg1 + index + 7] = {}
    plant[mode1_reg1 + index + 7]['name'] = ''
    plant[mode1_reg1 + index + 7]['odetype'] = 'lti ode'
    plant[mode1_reg1 + index + 7]['dynamics'] = {}
    plant[mode1_reg1 + index + 7]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg1 + index + 7]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg1 + index + 7]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg1 + index + 7]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg1 + index + 7]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg1 + index + 7]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg1 + index + 7]['dynamics']['angle'] = 'angle\' = 0\n'
    plant[mode1_reg1 + index + 7]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg1 + index + 7]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg1 + index + 7]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg1 + index + 7]['dynamics']['temp2'] = 'temp2\' = 0\n'    
    for i in range(NUM_RAYS):
        plant[mode1_reg1 + index + 7]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
        
    plant[mode1_reg1 + index + 7]['invariants'] = ['clock <= 0']
    plant[mode1_reg1 + index + 7]['transitions'] = {}

    # if last ray, need to transition to m0 (in composed transitions)
    #if nextAngle == LIDAR_RANGE + LIDAR_OFFSET:
    if curRay == NUM_RAYS:
        break

    plant[mode1_reg1 + index + 7]['transitions'][(mode1_reg1 + index + 7,mode1_reg1 + index + 8)] = {}
    plant[mode1_reg1 + index + 7]['transitions'][(mode1_reg1 + index + 7,mode1_reg1 + index + 8)]['guards1'] =\
                                ['clock = 0', 'f' + str((index + 8)/8) + ' >= ' + str(LIDAR_MAX_DISTANCE)]
    plant[mode1_reg1 + index + 7]['transitions'][(mode1_reg1 + index + 7,mode1_reg1 + index + 8)]['reset1'] =\
                                ['f' + str((index + 8)/8) + '\' := ' + str(LIDAR_MAX_DISTANCE),\
                                 'angle\' := y4 + ' + str(nextAngle),\
                                 'temp1\' := y4 + ' + str(nextAngle) + ' - theta_l',\
                                 'temp2\' := y4 + ' + str(nextAngle) + ' - theta_r']
    
    plant[mode1_reg1 + index + 7]['transitions'][(mode1_reg1 + index + 7,mode1_reg1 + index + 8)]['guards2'] =\
                                ['clock = 0', 'f' + str((index + 8)/8) + ' <= ' + str(LIDAR_MAX_DISTANCE)]
    plant[mode1_reg1 + index + 7]['transitions'][(mode1_reg1 + index + 7,mode1_reg1 + index + 8)]['reset2'] =\
                                ['angle\' := y4 + ' + str(nextAngle),\
                                 'temp1\' := y4 + ' + str(nextAngle) + ' - theta_l',\
                                 'temp2\' := y4 + ' + str(nextAngle) + ' - theta_r']

    nextAngle += LIDAR_OFFSET
    index += 8
    curRay += 1

#Region 2
nextAngle = -LIDAR_RANGE + LIDAR_OFFSET
curRay = 1
index = 0

#while nextAngle <= LIDAR_RANGE + LIDAR_OFFSET:
while curRay <= NUM_RAYS:

    # first mode
    plant[mode1_reg2 + index] = {}
    plant[mode1_reg2 + index]['name'] = ''
    plant[mode1_reg2 + index]['odetype'] = 'lti ode'
    plant[mode1_reg2 + index]['dynamics'] = {}
    plant[mode1_reg2 + index]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg2 + index]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg2 + index]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg2 + index]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg2 + index]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg2 + index]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg2 + index]['dynamics']['angle'] = 'angle\' = 0\n'
    plant[mode1_reg2 + index]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg2 + index]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg2 + index]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg2 + index]['dynamics']['temp2'] = 'temp2\' = 0\n'    
    for i in range(NUM_RAYS):
        plant[mode1_reg2 + index]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
        
    plant[mode1_reg2 + index]['invariants'] = ['clock <= 0']
    plant[mode1_reg2 + index]['transitions'] = {}
    
    #self transitions to convert angle to (-180, 180), i.e., (-pi, pi)
    plant[mode1_reg2 + index]['transitions'][(mode1_reg2 + index,mode1_reg2 + index)] = {}
    plant[mode1_reg2 + index]['transitions'][(mode1_reg2 + index,mode1_reg2 + index)]['guards1'] =\
                                        ['clock = 0', 'angle >= ' + str(np.pi)]
    plant[mode1_reg2 + index]['transitions'][(mode1_reg2 + index,mode1_reg2 + index)]['reset1'] =\
                                        ['angle\' := angle - ' + str(2 * np.pi),\
                                         'temp1\' := temp1 - ' + str(2 * np.pi),\
                                         'temp2\' := temp2 - ' + str(2 * np.pi)]
    plant[mode1_reg2 + index]['transitions'][(mode1_reg2 + index,mode1_reg2 + index)]['guards2'] =\
                                        ['clock = 0', 'angle <= ' + str(-np.pi)]
    plant[mode1_reg2 + index]['transitions'][(mode1_reg2 + index,mode1_reg2 + index)]['reset2'] =\
                                        ['angle\' := angle + ' + str(2 * np.pi),\
                                         'temp1\' := temp1 + ' + str(2 * np.pi),\
                                         'temp2\' := temp2 + ' + str(2 * np.pi)]

    #transition to correct wall

    #NB: some epsilons here
    # [-LIDAR_RANGE, theta_r]
    plant[mode1_reg2 + index]['transitions'][(mode1_reg2 + index,mode1_reg2 + index + 1)] = {}
    plant[mode1_reg2 + index]['transitions'][(mode1_reg2 + index,mode1_reg2 + index + 1)]['guards1'] =\
                                        ['clock = 0', 'angle >= ' + str(-np.pi),\
                                         'angle <= ' + str(np.pi),\
                                         'temp2 <= 0.01']
    plant[mode1_reg2 + index]['transitions'][(mode1_reg2 + index,mode1_reg2 + index + 1)]['reset1'] =\
                                        ['clock\' := 0', 'f' + str((index + 10)/10) + '\' := 0',\
                                         'angle\' := ' + str(np.pi / 2) + ' + angle']
    
    # (theta_r, -90)
    plant[mode1_reg2 + index]['transitions'][(mode1_reg2 + index,mode1_reg2 + index + 2)] = {}
    plant[mode1_reg2 + index]['transitions'][(mode1_reg2 + index,mode1_reg2 + index + 2)]['guards1'] =\
                                        ['clock = 0', 'angle >= ' + str(-np.pi),\
                                         'angle <= ' + str(np.pi),\
                                         'temp2 >= 0.01',\
                                         'angle <= ' + str(-np.pi/2)]
    plant[mode1_reg2 + index]['transitions'][(mode1_reg2 + index,mode1_reg2 + index + 2)]['reset1'] =\
                                        ['clock\' := 0', 'f' + str((index + 10)/10) + '\' := 0',\
                                         'angle\' := ' + str(np.pi) + ' + angle']

    # (-90, theta_l]
    plant[mode1_reg2 + index]['transitions'][(mode1_reg2 + index,mode1_reg2 + index + 3)] = {}
    plant[mode1_reg2 + index]['transitions'][(mode1_reg2 + index,mode1_reg2 + index + 3)]['guards1'] =\
                                        ['clock = 0', 'angle >= ' + str(-np.pi),\
                                         'angle <= ' + str(np.pi),\
                                         'temp1 <= 0.01',\
                                         'angle >= ' + str(-np.pi/2)]
    plant[mode1_reg2 + index]['transitions'][(mode1_reg2 + index,mode1_reg2 + index + 3)]['reset1'] =\
                                        ['clock\' := 0', 'f' + str((index + 10)/10) + '\' := 0',\
                                         'angle\' := angle']

    # (theta_l, LIDAR_RANGE]
    plant[mode1_reg2 + index]['transitions'][(mode1_reg2 + index,mode1_reg2 + index + 4)] = {}
    plant[mode1_reg2 + index]['transitions'][(mode1_reg2 + index,mode1_reg2 + index + 4)]['guards1'] =\
                                        ['clock = 0', 'angle >= ' + str(-np.pi),\
                                         'angle <= ' + str(np.pi),\
                                         'temp1 >= 0.01']
    plant[mode1_reg2 + index]['transitions'][(mode1_reg2 + index,mode1_reg2 + index + 4)]['reset1'] =\
                                        ['clock\' := 0', 'f' + str((index + 10)/10) + '\' := 0',\
                                         'angle\' := ' + str(np.pi/2) + ' - angle']

    # compute cos(angle) [-LIDAR_RANGE, theta_r]
    plant[mode1_reg2 + index + 1] = {}
    plant[mode1_reg2 + index + 1]['name'] = ''
    plant[mode1_reg2 + index + 1]['odetype'] = 'lti ode'
    plant[mode1_reg2 + index + 1]['dynamics'] = {}
    plant[mode1_reg2 + index + 1]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg2 + index + 1]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg2 + index + 1]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg2 + index + 1]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg2 + index + 1]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg2 + index + 1]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg2 + index + 1]['dynamics']['angle'] = 'angle\' = 0\n'
    plant[mode1_reg2 + index + 1]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg2 + index + 1]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg2 + index + 1]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg2 + index + 1]['dynamics']['temp2'] = 'temp2\' = 0\n'        
    for i in range(NUM_RAYS):
        plant[mode1_reg2 + index + 1]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'

    plant[mode1_reg2 + index + 1]['invariants'] = ['clock <= 0']
    plant[mode1_reg2 + index + 1]['transitions'] = {}
    plant[mode1_reg2 + index + 1]['transitions'][(mode1_reg2 + index + 1,mode1_reg2 + index + 5)] = {}
    plant[mode1_reg2 + index + 1]['transitions'][(mode1_reg2 + index + 1,mode1_reg2 + index + 5)]['guards1'] = ['clock = 0']
    plant[mode1_reg2 + index + 1]['transitions'][(mode1_reg2 + index + 1,mode1_reg2 + index + 5)]['reset1'] = ['clock\' := 0']
    
    # 
    plant[mode1_reg2 + index + 5] = {}
    plant[mode1_reg2 + index + 5]['name'] = 'sec_0_0_'
    plant[mode1_reg2 + index + 5]['odetype'] = 'lti ode'
    plant[mode1_reg2 + index + 5]['dynamics'] = {}
    plant[mode1_reg2 + index + 5]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg2 + index + 5]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg2 + index + 5]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg2 + index + 5]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg2 + index + 5]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg2 + index + 5]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg2 + index + 5]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg2 + index + 5]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg2 + index + 5]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg2 + index + 5]['dynamics']['temp2'] = 'temp2\' = 0\n'
    plant[mode1_reg2 + index + 5]['dynamics']['angle'] = 'angle\' = 0\n'    
    for i in range(NUM_RAYS):
        plant[mode1_reg2 + index + 5]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
        
    plant[mode1_reg2 + index + 5]['invariants'] = ['clock <= 0']
    plant[mode1_reg2 + index + 5]['transitions'] = {}
    plant[mode1_reg2 + index + 5]['transitions'][(mode1_reg2 + index + 5,mode1_reg2 + index + 9)] = {}
    plant[mode1_reg2 + index + 5]['transitions'][(mode1_reg2 + index + 5,mode1_reg2 + index + 9)]['guards1'] = ['clock = 0']
    plant[mode1_reg2 + index + 5]['transitions'][(mode1_reg2 + index + 5,mode1_reg2 + index + 9)]['reset1'] =\
                                ['clock\' := 0', 'f' + str((index + 10)/10) + '\' := (' + str(HALLWAY_WIDTH) + ' - y1) * angle']

    # compute cos(angle) (theta_r, -90) 
    plant[mode1_reg2 + index + 2] = {}
    plant[mode1_reg2 + index + 2]['name'] = ''
    plant[mode1_reg2 + index + 2]['odetype'] = 'lti ode'
    plant[mode1_reg2 + index + 2]['dynamics'] = {}
    plant[mode1_reg2 + index + 2]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg2 + index + 2]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg2 + index + 2]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg2 + index + 2]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg2 + index + 2]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg2 + index + 2]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg2 + index + 2]['dynamics']['angle'] = 'angle\' = 0\n'
    plant[mode1_reg2 + index + 2]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg2 + index + 2]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg2 + index + 2]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg2 + index + 2]['dynamics']['temp2'] = 'temp2\' = 0\n'        
    for i in range(NUM_RAYS):
        plant[mode1_reg2 + index + 2]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'

    plant[mode1_reg2 + index + 2]['invariants'] = ['clock <= 0', 'angle <= ' + str(np.pi / 2)]
    plant[mode1_reg2 + index + 2]['transitions'] = {}
    plant[mode1_reg2 + index + 2]['transitions'][(mode1_reg2 + index + 2,mode1_reg2 + index + 6)] = {}
    plant[mode1_reg2 + index + 2]['transitions'][(mode1_reg2 + index + 2,mode1_reg2 + index + 6)]['guards1'] = ['clock = 0']
    plant[mode1_reg2 + index + 2]['transitions'][(mode1_reg2 + index + 2,mode1_reg2 + index + 6)]['reset1'] = ['clock\' := 0']    

    # 
    plant[mode1_reg2 + index + 6] = {}
    plant[mode1_reg2 + index + 6]['name'] = 'sec_0_0_'
    plant[mode1_reg2 + index + 6]['odetype'] = 'lti ode'
    plant[mode1_reg2 + index + 6]['dynamics'] = {}
    plant[mode1_reg2 + index + 6]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg2 + index + 6]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg2 + index + 6]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg2 + index + 6]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg2 + index + 6]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg2 + index + 6]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg2 + index + 6]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg2 + index + 6]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg2 + index + 6]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg2 + index + 6]['dynamics']['temp2'] = 'temp2\' = 0\n'        
    plant[mode1_reg2 + index + 6]['dynamics']['angle'] = 'angle\' = 0\n'
    for i in range(NUM_RAYS):
        plant[mode1_reg2 + index + 6]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
        
    plant[mode1_reg2 + index + 6]['invariants'] = ['clock <= 0']
    plant[mode1_reg2 + index + 6]['transitions'] = {}
    plant[mode1_reg2 + index + 6]['transitions'][(mode1_reg2 + index + 6,mode1_reg2 + index + 9)] = {}
    plant[mode1_reg2 + index + 6]['transitions'][(mode1_reg2 + index + 6,mode1_reg2 + index + 9)]['guards1'] = ['clock = 0']
    plant[mode1_reg2 + index + 6]['transitions'][(mode1_reg2 + index + 6,mode1_reg2 + index + 9)]['reset1'] =\
                                ['clock\' := 0', 'f' + str((index + 10)/10) + '\' := (' + str(HALLWAY_WIDTH) + ' - y2) * angle']

    # compute cos(angle) (-90, theta_l]
    plant[mode1_reg2 + index + 3] = {}
    plant[mode1_reg2 + index + 3]['name'] = ''
    plant[mode1_reg2 + index + 3]['odetype'] = 'lti ode'
    plant[mode1_reg2 + index + 3]['dynamics'] = {}
    plant[mode1_reg2 + index + 3]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg2 + index + 3]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg2 + index + 3]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg2 + index + 3]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg2 + index + 3]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg2 + index + 3]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg2 + index + 3]['dynamics']['angle'] = 'angle\' = 0\n'
    plant[mode1_reg2 + index + 3]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg2 + index + 3]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg2 + index + 3]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg2 + index + 3]['dynamics']['temp2'] = 'temp2\' = 0\n'        
    for i in range(NUM_RAYS):
        plant[mode1_reg2 + index + 3]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'

    plant[mode1_reg2 + index + 3]['invariants'] = ['clock <= 0', '-angle <= ' + str(np.pi / 2)]
    plant[mode1_reg2 + index + 3]['transitions'] = {}
    plant[mode1_reg2 + index + 3]['transitions'][(mode1_reg2 + index + 3,mode1_reg2 + index + 7)] = {}
    plant[mode1_reg2 + index + 3]['transitions'][(mode1_reg2 + index + 3,mode1_reg2 + index + 7)]['guards1'] = ['clock = 0']
    plant[mode1_reg2 + index + 3]['transitions'][(mode1_reg2 + index + 3,mode1_reg2 + index + 7)]['reset1'] = ['clock\' := 0']

    # 
    plant[mode1_reg2 + index + 7] = {}
    plant[mode1_reg2 + index + 7]['name'] = 'sec_0_0_'
    plant[mode1_reg2 + index + 7]['odetype'] = 'lti ode'
    plant[mode1_reg2 + index + 7]['dynamics'] = {}
    plant[mode1_reg2 + index + 7]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg2 + index + 7]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg2 + index + 7]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg2 + index + 7]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg2 + index + 7]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg2 + index + 7]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg2 + index + 7]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg2 + index + 7]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg2 + index + 7]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg2 + index + 7]['dynamics']['temp2'] = 'temp2\' = 0\n'        
    plant[mode1_reg2 + index + 7]['dynamics']['angle'] = 'angle\' = 0\n'
    for i in range(NUM_RAYS):
        plant[mode1_reg2 + index + 7]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
        
    plant[mode1_reg2 + index + 7]['invariants'] = ['clock <= 0']
    plant[mode1_reg2 + index + 7]['transitions'] = {}
    plant[mode1_reg2 + index + 7]['transitions'][(mode1_reg2 + index + 7,mode1_reg2 + index + 9)] = {}
    plant[mode1_reg2 + index + 7]['transitions'][(mode1_reg2 + index + 7,mode1_reg2 + index + 9)]['guards1'] = ['clock = 0']
    plant[mode1_reg2 + index + 7]['transitions'][(mode1_reg2 + index + 7,mode1_reg2 + index + 9)]['reset1'] =\
                                            ['clock\' := 0', 'f' + str((index + 10)/10) + '\' := y2 * angle']

    # compute cos(angle) (theta_l, 180]
    plant[mode1_reg2 + index + 4] = {}
    plant[mode1_reg2 + index + 4]['name'] = ''
    plant[mode1_reg2 + index + 4]['odetype'] = 'lti ode'
    plant[mode1_reg2 + index + 4]['dynamics'] = {}
    plant[mode1_reg2 + index + 4]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg2 + index + 4]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg2 + index + 4]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg2 + index + 4]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg2 + index + 4]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg2 + index + 4]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg2 + index + 4]['dynamics']['angle'] = 'angle\' = 0\n'
    plant[mode1_reg2 + index + 4]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg2 + index + 4]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg2 + index + 4]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg2 + index + 4]['dynamics']['temp2'] = 'temp2\' = 0\n'        
    for i in range(NUM_RAYS):
        plant[mode1_reg2 + index + 4]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'

    plant[mode1_reg2 + index + 4]['invariants'] = ['clock <= 0']
    plant[mode1_reg2 + index + 4]['transitions'] = {}
    plant[mode1_reg2 + index + 4]['transitions'][(mode1_reg2 + index + 4,mode1_reg2 + index + 8)] = {}
    plant[mode1_reg2 + index + 4]['transitions'][(mode1_reg2 + index + 4,mode1_reg2 + index + 8)]['guards1'] = ['clock = 0']
    plant[mode1_reg2 + index + 4]['transitions'][(mode1_reg2 + index + 4,mode1_reg2 + index + 8)]['reset1'] = ['clock\' := 0']    

    # 
    plant[mode1_reg2 + index + 8] = {}
    plant[mode1_reg2 + index + 8]['name'] = 'sec_0_0_'
    plant[mode1_reg2 + index + 8]['odetype'] = 'lti ode'
    plant[mode1_reg2 + index + 8]['dynamics'] = {}
    plant[mode1_reg2 + index + 8]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg2 + index + 8]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg2 + index + 8]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg2 + index + 8]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg2 + index + 8]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg2 + index + 8]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg2 + index + 8]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg2 + index + 8]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg2 + index + 8]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg2 + index + 8]['dynamics']['temp2'] = 'temp2\' = 0\n'        
    plant[mode1_reg2 + index + 8]['dynamics']['angle'] = 'angle\' = 0\n'
    for i in range(NUM_RAYS):
        plant[mode1_reg2 + index + 8]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
        
    plant[mode1_reg2 + index + 8]['invariants'] = ['clock <= 0']
    plant[mode1_reg2 + index + 8]['transitions'] = {}
    plant[mode1_reg2 + index + 8]['transitions'][(mode1_reg2 + index + 8,mode1_reg2 + index + 9)] = {}
    plant[mode1_reg2 + index + 8]['transitions'][(mode1_reg2 + index + 8,mode1_reg2 + index + 9)]['guards1'] = ['clock = 0']
    plant[mode1_reg2 + index + 8]['transitions'][(mode1_reg2 + index + 8,mode1_reg2 + index + 9)]['reset1'] =\
                                            ['clock\' := 0', 'f' + str((index + 10)/10) + '\' := y1 * angle']

    # last mode
    plant[mode1_reg2 + index + 9] = {}
    plant[mode1_reg2 + index + 9]['name'] = ''
    plant[mode1_reg2 + index + 9]['odetype'] = 'lti ode'
    plant[mode1_reg2 + index + 9]['dynamics'] = {}
    plant[mode1_reg2 + index + 9]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg2 + index + 9]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg2 + index + 9]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg2 + index + 9]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg2 + index + 9]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg2 + index + 9]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg2 + index + 9]['dynamics']['angle'] = 'angle\' = 0\n'
    plant[mode1_reg2 + index + 9]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg2 + index + 9]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg2 + index + 9]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg2 + index + 9]['dynamics']['temp2'] = 'temp2\' = 0\n'        
    for i in range(NUM_RAYS):
        plant[mode1_reg2 + index + 9]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
        
    plant[mode1_reg2 + index + 9]['invariants'] = ['clock <= 0']
    plant[mode1_reg2 + index + 9]['transitions'] = {}

    # if last ray, need to transition to m0 (in composed transitions)
    #if nextAngle == LIDAR_RANGE + LIDAR_OFFSET:
    if curRay == NUM_RAYS:
        break

    plant[mode1_reg2 + index + 9]['transitions'][(mode1_reg2 + index + 9,mode1_reg2 + index + 10)] = {}
    plant[mode1_reg2 + index + 9]['transitions'][(mode1_reg2 + index + 9,mode1_reg2 + index + 10)]['guards1'] =\
                                        ['clock = 0', 'f' + str((index + 10)/10) + ' >= ' + str(LIDAR_MAX_DISTANCE)]
    plant[mode1_reg2 + index + 9]['transitions'][(mode1_reg2 + index + 9,mode1_reg2 + index + 10)]['reset1'] =\
                                        ['f' + str((index + 10)/10) + '\' := ' + str(LIDAR_MAX_DISTANCE),\
                                         'angle\' := y4 + ' + str(nextAngle),\
                                         'temp1\' := y4 + ' + str(nextAngle) + ' - theta_l',\
                                         'temp2\' := y4 + ' + str(nextAngle)+ ' - theta_r']
    
    plant[mode1_reg2 + index + 9]['transitions'][(mode1_reg2 + index + 9,mode1_reg2 + index + 10)]['guards2'] =\
                                        ['clock = 0', 'f' + str((index + 10)/10) + ' <= ' + str(LIDAR_MAX_DISTANCE)]
    plant[mode1_reg2 + index + 9]['transitions'][(mode1_reg2 + index + 9,mode1_reg2 + index + 10)]['reset2'] =\
                                        ['angle\' := y4 + ' + str(nextAngle),\
                                         'temp1\' := y4 + ' + str(nextAngle) + ' - theta_l',\
                                         'temp2\' := y4 + ' + str(nextAngle)+ ' - theta_r']

    nextAngle += LIDAR_OFFSET
    index += 10
    curRay += 1

#Region 3
nextAngle = -LIDAR_RANGE + LIDAR_OFFSET
index = 0
curRay = 1

#while nextAngle <= LIDAR_RANGE + LIDAR_OFFSET:
while curRay <= NUM_RAYS:

    # first mode
    plant[mode1_reg3 + index] = {}
    plant[mode1_reg3 + index]['name'] = ''
    plant[mode1_reg3 + index]['odetype'] = 'lti ode'
    plant[mode1_reg3 + index]['dynamics'] = {}
    plant[mode1_reg3 + index]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg3 + index]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg3 + index]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg3 + index]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg3 + index]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg3 + index]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg3 + index]['dynamics']['angle'] = 'angle\' = 0\n'
    plant[mode1_reg3 + index]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg3 + index]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg3 + index]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg3 + index]['dynamics']['temp2'] = 'temp2\' = 0\n'        
    for i in range(NUM_RAYS):
        plant[mode1_reg3 + index]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
        
    plant[mode1_reg3 + index]['invariants'] = ['clock <= 0']
    plant[mode1_reg3 + index]['transitions'] = {}
    
    #self transitions to convert angle to (-180, 180), i.e., (-pi, pi)
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index)] = {}
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index)]['guards1'] =\
                                        ['clock = 0', 'angle >= ' + str(np.pi)]
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index)]['reset1'] =\
                                        ['angle\' := angle - ' + str(2 * np.pi),\
                                         'temp1\' := temp1 - ' + str(2 * np.pi),\
                                         'temp2\' := temp2 - ' + str(2 * np.pi)]
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index)]['guards2'] =\
                                        ['clock = 0', 'angle <= ' + str(-np.pi)]
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index)]['reset2'] =\
                                        ['angle\' := angle + ' + str(2 * np.pi),\
                                         'temp1\' := temp1 + ' + str(2 * np.pi),\
                                         'temp2\' := temp2 + ' + str(2 * np.pi)]
    
    #transition to correct wall

    # [-LIDAR_RANGE, -90.6)
    
    # NB: the hardcoded numbers -90.6 and -89.4 mean that those rays
    # cannot possibly hit the wall within 10m and for 0.15m clearance;
    # tighter bounds can be obtained for 5m LIDAR range and 0.3m
    # clearance
    
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index + 1)] = {}
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index + 1)]['guards1'] =\
                                        ['clock = 0', 'angle >= ' + str(-np.pi),\
                                         'angle <= ' + str(np.pi), 'angle <= ' + str(-90.6 * np.pi / 180)]
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index + 1)]['reset1'] =\
                                        ['clock\' := 0', 'f' + str((index + 10)/10) + '\' := 0',\
                                         'angle\' := ' + str(np.pi) + ' + angle']

    # [-90.6, -89.4]
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index + 9)] = {}
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index + 9)]['guards1'] =\
                                        ['clock = 0', 'angle >= ' + str(-90.6 * np.pi / 180),\
                                         'angle <= ' + str(-89.4 * np.pi / 180)]
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index + 9)]['reset1'] =\
                                        ['clock\' := 0',\
                                         'f' + str((index + 10)/10) + '\' := ' + str(LIDAR_MAX_DISTANCE + 1),\
                                         'angle\' := angle']

    # (-89.4, theta_l]
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index + 2)] = {}
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index + 2)]['guards1'] =\
                                        ['clock = 0', 'angle >= ' + str(-np.pi), 'angle <= ' + str(np.pi),\
                                         'temp1 <= 0.01', 'angle >= ' + str(-89.4 * np.pi / 180)]
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index + 2)]['reset1'] =\
                                        ['clock\' := 0', 'f' + str((index + 10)/10) + '\' := 0',\
                                         'angle\' := angle']

    # (theta_l, theta_r]
    #NB: some epsilons here
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index + 3)] = {}
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index + 3)]['guards1'] =\
                                        ['clock = 0', 'angle >= ' + str(-np.pi),\
                                         'angle <= ' + str(np.pi), 'temp1 >= 0.01', 'temp2 <= 0.01']
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index + 3)]['reset1'] =\
                                        ['clock\' := 0', 'f' + str((index + 10)/10) + '\' := 0',\
                                         'angle\' := ' + str(np.pi/2) + ' - angle']

    # (theta_r, LIDAR_RANGE]
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index + 4)] = {}
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index + 4)]['guards1'] =\
                                        ['clock = 0', 'angle >= ' + str(-np.pi),\
                                         'angle <= ' + str(np.pi), 'temp2 >= 0.01']
    plant[mode1_reg3 + index]['transitions'][(mode1_reg3 + index,mode1_reg3 + index + 4)]['reset1'] =\
                                        ['clock\' := 0', 'f' + str((index + 10)/10) + '\' := 0',\
                                         'angle\' := ' + str(np.pi) + ' - angle']

    # compute cos(angle) [-LIDAR_RANGE, -90) 
    plant[mode1_reg3 + index + 1] = {}
    plant[mode1_reg3 + index + 1]['name'] = ''
    plant[mode1_reg3 + index + 1]['odetype'] = 'lti ode'
    plant[mode1_reg3 + index + 1]['dynamics'] = {}
    plant[mode1_reg3 + index + 1]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg3 + index + 1]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg3 + index + 1]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg3 + index + 1]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg3 + index + 1]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg3 + index + 1]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg3 + index + 1]['dynamics']['angle'] = 'angle\' = 0\n'
    plant[mode1_reg3 + index + 1]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg3 + index + 1]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg3 + index + 1]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg3 + index + 1]['dynamics']['temp2'] = 'temp2\' = 0\n'            
    for i in range(NUM_RAYS):
        plant[mode1_reg3 + index + 1]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'

    plant[mode1_reg3 + index + 1]['invariants'] = ['clock <= 0']
    plant[mode1_reg3 + index + 1]['transitions'] = {}
    plant[mode1_reg3 + index + 1]['transitions'][(mode1_reg3 + index + 1,mode1_reg3 + index + 5)] = {}
    plant[mode1_reg3 + index + 1]['transitions'][(mode1_reg3 + index + 1,mode1_reg3 + index + 5)]['guards1'] = ['clock = 0']
    plant[mode1_reg3 + index + 1]['transitions'][(mode1_reg3 + index + 1,mode1_reg3 + index + 5)]['reset1'] = ['clock\' := 0']
    
    # 
    plant[mode1_reg3 + index + 5] = {}
    plant[mode1_reg3 + index + 5]['name'] = 'sec_0_0_'
    plant[mode1_reg3 + index + 5]['odetype'] = 'lti ode'
    plant[mode1_reg3 + index + 5]['dynamics'] = {}
    plant[mode1_reg3 + index + 5]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg3 + index + 5]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg3 + index + 5]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg3 + index + 5]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg3 + index + 5]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg3 + index + 5]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg3 + index + 5]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg3 + index + 5]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg3 + index + 5]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg3 + index + 5]['dynamics']['temp2'] = 'temp2\' = 0\n'            
    plant[mode1_reg3 + index + 5]['dynamics']['angle'] = 'angle\' = 0\n'
    for i in range(NUM_RAYS):
        plant[mode1_reg3 + index + 5]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
        
    plant[mode1_reg3 + index + 5]['invariants'] = ['clock <= 0']
    plant[mode1_reg3 + index + 5]['transitions'] = {}
    plant[mode1_reg3 + index + 5]['transitions'][(mode1_reg3 + index + 5,mode1_reg3 + index + 9)] = {}
    plant[mode1_reg3 + index + 5]['transitions'][(mode1_reg3 + index + 5,mode1_reg3 + index + 9)]['guards1'] = ['clock = 0']
    plant[mode1_reg3 + index + 5]['transitions'][(mode1_reg3 + index + 5,mode1_reg3 + index + 9)]['reset1'] =\
                                    ['clock\' := 0', 'f' + str((index + 10)/10) + '\' := (' + str(HALLWAY_WIDTH) + ' - y2) * angle']

    # compute cos(angle) (-90, theta_l]
    plant[mode1_reg3 + index + 2] = {}
    plant[mode1_reg3 + index + 2]['name'] = ''
    plant[mode1_reg3 + index + 2]['odetype'] = 'lti ode'
    plant[mode1_reg3 + index + 2]['dynamics'] = {}
    plant[mode1_reg3 + index + 2]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg3 + index + 2]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg3 + index + 2]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg3 + index + 2]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg3 + index + 2]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg3 + index + 2]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg3 + index + 2]['dynamics']['angle'] = 'angle\' = 0\n'
    plant[mode1_reg3 + index + 2]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg3 + index + 2]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg3 + index + 2]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg3 + index + 2]['dynamics']['temp2'] = 'temp2\' = 0\n'            
    for i in range(NUM_RAYS):
        plant[mode1_reg3 + index + 2]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'

    plant[mode1_reg3 + index + 2]['invariants'] = ['clock <= 0']
    plant[mode1_reg3 + index + 2]['transitions'] = {}
    plant[mode1_reg3 + index + 2]['transitions'][(mode1_reg3 + index + 2,mode1_reg3 + index + 6)] = {}
    plant[mode1_reg3 + index + 2]['transitions'][(mode1_reg3 + index + 2,mode1_reg3 + index + 6)]['guards1'] = ['clock = 0']
    plant[mode1_reg3 + index + 2]['transitions'][(mode1_reg3 + index + 2,mode1_reg3 + index + 6)]['reset1'] = ['clock\' := 0']    

    # 
    plant[mode1_reg3 + index + 6] = {}
    plant[mode1_reg3 + index + 6]['name'] = 'sec_0_0_'
    plant[mode1_reg3 + index + 6]['odetype'] = 'lti ode'
    plant[mode1_reg3 + index + 6]['dynamics'] = {}
    plant[mode1_reg3 + index + 6]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg3 + index + 6]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg3 + index + 6]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg3 + index + 6]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg3 + index + 6]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg3 + index + 6]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg3 + index + 6]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg3 + index + 6]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg3 + index + 6]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg3 + index + 6]['dynamics']['temp2'] = 'temp2\' = 0\n'            
    plant[mode1_reg3 + index + 6]['dynamics']['angle'] = 'angle\' = 0\n'
    for i in range(NUM_RAYS):
        plant[mode1_reg3 + index + 6]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
        
    plant[mode1_reg3 + index + 6]['invariants'] = ['clock <= 0']
    plant[mode1_reg3 + index + 6]['transitions'] = {}
    plant[mode1_reg3 + index + 6]['transitions'][(mode1_reg3 + index + 6,mode1_reg3 + index + 9)] = {}
    plant[mode1_reg3 + index + 6]['transitions'][(mode1_reg3 + index + 6,mode1_reg3 + index + 9)]['guards1'] = ['clock = 0']
    plant[mode1_reg3 + index + 6]['transitions'][(mode1_reg3 + index + 6,mode1_reg3 + index + 9)]['reset1'] =\
                                            ['clock\' := 0', 'f' + str((index + 10)/10) + '\' := y2 * angle']

    # compute cos(angle) (theta_l, theta_r]
    plant[mode1_reg3 + index + 3] = {}
    plant[mode1_reg3 + index + 3]['name'] = ''
    plant[mode1_reg3 + index + 3]['odetype'] = 'lti ode'
    plant[mode1_reg3 + index + 3]['dynamics'] = {}
    plant[mode1_reg3 + index + 3]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg3 + index + 3]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg3 + index + 3]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg3 + index + 3]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg3 + index + 3]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg3 + index + 3]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg3 + index + 3]['dynamics']['angle'] = 'angle\' = 0\n'
    plant[mode1_reg3 + index + 3]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg3 + index + 3]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg3 + index + 3]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg3 + index + 3]['dynamics']['temp2'] = 'temp2\' = 0\n'            
    for i in range(NUM_RAYS):
        plant[mode1_reg3 + index + 3]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'

    plant[mode1_reg3 + index + 3]['invariants'] = ['clock <= 0']
    plant[mode1_reg3 + index + 3]['transitions'] = {}
    plant[mode1_reg3 + index + 3]['transitions'][(mode1_reg3 + index + 3,mode1_reg3 + index + 7)] = {}
    plant[mode1_reg3 + index + 3]['transitions'][(mode1_reg3 + index + 3,mode1_reg3 + index + 7)]['guards1'] = ['clock = 0']
    plant[mode1_reg3 + index + 3]['transitions'][(mode1_reg3 + index + 3,mode1_reg3 + index + 7)]['reset1'] = ['clock\' := 0']

    # 
    plant[mode1_reg3 + index + 7] = {}
    plant[mode1_reg3 + index + 7]['name'] = 'sec_0_0_'
    plant[mode1_reg3 + index + 7]['odetype'] = 'lti ode'
    plant[mode1_reg3 + index + 7]['dynamics'] = {}
    plant[mode1_reg3 + index + 7]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg3 + index + 7]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg3 + index + 7]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg3 + index + 7]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg3 + index + 7]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg3 + index + 7]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg3 + index + 7]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg3 + index + 7]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg3 + index + 7]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg3 + index + 7]['dynamics']['temp2'] = 'temp2\' = 0\n'            
    plant[mode1_reg3 + index + 7]['dynamics']['angle'] = 'angle\' = 0\n'
    for i in range(NUM_RAYS):
        plant[mode1_reg3 + index + 7]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
        
    plant[mode1_reg3 + index + 7]['invariants'] = ['clock <= 0']
    plant[mode1_reg3 + index + 7]['transitions'] = {}
    plant[mode1_reg3 + index + 7]['transitions'][(mode1_reg3 + index + 7,mode1_reg3 + index + 9)] = {}
    plant[mode1_reg3 + index + 7]['transitions'][(mode1_reg3 + index + 7,mode1_reg3 + index + 9)]['guards1'] = ['clock = 0']
    plant[mode1_reg3 + index + 7]['transitions'][(mode1_reg3 + index + 7,mode1_reg3 + index + 9)]['reset1'] =\
                                                ['clock\' := 0', 'f' + str((index + 10)/10) + '\' := y1 * angle']

    # compute cos(angle) (theta_r, LIDAR_RANGE)
    plant[mode1_reg3 + index + 4] = {}
    plant[mode1_reg3 + index + 4]['name'] = ''
    plant[mode1_reg3 + index + 4]['odetype'] = 'lti ode'
    plant[mode1_reg3 + index + 4]['dynamics'] = {}
    plant[mode1_reg3 + index + 4]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg3 + index + 4]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg3 + index + 4]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg3 + index + 4]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg3 + index + 4]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg3 + index + 4]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg3 + index + 4]['dynamics']['angle'] = 'angle\' = 0\n'
    plant[mode1_reg3 + index + 4]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg3 + index + 4]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg3 + index + 4]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg3 + index + 4]['dynamics']['temp2'] = 'temp2\' = 0\n'            
    for i in range(NUM_RAYS):
        plant[mode1_reg3 + index + 4]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'

    plant[mode1_reg3 + index + 4]['invariants'] = ['clock <= 0']
    plant[mode1_reg3 + index + 4]['transitions'] = {}
    plant[mode1_reg3 + index + 4]['transitions'][(mode1_reg3 + index + 4,mode1_reg3 + index + 8)] = {}
    plant[mode1_reg3 + index + 4]['transitions'][(mode1_reg3 + index + 4,mode1_reg3 + index + 8)]['guards1'] = ['clock = 0']
    plant[mode1_reg3 + index + 4]['transitions'][(mode1_reg3 + index + 4,mode1_reg3 + index + 8)]['reset1'] = ['clock\' := 0']
    
    # 
    plant[mode1_reg3 + index + 8] = {}
    plant[mode1_reg3 + index + 8]['name'] = 'sec_0_0_'
    plant[mode1_reg3 + index + 8]['odetype'] = 'lti ode'
    plant[mode1_reg3 + index + 8]['dynamics'] = {}
    plant[mode1_reg3 + index + 8]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg3 + index + 8]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg3 + index + 8]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg3 + index + 8]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg3 + index + 8]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg3 + index + 8]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg3 + index + 8]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg3 + index + 8]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg3 + index + 8]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg3 + index + 8]['dynamics']['temp2'] = 'temp2\' = 0\n'            
    plant[mode1_reg3 + index + 8]['dynamics']['angle'] = 'angle\' = 0\n'
    for i in range(NUM_RAYS):
        plant[mode1_reg3 + index + 8]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
        
    plant[mode1_reg3 + index + 8]['invariants'] = ['clock <= 0']
    plant[mode1_reg3 + index + 8]['transitions'] = {}
    plant[mode1_reg3 + index + 8]['transitions'][(mode1_reg3 + index + 8,mode1_reg3 + index + 9)] = {}
    plant[mode1_reg3 + index + 8]['transitions'][(mode1_reg3 + index + 8,mode1_reg3 + index + 9)]['guards1'] = ['clock = 0']
    plant[mode1_reg3 + index + 8]['transitions'][(mode1_reg3 + index + 8,mode1_reg3 + index + 9)]['reset1'] =\
                                ['clock\' := 0', 'f' + str((index + 10)/10) + '\' := (' + str(HALLWAY_WIDTH) + ' - y2) * angle']    

    # last mode
    plant[mode1_reg3 + index + 9] = {}
    plant[mode1_reg3 + index + 9]['name'] = ''
    plant[mode1_reg3 + index + 9]['odetype'] = 'lti ode'
    plant[mode1_reg3 + index + 9]['dynamics'] = {}
    plant[mode1_reg3 + index + 9]['dynamics']['y1'] = 'y1\' = 0\n'
    plant[mode1_reg3 + index + 9]['dynamics']['y2'] = 'y2\' = 0\n'
    plant[mode1_reg3 + index + 9]['dynamics']['y3'] = 'y3\' = 0\n'
    plant[mode1_reg3 + index + 9]['dynamics']['y4'] = 'y4\' = 0\n'
    plant[mode1_reg3 + index + 9]['dynamics']['k'] = 'k\' = 0\n'
    plant[mode1_reg3 + index + 9]['dynamics']['u'] = 'u\' = 0\n'
    plant[mode1_reg3 + index + 9]['dynamics']['angle'] = 'angle\' = 0\n'
    plant[mode1_reg3 + index + 9]['dynamics']['theta_l'] = 'theta_l\' = 0\n'
    plant[mode1_reg3 + index + 9]['dynamics']['theta_r'] = 'theta_r\' = 0\n'
    plant[mode1_reg3 + index + 9]['dynamics']['temp1'] = 'temp1\' = 0\n'
    plant[mode1_reg3 + index + 9]['dynamics']['temp2'] = 'temp2\' = 0\n'            
    for i in range(NUM_RAYS):
        plant[mode1_reg3 + index + 9]['dynamics']['f' + str(i + 1)] = 'f' + str(i + 1) + '\' = 0\n'
        
    plant[mode1_reg3 + index + 9]['invariants'] = ['clock <= 0']
    plant[mode1_reg3 + index + 9]['transitions'] = {}

    # if last ray, need to transition to m0 (in composed transitions)
    #if nextAngle == LIDAR_RANGE + LIDAR_OFFSET:
    if curRay == NUM_RAYS:
        break
    
    plant[mode1_reg3 + index + 9]['transitions'][(mode1_reg3 + index + 9,mode1_reg3 + index + 10)] = {}
    
    plant[mode1_reg3 + index + 9]['transitions'][(mode1_reg3 + index + 9,mode1_reg3 + index + 10)]['guards1'] =\
                                            ['clock = 0', 'f' + str((index + 10)/10) + ' >= ' + str(LIDAR_MAX_DISTANCE)]
    plant[mode1_reg3 + index + 9]['transitions'][(mode1_reg3 + index + 9,mode1_reg3 + index + 10)]['reset1'] =\
                                            ['f' + str((index + 10)/10) + '\' := ' + str(LIDAR_MAX_DISTANCE),\
                                             'angle\' := y4 + ' + str(nextAngle),\
                                             'temp1\' := y4 + ' + str(nextAngle) + ' - theta_l',\
                                             'temp2\' := y4 + ' + str(nextAngle) + ' - theta_r']
    
    plant[mode1_reg3 + index + 9]['transitions'][(mode1_reg3 + index + 9,mode1_reg3 + index + 10)]['guards2'] =\
                                        ['clock = 0', 'f' + str((index + 10)/10) + ' <= ' + str(LIDAR_MAX_DISTANCE)]
    plant[mode1_reg3 + index + 9]['transitions'][(mode1_reg3 + index + 9,mode1_reg3 + index + 10)]['reset2'] =\
                                        ['angle\' := y4 + ' + str(nextAngle),\
                                         'temp1\' := y4 + ' + str(nextAngle) + ' - theta_l',\
                                         'temp2\' := y4 + ' + str(nextAngle) + ' - theta_r']

    nextAngle += LIDAR_OFFSET
    index += 10
    curRay += 1

filename = 'dynamics_' + str(NUM_RAYS) + '.pickle'

try:
    with open(filename, 'wb') as f:
        pickle.dump(plant, f, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', filename, ':', e)
