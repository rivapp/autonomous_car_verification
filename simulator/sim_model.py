from Car import World
import numpy as np
import random
from keras import models
import sys

def normalize(s):
    mean = [2.5]
    spread = [5.0]
    return (s - mean) / spread

def main(argv):

    input_filename = argv[0]
    
    model = models.load_model(input_filename)
    
    hallWidths = [1.5, 1.5, 1.5, 1.5]
    hallLengths = [20, 20, 20, 20]
    turns = ['right', 'right', 'right', 'right']
    car_dist_s = hallWidths[0]/2.0 
    car_dist_f = 9.9
    car_heading = 0
    episode_length = 130
    time_step = 0.1
    time = 0

    lidar_field_of_view = 115
    lidar_num_rays = model.get_layer(index=0).input_shape[1]
    lidar_noise = 0
    missing_lidar_rays = 0
    
    w = World(hallWidths, hallLengths, turns,\
              car_dist_s, car_dist_f, car_heading,\
              episode_length, time_step, lidar_field_of_view,\
              lidar_num_rays, lidar_noise, missing_lidar_rays)

    throttle = 16

    rew = 0

    observation = w.scan_lidar()
    
    for e in range(episode_length):

        observation = normalize(observation)
        
        delta = 15 * model.predict(observation.reshape(1,len(observation)))

        delta = np.clip(delta, -15, 15)
        
        observation, reward, done, info = w.step(delta, throttle)

        time += time_step
        
        rew += reward

        if done:
            break
        
    print('velocity: ' + str(w.car_V))
    print('distance from side wall: ' + str(w.car_dist_s))
    print('distance from front wall: ' + str(w.car_dist_f))
    print('heading: ' + str(w.car_heading))
    print('last control input: ' + str(delta))
    print('steps: ' + str(e))
    print('final reward: ' + str(rew))
    w.plot_trajectory()
    #w.plot_lidar()
    
if __name__ == '__main__':
    main(sys.argv[1:])
