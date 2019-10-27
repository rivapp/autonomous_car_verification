from Car import World
import numpy as np
import random
from keras import models
import matplotlib.pyplot as plt
import sys

def relu(x):
    relu = np.maximum(0, x)

    return relu

#this is just for testing purposes
def relu_predict(model, inputs):
    weights = {}
    offsets = {}

    layerCount = 1

    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            weights[layerCount] = layer.get_weights()[0]
            offsets[layerCount] = layer.get_weights()[1]

            layerCount += 1

    curNeurons = inputs

    for layer in range(layerCount-1):
        curNeurons = curNeurons.dot(weights[layer + 1]) + offsets[layer + 1]

        if layer <= layerCount - 3:
            curNeurons = relu(curNeurons)

    return curNeurons

#this is just for testing purposes
def tanh_predict(model, inputs):
    weights = {}
    offsets = {}

    layerCount = 1

    for layer in model.layers:
        
        if len(layer.get_weights()) > 0:
            weights[layerCount] = layer.get_weights()[0]
            offsets[layerCount] = layer.get_weights()[1]

            layerCount += 1

    curNeurons = inputs

    for layer in range(layerCount-1):
        curNeurons = curNeurons.dot(weights[layer + 1]) + offsets[layer + 1]

        curNeurons = np.tanh(curNeurons)

    return curNeurons

def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))

    return sigm

def swish_predict(model, inputs):
    weights = {}
    offsets = {}

    layerCount = 1

    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            weights[layerCount] = layer.get_weights()[0]
            offsets[layerCount] = layer.get_weights()[1]

            layerCount += 1

    curNeurons = inputs

    for layer in range(layerCount-1):
        curNeurons = curNeurons.dot(weights[layer + 1]) + offsets[layer + 1]

        if layer <= layerCount - 3:
            curNeurons = curNeurons * sigmoid(curNeurons)
            #curNeurons = relu(curNeurons)

    return curNeurons

def normalize(s):
    mean = [2.5]
    spread = [5.0]
    return (s - mean) / spread

def main(argv):

    input_filename = argv[0]
    
    model = models.load_model(input_filename)

    numTrajectories = 100

    numHalls = 4
    hallWidth = 1.5
    hallLength = 20

    turns = ['right', 'right', 'right', 'right']
    car_dist_s = hallWidth/2.0
    car_dist_f = 9.9
    car_heading = 0
    episode_length = 70
    time_step = 0.1

    lidar_field_of_view = 115
    lidar_num_rays = model.get_layer(index=0).input_shape[1]
    lidar_noise = 0.2
    missing_lidar_rays = 5

    num_unsafe = 0

    w = World(numHalls, hallWidth, hallLength, turns,\
              car_dist_s, car_dist_f, car_heading,\
              episode_length, time_step, lidar_field_of_view,\
              lidar_num_rays, lidar_noise, missing_lidar_rays, True)

    throttle = 16
    
    allX = []
    allY = []
    allR = []

    for step in range(numTrajectories):

        w.reset()

        observation = w.scan_lidar()

        rew = 0

        for e in range(episode_length):

            observation = normalize(observation)

            delta = 15 * model.predict(observation.reshape(1,len(observation)))

            observation, reward, done, info = w.step(delta, throttle)

            if done:

                if e < episode_length - 1:
                    num_unsafe += 1
                
                break

            rew += reward

        allX.append(w.allX)
        allY.append(w.allY)
        allR.append(rew)


    print(np.mean(allR))
    print('number of crashes: ' + str(num_unsafe))
    
    fig = plt.figure(figsize=(12,10))
    w.plotHalls()
    
    plt.ylim((-1,11))
    plt.xlim((-1.75,10.25))
    plt.tick_params(labelsize=20)

    for i in range(numTrajectories):
        plt.plot(allX[i], allY[i], 'r-')

    plt.show()
    
if __name__ == '__main__':
    main(sys.argv[1:])
