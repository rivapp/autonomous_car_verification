import sys
sys.path.append('../simulator')

from Car import World
import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3

def normalize(s):
    mean = [2.5]
    spread = [5.0]
    return (s - mean) / spread

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env, seed, eval_episodes=10):
    eval_env = env
    #eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False

        while not done:
            action = policy.select_action(np.array(normalize(state)))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    #print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("Evaluation over " + str(eval_episodes) + " episodes: " + str(avg_reward))
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="Pendulum-v0")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int) # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e5, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    #file_name = f"{args.policy}_{args.env}_{args.seed}"
    file_name = str(args.policy) + '_racing_' + str(args.seed)
    
    print("---------------------------------------")
    #print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    hallWidths = [1.5, 1.5, 1.5, 1.5]
    hallLengths = [20, 20, 20, 20]
    turns = ['right', 'right', 'right', 'right']
    car_dist_s = hallWidths[0]/2.0 - 0.1
    car_dist_f = 9.9
    car_heading = 0
    episode_length = 130
    time_step = 0.1
    time = 0

    lidar_field_of_view = 115
    lidar_num_rays = 21
    lidar_noise = 0.1 #m
    missing_lidar_rays = 0
    
    env = World(hallWidths, hallLengths, turns,\
                car_dist_s, car_dist_f, car_heading,\
                episode_length, time_step, lidar_field_of_view,\
                lidar_num_rays, lidar_noise, missing_lidar_rays)

    test_env = World(hallWidths, hallLengths, turns,\
                     car_dist_s, car_dist_f, car_heading,\
                     episode_length, time_step, lidar_field_of_view,\
                     lidar_num_rays, lidar_noise, missing_lidar_rays)    
        

    #env = gym.make(args.env)

    # Set seeds
    #env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        #policy.load(f"./models/{policy_file}")
        policy.load("./models/'" + str(policy_file))

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, test_env, args.seed)]

    state, done = env.reset(), False
    state = normalize(state)
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        next_state = normalize(next_state)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            #print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            print("Total T: " + str(t+1) + " Episode Num: " + str(episode_num+1) +
                  " Episode T: " + str(episode_timesteps) +" Reward: " + str(episode_reward))
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, test_env, args.seed))
            np.save("./results/" + str(file_name), evaluations)
            policy.save("./models/tanh_" +\
                        str(env.observation_space.shape[0]) + "_m" + str(missing_lidar_rays) + '_')
            