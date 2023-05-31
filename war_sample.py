import gymnasium as gym
import numpy as np
env = gym.make("FrozenLake-v1",render_mode="ansi",desc=["SFFF", "FFFF", "FFFF", "FFFF"],is_slippery=False)
from scipy.stats import entropy
import random

def get_samples_war(env,h,n,rand_start=False):
    S = env.observation_space.n
    s = np.zeros((n,S))
    A = env.action_space.n
    a= np.zeros((n,A))
    F = np.zeros(n)
    s_a = np.zeros((n,S*A))
    # Initialize an empty list to store the trajectory
    trajectory = [0]*h
    trajectories = [trajectory]*n
    scores = [0]*n
    #comment
    # Loop over the steps

    for j in range(n):
        temp_trajectory = [0]*h
        env.reset()
        s[j,1]+=1
        action_1 = env.action_space.sample()
        s_a[j,0+action_1]+=1
        for i in range(h):
        # Choose a random action
            if(action_1 == None):
                #action = env.action_space.sample()
                action = action
                #action = 0
                #action = random.randint(0,A-1)
            else:
                action = action_1
                action_1 = None
            a[j,action]+=1
            # Take the action and get the next state, reward, done flag, and info dictionary
            next_state, reward, done, info,extra = env.step(action)
            action = env.action_space.sample()
            s[j,next_state]+=1
            s_a[j,next_state*A+action]+=1
            
            # Append the (state, reward) tuple to the trajectory list
            temp_trajectory.append((env.s, action, reward))
            
            # Update the state
            env.s = next_state
            
            # Render the environment
            env.render()
            
            # Check if the episode is done
            if done:
                break

        a[j,] = a[j,]/np.sum(a[j,])
        s[j,] = s[j,]/np.sum(s[j,])
        s_a[j,] = s_a[j]/np.sum(s_a[j,])
        #print("s[j,0]")
        #print(s[j,0])
        #print("s[j,]")
        #print(s[j,])
        score = -np.sqrt(s[j,0]) - np.sqrt(s[j,2])
        scores[j] = 10*score

        trajectories[j] = temp_trajectory
    return (np.array(s_a),np.array(scores))