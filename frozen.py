import gymnasium as gym
import numpy as np
env = gym.make("FrozenLake-v1",render_mode="human",desc=["SFFF", "FFFF", "FFFF", "FFFF"],is_slippery=False)
from scipy.stats import entropy
import random

from gymnasium.envs.toy_text.frozen_lake import generate_random_map

gSize = 4 # grid size
nH = 0.    # desired number of hole: this is probabilistic
pH = 1 - nH/(gSize*gSize) # probability of  grid being a hole

def runif_in_simplex(n):
  ''' Return uniformly random vector in the n-simplex '''

  k = np.random.exponential(scale=1.0, size=n)
  return k / sum(k)
desc = generate_random_map(size= gSize, p= pH)
#desc[-1] = "FFFF"
#print(desc)
def choose_start(k,l):
    arr = []
    for i in range(4):
        if(i == k):
            str = ""
            for j in range(4):
                if(j == l):
                    str = str + "S"
                else:
                    str = str + "F"
            arr.append(str)
        else:
            arr.append("FFFF")
    return arr

def get_samples(env,h,n,rand_start=False):
    S = env.observation_space.n
    A = env.action_space.n
    F = np.zeros(n)
    # Initialize an empty list to store the trajectory
    trajectory = [0]*h
    trajectories = [trajectory]*n
    entropies = [0]*n
    square = [0]*n
    #comment
    # Loop over the steps
    acbd = runif_in_simplex(4)
    for j in range(n):
        h = np.random.randint(1,100)
        trajectory = [0]*h
        trajectories = [trajectory]*n
        if j%20:
            acbd = runif_in_simplex(4)
        s = np.zeros((n,S))
        a= np.zeros((n,A))
        s_a = np.zeros((n,S*A))
        temp_trajectory = [0]*h
        i=0
        k=0
        if(rand_start==True):
            i = random.randint(0,3)
            k = random.randint(0,3)
            desc = choose_start(i,k)
            env = gym.make("FrozenLake-v1",render_mode="ansi",desc=desc,is_slippery=False)
        env.reset()
        s[j,0]+=1
        action_1 = np.random.choice([0,1,2,3],p=acbd)
        s_a[j,(4*i+k)*A+action_1]+=1
        for i in range(h):
        # Choose a random action
            if(action_1 == None):
                #action = env.action_space.sample()
                action = action
                #action = 0
                #action = random.randint(0,A-1)
                #print(action)
            else:
                action = action_1
                action_1 = None
            a[j,action]+=1
            # Take the action and get the next state, reward, done flag, and info dictionary
            next_state, reward, done, info,extra = env.step(action)
            action = np.random.choice([0,1,2,3],p=acbd)
            s[j,next_state]+=1
            #print("step")
            #print(next_state)
            #print(action)
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
        ent = -10*entropy(10*s_a[j,])
        square[j] = np.sum(10*(100*s_a[j,]+5)**2)
        #print(square)
        #print("s_a, entropy")
        #print(s_a[j,])
        #print(ent)
        entropies[j] = ent

        trajectories[j] = temp_trajectory
    return (np.array(s_a),np.array(square))