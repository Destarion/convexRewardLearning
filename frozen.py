import gymnasium as gym
import numpy as np
env = gym.make("FrozenLake-v1",render_mode="ansi")
from scipy.stats import entropy

# Define the number of steps to take
h = 10
n = 10
S = env.observation_space.n
s = np.zeros((n,S))
A = env.observation_space.n
a= np.zeros((n,A))
F = np.zeros(n)
s_a = np.zeros((n,S*A))
# Initialize an empty list to store the trajectory
trajectory = [0]*h
trajectories = [trajectory]*n
entropies = [0]*n

# Loop over the steps

for j in range(n):
    temp_trajectory = [0]*h
    env.reset()
    emp_h = h+1
    s[j,0]+=1
    action_1 = env.action_space.sample()
    s_a[j,0+action_1]+=1
    for i in range(h):
        print("i")
        print(i)
    # Choose a random action
        if(action_1 == None):
            action = env.action_space.sample()
        else:
            action = action_1
        a[j,action]+=1
        # Take the action and get the next state, reward, done flag, and info dictionary
        next_state, reward, done, info,extra = env.step(action)
        s[j,next_state]+=1
        s_a[j,next_state*A+action_1]+=1
        
        # Append the (state, action, reward) tuple to the trajectory list
        temp_trajectory.append((env.s, action, reward))
        
        # Update the state
        env.s = next_state
        
        # Render the environment
        env.render()
        
        # Check if the episode is done
        if done:
            emp_h = i+1
            break

    a[j,] = a[j,]/emp_h
    s[j,] = s[j,]/emp_h
    s_a[j,] = s_a[j]/emp_h
    ent = entropy(s_a[j,])
    entropies[j] = ent

    trajectories[j] = temp_trajectory
# Print the trajectory
print(entropies[n-1])