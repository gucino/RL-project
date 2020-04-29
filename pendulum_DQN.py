# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:56:15 2020

@author: Tisana
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:23:54 2020

@author: Tisana
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:52:02 2020

@author: Tisana
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:30:19 2020

@author: Tisana
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:43:37 2020

@author: Tisana
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:41:50 2020

@author: Tisana
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:16:00 2020

@author: Tisana
"""

import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

######################################
######################################
######################################


#parameter     
env = gym.make('Pendulum-v0')
learning_rate=0.001
num_sample_to_draw=32
discount_factor=0.99
num_episode=2000
replay_sample_capacity=20000
c=1
epsilon=1
epsilon_decay=0.995
epsilon_min=0.01
num_input_Q=env.observation_space.shape[0]
num_hidden1_Q=40
num_hidden2_Q=40
num_output_Q=5
to_render=5000
######################################
######################################
######################################
#initialise replay buffer
replay_buffer=deque(maxlen=replay_sample_capacity)
######################################
######################################
######################################
#initialise Q network
Q_network=Sequential()
  
   
Q_network.add(Dense(num_hidden1_Q,input_shape=(num_input_Q,),activation="relu"))  
Q_network.add(Dense(num_hidden2_Q,activation="relu"))
Q_network.add(Dense(num_output_Q,activation='linear'))

#learning process
Q_network.compile(loss="mse", optimizer=Adam(learning_rate))

######################################
######################################
######################################
#initialise target nwtwork
Q_target_network=Sequential()
   
Q_target_network.add(Dense(num_hidden1_Q,input_shape=(num_input_Q,),activation="relu"))   
Q_target_network.add(Dense(num_hidden2_Q,activation="relu"))
Q_target_network.add(Dense(num_output_Q,activation='linear'))

#learning process
Q_target_network.compile(loss="mse", optimizer=Adam(learning_rate))

#make target network have same weight as Q network
Q_target_network.set_weights(Q_network.get_weights())
######################################
######################################
######################################  
#random agent
'''
for j in range(0,10):
    c=env.reset()
    for i in range(1000):
        if i%100==0:
            print(i)
        env.render()
        action=env.action_space.sample()
        
        next_state,reward,done,_=env.step(action) # take a random action
     

'''
######################################
######################################
######################################  
reward_list=[]
for each_episode in range(0,num_episode):
    current_state=env.reset()
    current_state=current_state[np.newaxis,:]
    print(each_episode)
    reward_per_step=[]
    done=False
    #for each time step
    while done==False:
        if each_episode%to_render==0:
            env.render()
        #choose action to take
        rand=np.random.random()
        if rand<0:
            #take radnom action
            action_to_take=[np.random.uniform(-2,2)]
            #action_to_take=[np.random.choice(np.arange(0,num_output_Q))-2]
        else:
            Q_value=Q_network.predict(current_state)
            action_to_take=[np.argmax(Q_value[0])]  
    
  
    
        #observe new state and reward
        next_state,reward,done,_=env.step(action_to_take) 
        next_state=next_state.reshape(1,3)
        
        #modify reward function
        reward= 25*np.exp(-1*(next_state[0,0]-1)*(next_state[0,0]-1)/0.001)-100*np.abs(10*0.5 - (10*0.5*next_state[0,0] + 0.5*0.3333*next_state[0,2] * next_state[0,2])) + 100*np.abs(10*0.5 - (10*0.5*current_state[0,0] + 0.5*0.3333*current_state[0,2] * current_state[0,2]))#append experience
        reward_per_step.append(reward)
        replay_buffer.append([current_state,action_to_take,reward,next_state,done])
        
        #udpate state
        current_state=next_state
        
        #learning process
        if len(replay_buffer)>=num_sample_to_draw:
    
            #sampling from replay buffer
            sample_current_state_list=[]
            sample_next_state_list=[]
            sample_list=random.sample(replay_buffer,num_sample_to_draw)
            
            for each_sample in sample_list:
                sample_current_state,sample_action,sample_reward,\
                sample_next_state,sample_terminate=each_sample
                
                sample_current_state_list.append(sample_current_state)
                sample_next_state_list.append(sample_next_state) 
            sample_current_state_list=np.array(sample_current_state_list).reshape(num_sample_to_draw, num_input_Q)
            sample_next_state_list=np.array(sample_next_state_list).reshape(num_sample_to_draw, num_input_Q)
                
            Q_next_list=Q_target_network .predict(np.array(sample_next_state_list))
            Q_value_list=Q_network.predict(np.array(sample_current_state_list))   
            
            #adjucted Q Q value list respect to target
            Q_value_list_2=[]
            for i in range(0,len(sample_list)):
                sample_current_state,sample_action,sample_reward,\
                sample_next_state,sample_terminate=sample_list[i]  
                
                Q_target=sample_reward  
                if sample_terminate==False:
                    Q_next=Q_next_list[i]
                    #Q_target=(sample_reward+(discount_factor*Q_next))
                    Q_target=(sample_reward+(discount_factor*np.amax(Q_next)))
            
                Q_value=Q_value_list[i]
                #adjust Q value output respect to target value
                Q_value[sample_action]=Q_target
                Q_value_list_2.append(Q_value)
            #learn in parallel
            Q_network.fit(np.array(sample_current_state_list),\
                               np.array(Q_value_list_2),verbose=0)  

    #decay epsilon
    epsilon*=epsilon_decay
    epsilon=max(epsilon,epsilon_min)

    #set Q target to Q
    if each_episode%c==0:
        Q_target_network.set_weights(Q_network.get_weights())           

    print(" ",sum(reward_per_step))
    sum_reward=sum(reward_per_step)  
    reward_list.append(sum_reward)

    

#plot
def smmooth(data,num_interval):
    smooth_data=[]
    increment=len(data)/num_interval
    start=0
    for each_interval in range(0,num_interval):
        
        end=start+increment
        smooth_data.append(np.mean(data[int(start):int(end)]))
        start+=increment
    return np.array(smooth_data)   

#plot
plt.figure()
plt.title("reward vs episode (pendulum)")
plt.plot(np.arange(0,len(reward_list)),reward_list)
plt.xlabel("num episode")
plt.ylabel("reward")
plt.show()
plt.savefig("reward vs episode (pendulum)")

num_interval=200
smooth=smmooth(reward_list,num_interval)
plt.figure()
plt.title("reward vs episode (pendulum) (smoothed)")
plt.plot(np.arange(0,len(smooth)),smooth)
plt.xlabel("num episode")
plt.ylabel("reward")
plt.show()
plt.savefig("reward vs episode (pendulum) (smoothed)")




