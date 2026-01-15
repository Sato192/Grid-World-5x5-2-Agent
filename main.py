import numpy as np
from random import choice
import matplotlib.pyplot as plt


class Agent:
    def __init__(self,x,y):
        if (x > 4 or x <0 or y >4 or y<0):
            raise Exception("agent must be in 5*5 grid world")
        self.x=x
        self.y=y
        self.Q=self.initialize_states() # 25 state and 5 possibl actions (up,down,left,right,stay)
        self.reward=0
    def initialize_states(self):
                        #s1  s2  a1 a2
        Q=np.random.rand(5*5,5*5, 5,  5)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    for m in range(5):
                        if i==0 :
                            Q[5*i+j,5*k+m,1,:]=  -9999999 #this state of agent 1 don't have up action
                        if k==0 :
                            Q[5*i+j,5*k+m,:,1]=  -9999999 #this state of agent 2 don't have up action
                        if j==0:
                            Q[5*i+j,5*k+m,2,:]=  -9999999 #this state of agent 1 don't have left action
                        if m==0:
                            Q[5*i+j,5*k+m,:,2]=  -9999999 #this state of agent 2 don't have left action
                        if i==4 :
                            Q[5*i+j,5*k+m,0,:]=  -9999999 #this state of agent 1 don't have down action
                        if k==4 :
                            Q[5*i+j,5*k+m,:,0]=  -9999999 #this state of agent 2 don't have down action
                        if j==4:
                            Q[5*i+j,5*k+m,3,:]=  -9999999 #this state of agent 1 don't have right action
                        if m==4:
                            Q[5*i+j,5*k+m,:,3]=  -9999999 #this state of agent 2 don't have right action
        return Q

class GirdWorldEnv:
    
    def __init__(self,agent1,agent2):
        self.grid = np.zeros(shape=(5,5))
        self.grid[4,4]=1 #the goal 
        self.agent1=agent1
        self.agent2=agent2
                    # 0        1        2          3       4
                    # down     #up    #left      #right  #stay
                    # [[1,0]  ,[-1,0], [0,-1] ,   [0,1] , [0,0]]
        self.actions={0:[1,0],1:[-1,0],2:[0,-1],3:[0,1],4:[0,0]}
    
    def get_action_space(self,agent):
 
        posibble_actions=[]
        if agent.y > 0:
            posibble_actions.append(2) #agent can go left
        if agent.y < 4:
            posibble_actions.append(3) # agent can go right
            
        if agent.x > 0:
            posibble_actions.append(1) # agent can go down
        if agent.x < 4 :
            posibble_actions.append(0) #agent can go up
        posibble_actions.append(4)
        return posibble_actions
    
    def step(self,actions):
        tirminated=False
        x1,y1=actions[0]
        x2,y2=actions[1]
        reward=[0,0]
        #move agent 1 
        self.agent1.x+=x1
        self.agent1.y+=y1
        print(f"agent 1 at pos {agent1.x,agent1.y}")
        #move agent2
        self.agent2.x+=x2
        self.agent2.y+=y2
        print(f"agent 2 at pos {agent2.x,agent2.y}")
        if self.grid[self.agent1.x,self.agent1.y]==self.grid[self.agent2.x,self.agent2.y]==1:
            reward=[1,1]
            tirminated=True
        # elif self.grid[self.agent1.x,self.agent1.y]==1 or self.grid[self.agent2.x,self.agent2.y]==1:
        #     tirminated = True
        # else:
        #     pass
        return reward,tirminated


    def get_next_states(self,agent):
        possible_state=self.actions[self.get_action_space(agent)[0]]
        
        next_state=[agent.x + possible_state[0] , agent.y + possible_state[1]]
        
        return next_state
    
    
def QLearning(env : GirdWorldEnv,agent1,agent2,sigma=0.01,alpha=0.5,gamma=0.95,episodes=100):
    agent1.x=0
    agent1.y=0
    agent2.x=0
    agent2.y=4
    agent1.reward = 0
    agent2.reward = 0
    for i in range(episodes):
        #choosing agent 1 and agent2 actions
        if sigma > np.random.rand():
            action1_index=choice(env.get_action_space(agent1))
            action2_index =choice(env.get_action_space(agent2))
        else :                
            action_index = np.argmax( agent1.Q[agent1.x*5 + agent1.y,agent2.x*5+agent2.y])
            action1_index,action2_index = np.unravel_index(action_index,agent1.Q[agent1.x*5 + agent1.y,agent2.x*5+agent2.y].shape)
            

        action1=env.actions[action1_index]
        action2=env.actions[action2_index]
        
        #saving the curennt state before moving
        x1=agent1.x
        y1=agent1.y
        x2=agent2.x
        y2=agent2.y
        #doing the action and get the next states for both agents
        reward,terminated = env.step([action1,action2])
        agent1.reward+=reward[0]
        agent2.reward+=reward[1]
        
        
        Q_next=agent1.Q[5*agent1.x + agent1.y,5*agent2.x+agent2.y]
        

        Q_n_max1=np.max(Q_next) #maximam action value for next states for agent 1 and agent 2
        
        #updating current action value 
        
        agent1.Q[5*x1+y1,5*x2+y2,action1_index,action2_index] = agent1.Q[5*x1+y1,5*x2+y2,action1_index,action2_index] + alpha*(reward[0] + gamma*Q_n_max1 -agent1.Q[5*x1+y1,5*x2+y2,action1_index,action2_index]  )
        # agent2.Q[5*x2+y2,action2_index] = agent2.Q[5*x2+y2,action2_index] + alpha*(reward[1] + gamma*Q_n_max2 -agent2.Q[5*x2+y2,action2_index]  )
        if terminated:
            break
    return agent1.reward,agent1.reward/(i+1)

    
agent1= Agent(0,0)
agent2= Agent(0,4)



env=GirdWorldEnv(agent1,agent2)


metirc=[]
rewards=0
for i in range(5000):
    print("episode :------------------------------------",(i+1),'--------------------------------------------')
    prevQ=np.copy(agent1.Q)
    r,episode_metric=QLearning(env,agent1,agent2,sigma=1/(i+1),alpha=0.1)
    metirc.append(episode_metric)
    rewards+=r
    if (np.linalg.norm(prevQ-agent1.Q) < 10**-3 ): #if action value converge -> Stop
        break



plt.title("Q learning (reward / episodes)")
plt.plot(metirc,label="r/e")
plt.legend()
plt.show()






print(f"agent 1 reward : {rewards}/{i+1}")



