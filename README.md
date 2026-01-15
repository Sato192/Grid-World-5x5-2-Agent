Using Central Q-learning (CQL) 
In a grid World 5x5 , two agents should arrive at the goal at the same time , action space is  {up,dow,left,right,stay} . each step without arriving at the goal have the reward of 0 , if both arrive at the goal at the same time both get the reward of [1,1] . If one agent arrive at the goal without the other agent arriving as will , the episode will be terminated and both will get 0 reward

note : The agents don't communicate with each other 

