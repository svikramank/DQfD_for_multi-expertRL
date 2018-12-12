# DQfD

An implement of DQfD（Deep Q-learning from Demonstrations) raised by DeepMind:Learning from Demonstrations for Real World Reinforcement Learning

It also compared DQdD with Double DQN on the CartPole game, and the comparation shows that DQfD outperforms Double DQN.

## Comparation DQfD trained using multiple experts with Double DQN

![figure_1](/plots/newplots/plt1.png)

At the end of training, the epsilion used in greedy_action is 0.1, and thats the reason why the curves is not so stable.


## Get the expert demo data

Compared to double DQN, a improvement of DQfD is pre-training. DQfD initially trains solely on the demonstration data before starting any interaction with the environment. This code used a network fine trained by Double DQN to generate the demo data.

You can see the details in function:
```
  get_demo_data()
```

## Get Double DQN scores

For comparation, I first trained an network through Double DQN, witch has the same parameters with the DQfD.
```
    # --------------------------  get DDQN scores ----------------------------------
    ddqn_sum_scores = np.zeros(Config.episode)
    for i in range(Config.iteration):
        scores = run_DDQN(i, env)
        ddqn_sum_scores = [a + b for a, b in zip(scores, ddqn_sum_scores)]
    ddqn_mean_scores = ddqn_sum_scores / Config.iteration
```

## Get DQfD scores

```
    # ----------------------------- get DQfD scores --------------------------------
    dqfd_sum_scores = np.zeros(Config.episode)
    for i in range(Config.iteration):
        scores = run_DQfD(i, env)
        dqfd_sum_scores = [a + b for a, b in zip(scores, dqfd_sum_scores)]
    dqfd_mean_scores = dqfd_sum_scores / Config.iteration
```
## Map

Finaly, we can use this function to show the difference between Double DQN and DQfD.
```
    map(dqfd_scores=dqfd_mean_scores, ddqn_scores=ddqn_mean_scores, xlabel='Red: dqfd         Blue: ddqn', ylabel='Scores')
```

## Abstract 

In this work, we make use of demonstration data from multiple experts to pre-train the agent so that it can perform well in the task from the start of learning, and then continue improving from its own self-generated data. Enabling learning in this framework opens up the possibility of applying RL to many real world problems where we have multiple experts' data available but accurate simulators don’t exist.

Our method is closest to the recent approach - Deep Q-Learning  From  Demonstrations  (DQfD)  which  combines demonstrations  with  reinforcement  learning.  DQfD  improves  learning  speed  on  Atari,  including  a  margin  loss which encourages the expert actions to have higher Q-values than all other actions. This loss can make improving upon the demonstrator policy impossible which is not the case for our method. Prior work has previously explored improving beyond the demonstrator policy but by using only one expert's data. While our approach is a multi-expert approach where we are trying to see whether the agent is able to explore quickly by building upon the knowledge gained from the experts data since all the experts have been trained using a different technique and are expected to have a different policy to perform exploration and exploitation. While previous  work  focuses  on  beating  the  experts it learned from, we show that we can extend the state of the art in RL with demonstrations by introducing new methods to incorporate demonstrations from multiple experts and quickly converges along with defeating those individual experts.

We evaluated the multi-expert DQfD on Cart-Pole-v1 setup. For our experiments, we evaluated four different algorithms: Full DQfD from 2 experts (DDQN and Actor-Critic), DDQN without any demonstration, DQfD from one expert data (DDQN Expert) and DQfD from one expert data (Actor-Critic Expert). We first trained two agents, one with DDQN with experience replay buffer and target network and another with Actor Critic setup. Both these experts were trained till their performance reached a decent level but not the best score of 500. The AC expert was stopped when it started to attain a mean score of 185.7 with a SD of 41.1 whereas the DDQN expert was stopped when it started to attain a mean score of 336.1 and a SD of 74.7 on the Cart-Pole-v1 task. Once we had these two experts at our disposal, we started the training of our agent using DQN but the main difference was that the agent was pre-trained on the roll-outs of these two agents before actually interacting with the environment. We had to be careful about the number of roll-outs we were stacking together of those two experts. If they are not approximately similar in count then our agent will tend to adapt to the policy of the expert with more number of roll-outs present in the buffer memory.

From the results shown in the paper below we can clearly see that the  DQfD with multi-expert outperforms the vanilla DDQN by a fair margin.  Also  one  thing  to  notice  here  is  that  the  vanilla DDQN model is never able to achieve the highest score of 500  (not  even  once)  whereas  our  DQfD  with  two  experts(AC and DDQN) where both these experts had an average score  of  150-300,  is  able  to  achieve  a  max  score  of 500. Which means it not only beats the vanilla DDQN by a huge margin but it also outperforms those individual experts fairly well. 

Another key finding was that DQfD is able to learn faster when it is pre-trained on data from several experts as compared to when  it’s  shown  data  from  just  one  expert. In Figure 2 it shows that our DQfD from multi-experts achieves a score of 300-350 pretty soon and then keeps on gradually increasing and stabilizes around 450-500. The single expert trained DQfD also eventually reaches that range but it takes it more time to build upon the knowledge gained from single expert.

The  same  thing  can  be  observed  in  Figure  4  where  we compare DQfD on one expert (DQN) v/s DQfD from multi-expert. Even here we  can observe the same behavior although not as evident as in Figure 2 because our DQfD from DQN expert had an average  score of 350 which means it was a really good expert in itself as compared to the Actor-Critic expert.



