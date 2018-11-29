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



