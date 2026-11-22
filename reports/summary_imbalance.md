|                                     |   capture rate (%) |   £ per MW per year |   cycles per day |
|:------------------------------------|-------------------:|--------------------:|-----------------:|
| Perfect foresight (upper bound)     |              100   |              103630 |             3.39 |
| Rule: yesterday's pattern           |                8.3 |                8565 |             0.86 |
| Forecast + optimiser                |               26.1 |               27012 |             1.51 |
| RL: DQN (seed 0)                    |               26   |               26947 |             1.25 |
| RL: DQN (seed 1)                    |               29.8 |               30933 |             1.5  |
| RL: DQN (seed 2)                    |               27.3 |               28246 |             1.27 |
| RL: PPO (seed 0)                    |               24   |               24835 |             1.55 |
| RL: PPO (seed 1)                    |               22.3 |               23116 |             1.34 |
| RL: PPO (seed 2)                    |               21.1 |               21905 |             1.17 |
| RL: DQN, mean of 3 seeds (± sd 2.0) |               27.7 |               28708 |             1.34 |
| RL: PPO, mean of 3 seeds (± sd 1.4) |               22.5 |               23286 |             1.35 |

Capture rate (%) on calm and volatile days

|                                 |   calm |   volatile |
|:--------------------------------|-------:|-----------:|
| Perfect foresight (upper bound) |  100   |      100   |
| Rule: yesterday's pattern       |    3.7 |       14.9 |
| Forecast + optimiser            |   20.4 |       34.4 |
| RL: DQN (seed 0)                |   22.4 |       31.3 |
| RL: DQN (seed 1)                |   27   |       34   |
| RL: DQN (seed 2)                |   24.7 |       30.9 |
| RL: PPO (seed 0)                |   20.5 |       29.1 |
| RL: PPO (seed 1)                |   19.3 |       26.7 |
| RL: PPO (seed 2)                |   19.3 |       23.9 |