|                                     |   capture rate (%) |   £ per MW per year |   cycles per day |
|:------------------------------------|-------------------:|--------------------:|-----------------:|
| Perfect foresight (upper bound)     |              100   |               34947 |             1.51 |
| Rule: yesterday's pattern           |               29.9 |               10437 |             0.84 |
| Forecast + optimiser                |               69.6 |               24328 |             1.34 |
| RL: DQN (seed 0)                    |               45.2 |               15810 |             0.83 |
| RL: DQN (seed 1)                    |               43   |               15026 |             0.72 |
| RL: DQN (seed 2)                    |               42.8 |               14962 |             0.94 |
| RL: PPO (seed 0)                    |               31.6 |               11032 |             0.4  |
| RL: PPO (seed 1)                    |               28.8 |               10049 |             0.3  |
| RL: PPO (seed 2)                    |               21.5 |                7513 |             0.18 |
| RL: DQN, mean of 3 seeds (± sd 1.4) |               43.7 |               15266 |             0.83 |
| RL: PPO, mean of 3 seeds (± sd 5.2) |               27.3 |                9531 |             0.29 |

Capture rate (%) on calm and volatile days

|                                 |   calm |   volatile |
|:--------------------------------|-------:|-----------:|
| Perfect foresight (upper bound) |  100   |      100   |
| Rule: yesterday's pattern       |   29.7 |       30   |
| Forecast + optimiser            |   62.1 |       76.4 |
| RL: DQN (seed 0)                |   38.8 |       51   |
| RL: DQN (seed 1)                |   28.1 |       56.4 |
| RL: DQN (seed 2)                |   36.1 |       48.9 |
| RL: PPO (seed 0)                |   16.1 |       45.4 |
| RL: PPO (seed 1)                |   11.2 |       44.5 |
| RL: PPO (seed 2)                |    6.9 |       34.6 |