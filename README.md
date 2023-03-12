#  Advance Statistical Learning Course.

The project requirements are stored in `requirements.txt` file. To install all the necessary dependencies run the following command : 

```bash
pip install -r requirements.txt

``` 

The `frozen_lake.ipynb` file contains the steps of our experiments for the frozen lake part. The main script to launch the simulation is :

```bash
python3 main.py 
``` 
To exit the simulation press the `ESCAPE` key repeatedly

## 1st Environment : Frozen Lake 
![ Frozen lake](images/frozen_lake.gif) 

This environment is part of the Toy Text environments, full documentation is available [Here](https://gymnasium.farama.org/environments/toy_text/frozen_lake/). The game starts with the agent at location [0,0] of the frozen lake with the goal located at [3,3] for the 4x4 environment or [7,7] for the 8x8 environment, here are some neccessary features : 

- Holes in the ice are distributed in set locations when using a pre-determined map or in random locations when a random map is generated. 
- The player can make a move until they either reach the goal or fall in a hole.
- The lake is slippery by default o the player may move perpendicular to the intended direction sometimes (see `is_slippery`). Furthermore, randomly generated worlds will always have a path to the goal.

To tackle the problem we will implement two different methods : 

### 1. Q-Learning
### 2. SARSA


## 2nd Environment : Acrobat
![ Acrobat](images/acrobot.gif) 

