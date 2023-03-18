#  Advance Statistical Learning Course.

The project requirements are stored in `requirements.txt` file. To install all the necessary dependencies run the following command : 

```bash
pip install -r requirements.txt

``` 

The `frozen_lake.ipynb` file contains the steps of our experiments for the frozen lake part. The `acrobot.ipynb` file contains the coressponding experiments and results for the Acrobot envirement.


 The main script to launch the simulation `main.py` here an example of how to launch it :

```bash
python3 main.py 
    --mode Train
    --methode Qlearning
    --env_name Frozen_lake
    --save True
``` 

- mode : Set the mode of execution to either `Train` or `Play`. Default value is `Train`.
- method : Set the method to use for the RL algorithm. Available options are "Qlearning", "SARSA", and "REINFORCE". Default value is "Qlearning".
- env_name : Set the name of the environment to use. Available options are `Frozen_lake` and `Acrobot`. Default value is "`Frozen_lake`.
- save : Specify whether to save the trained model or weights. Default value is `True`.
- visualize : Specify whether to visualize and render the environment. Default value is `False`.

To exit the simulation press the `CTRL + C` keys

## 1st Environment : Frozen Lake 
![ Frozen lake](images/frozen_lake.gif) 

This environment is part of the Toy Text environments, full documentation is available [Here](https://gymnasium.farama.org/environments/toy_text/frozen_lake/). The game starts with the agent at location [0,0] of the frozen lake with the goal located at [3,3] for the 4x4 environment or [7,7] for the 8x8 environment, here are some neccessary features : 

- Holes in the ice are distributed in set locations when using a pre-determined map or in random locations when a random map is generated. 
- The player can make a move until they either reach the goal or fall in a hole.
- The lake is slippery by default o the player may move perpendicular to the intended direction sometimes (see `is_slippery`). Furthermore, randomly generated worlds will always have a path to the goal.


## 2nd Environment : Acrobat
![ Acrobat](images/acrobot.gif) 

The Acrobot environment in Gym is a reinforcement learning problem full documentation can be accessed [here](https://www.gymlibrary.dev/environments/classic_control/acrobot/). The system consists of a two-link chain connected linearly, with one end of the chain fixed, and the joint between the two links actuated. The objective is to apply torque to the actuated joint to swing the free end of the chain above a specified height, starting from the initial state of the chain hanging downwards.