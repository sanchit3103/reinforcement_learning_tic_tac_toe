# Implementation of Reinforcement learning in Tic-tac-toe game

<p align="justify">
This project focuses on the implementation of Reinforcement Learning (RL) algorithm in tic-tac-toe game. In this, we are playing against a purely random opponent, which means when the opponent needs to make a move, it will choose uniformly randomly from the empty spots, and put its mark in it. The opponent plays first and picks an arbitrary slot for its move. The game is modelled as an MDP with the player taking actions from relevant states in the game, and the opponent acting as the environment. The reward function is defined for win, lose, draw and non-terminal states of the game. The value iteration algorithm is implemented to determine the optimal actions for the player for each state of the game. The results for different trajectories of the game are shown below.
</p>

## Project implementation 
<p align="center">
  
  <img src = "https://user-images.githubusercontent.com/4907348/229299436-56deb00b-ccad-4039-8220-4fae9a6667c4.jpg"/>, &nbsp;&nbsp; <img src = "https://user-images.githubusercontent.com/4907348/229299472-e1bb4413-920f-4ff1-bdf5-7b6471947e5c.jpg"/>, &nbsp;&nbsp; <img src = "https://user-images.githubusercontent.com/4907348/229299489-f6d58094-7011-48cb-9db0-9b5fbae49c8c.jpg"/> 

</p>

## Details to run the code

* <b> tct.py: </b> Random trajectory generator for the project. Also generates the trajectory with optimal actions.
* <b> val_iteration.py: </b> Run this file to execute value iteration algorithm on the game and obtain results.
