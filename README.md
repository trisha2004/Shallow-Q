#  Shallow Q-Network (SQN) for Tic-Tac-Toe

**Course**: CS F407 - Artificial Intelligence  
**Institute**: BITS Pilani, K. K. Birla Goa Campus  
**Assignment**: Programming Assignment 2 – Shallow Q-Network (SQN)  
---
##  Overview

This repository implements a **Shallow Q-Network (SQN)** agent to play the classic game **Tic-Tac-Toe** using **Reinforcement Learning**. The model is trained via Q-learning with experience replay and an epsilon-greedy exploration strategy.
---

## Project Structure

Project files:
- `PlayerSQN.py` — Main script with PlayerSQN implementation and training loop  
- `NN_training.py` — Contains model definition and experience replay logic  
- `TicTacToe.py` — Game environment logic      
- `README.md` — You're reading it!

---

##  How It Works

### � Game: Tic-Tac-Toe

- 9-cell board represented as a list  
- Player 1: smart/random agent (controlled by `smartMovePlayer1` probability)  
- Player 2: your learning agent (SQN)

###  Agent: SQN

- Input: 9 board positions  
- Output: 9 Q-values (one for each possible action)  
- Hidden Layers: 2 × 64 neurons, ReLU activation  
- Output Layer: Linear activation (for Q-values)  
- Loss Function: Mean Squared Error (MSE)  
- Optimizer: Adam (learning rate = 0.001)

---

##  Requirements

Install dependencies with:


Ensure you are using **Python ≥ 3.8**.

---

## � Running the Program

###  Train and Play a Game


- `0.5` is the smartness probability of Player 1 (range 0.0 to 1.0)

### Evaluate Your SQN


- Assesses your agent with Player 1 smartness set to 0.0 and 0.8  
- Awards marks based on total reward across 3 games

---

##  Rewards 

| Player 1 Smartness | Total Reward for SQN |  
|--------------------|----------------------|
| 0.0                | ≥ 1                  | 
|                    | = 0                  | 
|                    | < 0                  | 
| 0.8                | ≥ 1                  |
|                    | = 0                  | 
|                    | < 0                  |
---

##  Training Strategy

- **Experience Replay**: Stores (state, action, reward, next_state, done)  
- **Mini-Batch Learning**: Uses random batches to reduce correlation  
- **Bellman Update**: Q(s, a) ← r + γ max_a' Q(s', a')  
- **Epsilon-Greedy Policy**: Starts with ε=1.0 and decays to ε=0.1

