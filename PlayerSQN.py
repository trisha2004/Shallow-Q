import sys
import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from TicTacToe import *  # Assuming TicTacToe is imported from another file

class PlayerSQN:
    def __init__(self):
        """
        Initializes the PlayerSQN class 
        """
        # SQN parameters
        self.model = self.create_model()
        self.epsilon = 1.0  
        self.epsilon_minimum = 0.01
        self.epsilon_decay = 0.997
        self.gamma = 0.95  
        self.buffer = deque(maxlen=5000) 
    
    def move(self, state):
        """
        Determines Player 2's move using epsilon-greedy policy based on the current state of the game.
        """
        if np.random.rand() <= self.epsilon:
            action = random.choice(self.get_valid_moves(state))
        else:
            state_input = np.reshape(state, [1, 9])
            q_values = self.model.predict(state_input, verbose=0)[0]
            valid_actions = self.get_valid_moves(state)
            action = max(valid_actions, key=lambda x: q_values[x])
        return action

    def create_model(self):
        """
        Constructs a neural network model for Shallow Q-Network.
        """
        model = Sequential()
        model.add(Dense(64, input_dim=9, activation='relu'))  
        model.add(Dense(64, activation='relu'))               
        model.add(Dense(9, activation='linear'))             
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def storage(self, state, action, reward, next_state, done):
        
        self.buffer.append((state, action, reward, next_state, done))
        
    
    def train(self, batch_size=32, numberofbatches=5):
        """
        Trains the SQN model
     """
        if len(self.buffer) < batch_size:
            return

        for _ in range(numberofbatches):  
            mini_batch = random.sample(self.buffer, batch_size)
            for state, action, reward, next_state, done in mini_batch:
                target = reward
                if not done:
                    next_input = np.reshape(next_state, [1, 9])
                    target += self.gamma * np.max(self.model.predict(next_input, verbose=0))

                state_input = np.reshape(state, [1, 9])
                q_values = self.model.predict(state_input, verbose=0)
                q_values[0][action] = target

                self.model.fit(state_input, q_values, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay


    def get_valid_moves(self, state):
        
        return [i for i in range(9) if state[i] == 0]

  
def main(smartMovePlayer1):
    """
    Simulates a TicTacToe game between Player 1 (random move player) and Player 2 (SQN-based player).
    """
    playerSQN = PlayerSQN()
    numberofepisodes = 100 

    for episode in range(1, numberofepisodes + 1):
        game = TicTacToe(smartMovePlayer1, playerSQN)  
        state = game.board
        done = False
        while not done:
            # Player 1 
            action = random.choice(playerSQN.get_valid_moves(state))
            isSuccess = game.make_move(action, 1)
            if isSuccess:
                if game.current_winner == 1 or game.is_full():
                    reward = -1 
                    playerSQN.storage(state, action, reward, game.board, True)
                    playerSQN.train(batch_size=32)
                    break

            # Player 2 
            action = playerSQN.move(state)
            isSuccess = game.make_move(action, 2)
            if isSuccess:
                if game.current_winner == 2:
                    reward = 1  
                    playerSQN.storage(state, action, reward, game.board, True)
                    playerSQN.train(batch_size=32)
                    break
                elif game.is_full():
                    reward = 0  
                    playerSQN.storage(state, action, reward, game.board, True)
                    playerSQN.train(batch_size=32)
                    break
                else:
                    reward = 0.1  # Reward for a valid move
                    playerSQN.storage(state, action, reward, game.board, False)
                    playerSQN.train(batch_size=32)

        # Epsilon decay
        if playerSQN.epsilon > playerSQN.epsilon_minimum:
            playerSQN.epsilon *= playerSQN.epsilon_decay
        print(f"Episode {episode}: Epsilon = {playerSQN.epsilon:.4f}")

   
    print("\nEvaluating the trained agent:")
    number_of_wins, number_of_losses, number_of_draws = 0, 0, 0
    for _ in range(10):
      game = TicTacToe(smartMovePlayer1, playerSQN)
        state = game.board
        done = False
        while not done:
            if random.random() < 0.5: 
                action = random.choice(playerSQN.get_valid_moves(state))
                game.make_move(action, 1)
            else:  # Player 2 
                action = playerSQN.move(state)
                game.make_move(action, 2)

            done = game.current_winner is not None or game.is_full()
            if game.current_winner == 2:
                number_of_wins += 1
            elif game.current_winner == 1:
                number_of_losses += 1
            else:
                number_of_draws += 1

    print(f"Evaluation - Wins: {number_of_wins}, Losses: {number_of_losses}, Draws: {number_of_draws}")

if __name__ == "__main__":
    try:
        smartMovePlayer1 = float(sys.argv[1])
        assert 0 <= smartMovePlayer1 <= 1
    except:
        print("Usage: python YourBITSid.py <smartMovePlayer1Probability>")
        sys.exit(1)

    main(smartMovePlayer1)
