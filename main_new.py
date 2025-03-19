import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Reshape, Dropout

from utt.keras.UltimateTicTacToeNNet import UltimateTicTacToeNNet


class UltimateTicTacToeGame:
    """
    Ultimate Tic Tac Toe game implementation.
    
    Board representation:
    - Each small board is a 3x3 tic-tac-toe board
    - The entire game is a 3x3 grid of these small boards (9x9 total)
    - Players must play in the board corresponding to the last move's position
    - Win by getting 3 small boards in a row
    """
    def __init__(self):
        # Game board is represented as:
        # board[0] = player 1 positions (1 where player 1 has played)
        # board[1] = player 2 positions (1 where player 2 has played)
        # board[2] = valid moves (1 for valid, 0 for invalid)
        self.n = 3  # Size of each small tic-tac-toe board
        self.board_size = self.n**2  # 9x9 board
        self.action_size = self.n**4  # 81 possible actions
        
        # Additional state variables
        self.small_board_status = np.zeros((self.n, self.n), dtype=np.int8)  # Tracks small board wins
        self.current_player = 1
        self.next_board = None  # Which small board to play in next (None means any board)

    def getInitBoard(self):
        """Initialize the board state"""
        # Initialize board to zeros with shape (3, 9, 9)
        # Channel 0: Player 1 positions
        # Channel 1: Player 2 positions
        # Channel 2: Valid moves indicator
        board = np.zeros((3, self.n**2, self.n**2), dtype=np.int8)
        board[2] = 1  # All moves are valid at the start
        return board
    
    def getBoardSize(self):
        """Return board dimensions"""
        return (3, self.n**2, self.n**2)
    
    def getActionSize(self):
        """Return number of possible actions"""
        return self.action_size
        
    def getNextState(self, board, player, action):
        """Returns board after applying action"""
        board_copy = np.copy(board)
        
        # Convert action to coordinates
        action_row, action_col = action // self.n**2, action % self.n**2
        
        # Update board with the move
        player_idx = 0 if player == 1 else 1
        board_copy[player_idx, action_row, action_col] = 1
        
        # Calculate which small board the action was in
        small_board_row, small_board_col = action_row // self.n, action_col // self.n
        
        # Calculate the position within the small board
        pos_row, pos_col = action_row % self.n, action_col % self.n
        
        # Next player must play in the small board corresponding to this position
        next_small_board_row, next_small_board_col = pos_row, pos_col
        
        # Update valid moves
        board_copy[2] = 0  # Reset all to invalid
        
        # Check if the targeted small board is full or already won
        target_board_status = self.getSmallBoardStatus(board_copy, next_small_board_row, next_small_board_col)
        
        if target_board_status != 0:  # Board is won or full
            # Player can play in any unfilled and unwon small board
            for i in range(self.n):
                for j in range(self.n):
                    if self.getSmallBoardStatus(board_copy, i, j) == 0:  # Board is still playable
                        # Mark all empty cells in this small board as valid
                        start_row, start_col = i * self.n, j * self.n
                        for r in range(self.n):
                            for c in range(self.n):
                                if board_copy[0, start_row + r, start_col + c] == 0 and board_copy[1, start_row + r, start_col + c] == 0:
                                    board_copy[2, start_row + r, start_col + c] = 1
        else:
            # Mark valid moves in the targeted small board
            start_row, start_col = next_small_board_row * self.n, next_small_board_col * self.n
            for r in range(self.n):
                for c in range(self.n):
                    if board_copy[0, start_row + r, start_col + c] == 0 and board_copy[1, start_row + r, start_col + c] == 0:
                        board_copy[2, start_row + r, start_col + c] = 1
        
        # Switch player
        next_player = -player
        
        return board_copy, next_player
    
    def getSmallBoardStatus(self, board, row, col):
        """Check if a small board is won or full"""
        start_row, start_col = row * self.n, col * self.n
        small_board_p1 = board[0, start_row:start_row+self.n, start_col:start_col+self.n]
        small_board_p2 = board[1, start_row:start_row+self.n, start_col:start_col+self.n]
        
        # Check if player 1 won
        if self.checkWin(small_board_p1):
            return 1
        
        # Check if player 2 won
        if self.checkWin(small_board_p2):
            return -1
        
        # Check if board is full
        if np.all(small_board_p1 + small_board_p2 == 1):
            return 0.5  # Draw
            
        return 0  # Still in play
    
    def checkWin(self, board):
        """Check if a player has won a small board"""
        # Check rows
        for i in range(self.n):
            if np.all(board[i, :] == 1):
                return True
                
        # Check columns
        for i in range(self.n):
            if np.all(board[:, i] == 1):
                return True
                
        # Check diagonals
        if np.all(np.diag(board) == 1):
            return True
        if np.all(np.diag(np.fliplr(board)) == 1):
            return True
            
        return False
    
    def updateSmallBoardStatus(self, board):
        """Update the status of all small boards"""
        small_board_status = np.zeros((self.n, self.n), dtype=np.int8)
        
        for i in range(self.n):
            for j in range(self.n):
                small_board_status[i, j] = self.getSmallBoardStatus(board, i, j)
        
        return small_board_status
    
    def getValidMoves(self, board):
        """Returns a binary array of valid moves"""
        return board[2].flatten()
    
    def getGameEnded(self, board):
        """
        Returns:
        0 if game is not ended
        1 if player 1 won
        -1 if player 2 won
        0.1 for a draw
        """
        # Update small board status
        small_board_status = self.updateSmallBoardStatus(board)
        
        # Check if any player has won
        # Check rows
        for i in range(self.n):
            if np.all(small_board_status[i, :] == 1):
                return 1
            if np.all(small_board_status[i, :] == -1):
                return -1
                
        # Check columns
        for i in range(self.n):
            if np.all(small_board_status[:, i] == 1):
                return 1
            if np.all(small_board_status[:, i] == -1):
                return -1
                
        # Check diagonals
        if np.all(np.diag(small_board_status) == 1) or np.all(np.diag(np.fliplr(small_board_status)) == 1):
            return 1
        if np.all(np.diag(small_board_status) == -1) or np.all(np.diag(np.fliplr(small_board_status)) == -1):
            return -1
        
        # Check if there are any valid moves left
        if np.sum(board[2]) == 0:
            return 0.1  # Draw
            
        # Game is still ongoing
        return 0
    
    def getCanonicalForm(self, board, player):
        """Convert board to canonical form (from current player's perspective)"""
        if player == 1:
            return board
        else:
            # Swap player 1 and player 2 perspectives
            canonical_board = np.copy(board)
            canonical_board[0], canonical_board[1] = board[1], board[0]
            return canonical_board
    
    def getSymmetries(self, board, pi):
        """
        Returns symmetrically equivalent board positions and corresponding policies.
        For Ultimate Tic Tac Toe, we'll just handle rotations and reflections of the entire board.
        """
        # For simplicity, we'll just return the original board and policy for now
        # A more complete implementation would include rotations and reflections
        return [(board, pi)]
    
    def stringRepresentation(self, board):
        """String representation of the board for MCTS dictionary keys"""
        return board.tobytes()
    
    def display(self, board):
        """Print the board state"""
        p1_board = board[0]
        p2_board = board[1]
        
        print("-" * 25)
        for i in range(9):
            row_str = "| "
            for j in range(9):
                if p1_board[i, j] == 1:
                    row_str += "X "
                elif p2_board[i, j] == 1:
                    row_str += "O "
                else:
                    row_str += ". "
                
                # Add vertical separators between small boards
                if j % 3 == 2 and j < 8:
                    row_str += "| "
            
            row_str += "|"
            print(row_str)
            
            # Add horizontal separators between small boards
            if i % 3 == 2 and i < 8:
                print("-" * 25)
        
        print("-" * 25)


class RandomPlayer:
    """Player that selects a random valid move"""
    def __init__(self, game):
        self.game = game
        
    def play(self, board):
        valid_moves = self.game.getValidMoves(board)
        valid_indices = np.where(valid_moves == 1)[0]
        
        if len(valid_indices) > 0:
            return np.random.choice(valid_indices)
        
        return -1  # No valid moves


import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import time
import os
import pickle
from tqdm import tqdm

class MCTS:
    """
    Monte Carlo Tree Search implementation for AlphaZero
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}   # stores #times board s was visited
        self.Ps = {}   # stores initial policy (returned by neural net)
        
        self.Es = {}   # stores game.getGameEnded ended for board s
        self.Vs = {}   # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts] if counts_sum != 0 else [1/len(counts) for _ in counts]
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        
        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        s = self.game.stringRepresentation(canonicalBoard)

        # Terminal node
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard)
        if self.Es[s] != 0:
            # Terminal state
            return -self.Es[s]

        # Leaf node
        if s not in self.Ps:
            # Evaluate the leaf using neural network
            # The input to the network is a 3×9×9 tensor
            self.Ps[s], v = self.nnet.predict(canonicalBoard[np.newaxis, :])
            self.Ps[s] = self.Ps[s][0]
            v = v[0][0]
            
            # Get valid moves
            valids = self.game.getValidMoves(canonicalBoard)
            
            # Mask invalid moves
            self.Ps[s] = self.Ps[s] * valids
            
            # Normalize probabilities
            sum_Ps = np.sum(self.Ps[s])
            if sum_Ps > 0:
                self.Ps[s] /= sum_Ps  # renormalize
            else:
                # If all valid moves were masked, do a uniform distribution over valid moves
                print("All valid moves were masked, doing a uniform distribution")
                self.Ps[s] = valids / np.sum(valids)
                
            self.Vs[s] = valids
            self.Ns[s] = 0
            
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # Pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * np.sqrt(self.Ns[s] + 1e-8)  # Q = 0

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v


class AlphaZeroTrainer:
    """
    This class implements the AlphaZero training algorithm
    """
    def __init__(self, game, neural_net, args):
        self.game = game
        self.nnet = neural_net
        self.args = args
        self.mcts = MCTS(game, neural_net, args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        
    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi, v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, -1 if the player lost
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board)

            if r != 0:
                # Update all game values in trainExamples based on final result
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenOfQueue).
        """
        for i in tqdm(range(1, self.args.numIters + 1)):
            print(f'Starting Iteration {i}...')
            
            # Self-play phase
            iterationTrainExamples = []
            for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                iterationTrainExamples.extend(self.executeEpisode())
                
            # Save the iteration examples to history
            self.trainExamplesHistory.append(iterationTrainExamples)
            
            # Remove oldest examples if we exceed our memory capacity
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                self.trainExamplesHistory.pop(0)
                
            # Shuffle and prepare training data
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            random.shuffle(trainExamples)
            
            # Training phase
            # Extract training data
            training_states = np.array([x[0] for x in trainExamples])
            training_targets_pi = np.array([x[1] for x in trainExamples])
            training_targets_v = np.array([x[2] for x in trainExamples])
            
            # Actually train the network
            print("Training the neural network...")
            self.nnet.model.fit(
                x=training_states,
                y=[training_targets_pi, training_targets_v],
                batch_size=self.args.batch_size,
                epochs=self.args.epochs,
                verbose=1
            )
            
            print(f'Finished training iteration {i}')
            
            # Save the model after each iteration
            folder = self.args.checkpoint
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder, f'iteration_{i}.h5')
            self.nnet.model.save(filename)
    
    def save_model(self, filename='best_model.h5'):
        """
        Save the current model to a file
        """
        filepath = os.path.join(self.args.checkpoint, filename)
        self.nnet.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filename='best_model.h5'):
        """
        Load a model from a file
        """
        filepath = os.path.join(self.args.checkpoint, filename)
        if os.path.exists(filepath):
            self.nnet.model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
            return True
        return False


class AlphaZeroPlayer:
    """
    Player that uses the trained AlphaZero neural network
    """
    def __init__(self, game, nnet, args, mcts_simulations=None):
        self.game = game
        self.nnet = nnet
        self.args = args
        
        # Allow different number of MCTS simulations for play vs training
        if mcts_simulations is not None:
            play_args = args
            play_args.numMCTSSims = mcts_simulations
            self.mcts = MCTS(game, nnet, play_args)
        else:
            self.mcts = MCTS(game, nnet, args)
        
    def play(self, board, temp=0):
        """Make a move using MCTS guided by the neural network"""
        canonicalBoard = self.game.getCanonicalForm(board, 1)
        pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
        
        # For actual play, we should choose the action with highest probability
        return np.argmax(pi)


class HumanPlayer:
    """
    Player class for a human player
    """
    def __init__(self, game):
        self.game = game
        
    def play(self, board):
        """Get move from human player via console input"""
        valid_moves = self.game.getValidMoves(board)
        print("\nYour turn. Valid moves:")
        
        # Display valid moves in a more readable format
        for action in range(len(valid_moves)):
            if valid_moves[action] == 1:
                row, col = action // 9, action % 9  # For 9x9 board
                board_row, board_col = row // 3, col // 3
                pos_row, pos_col = row % 3, col % 3
                print(f"Move {action}: Small board ({board_row},{board_col}), Position ({pos_row},{pos_col})")
        
        # Get and validate input
        while True:
            try:
                move = int(input("\nEnter your move (number): "))
                if 0 <= move < len(valid_moves) and valid_moves[move] == 1:
                    return move
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Please enter a valid number.")


def play_game(game, player1, player2, display=True):
    """Play a game between two players"""
    board = game.getInitBoard()
    player = 1
    
    while True:
        if display:
            game.display(board)
            
        if player == 1:
            action = player1.play(board)
            print(f"Player 1 plays: {action}")
        else:
            action = player2.play(board)
            print(f"Player 2 plays: {action}")
            
        board, player = game.getNextState(board, player, action)
        
        if game.getGameEnded(board) != 0:
            if display:
                game.display(board)
                result = game.getGameEnded(board)
                if result == 1:
                    print("Player 1 wins!")
                elif result == -1:
                    print("Player 2 wins!")
                else:
                    print("It's a draw!")
            return game.getGameEnded(board)


class Args:
    """Arguments for AlphaZero"""
    def __init__(self):
        # Neural Network params
        self.lr = 0.001
        self.dropout = 0.3
        self.epochs = 10
        self.batch_size = 64
        self.num_channels = 128
        
        # MCTS params
        self.numMCTSSims = 50  # Number of MCTS simulations per move
        self.cpuct = 1.0  # Exploration constant
        
        # Training params
        self.numIters = 10  # Number of training iterations
        self.numEps = 100  # Number of complete self-play games per iteration
        self.tempThreshold = 15  # Move threshold for temperature change
        self.numItersForTrainExamplesHistory = 20  # Max number of iterations to keep in history
        
        # Misc
        self.checkpoint = './temp/'
        self.load_model = False

if __name__ == "__main__":
    # Initialize game
    game = UltimateTicTacToeGame()
    
    # Initialize neural network
    args = Args()
    nnet = UltimateTicTacToeNNet(game, args)
    
    # Initialize AlphaZero trainer
    trainer = AlphaZeroTrainer(game, nnet, args)
    
    # Load model if available
    if args.load_model:
        trainer.load_model()
    
    # Start training
    trainer.learn()
    
    # Save the final model
    trainer.save_model()