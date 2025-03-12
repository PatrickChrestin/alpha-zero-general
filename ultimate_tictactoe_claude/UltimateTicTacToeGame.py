from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .UltimateTicTacToeLogic import Board
import numpy as np

"""
Game class implementation for Ultimate TicTacToe.
Based on the TicTacToeGame by Evgeny Tyurin.
"""
class UltimateTicTacToeGame(Game):
    def __init__(self):
        self.last_move = None  # Track the last move explicitly

    def getInitBoard(self):
        # Return initial board (numpy board)
        b = Board()
        # Convert the 3D structure to a 9x9 numpy array
        board_matrix = np.zeros((9, 9), dtype=np.int8)
        
        # The initial board is all zeros
        return board_matrix

    def getBoardSize(self):
        # (a,b) tuple
        return (9, 9)

    def getActionSize(self):
        # Return number of actions: 9 small boards × 9 positions each + 1 pass action
        return 9 * 9 + 1

    def getNextState(self, board, player, action):
        # If player takes action on board, return next (board, player)
        # Action must be a valid move
        if action == 9 * 9:  # Pass move
            return (board, -player)
        
        # Convert action to (big_x, big_y, small_x, small_y) coordinates
        big_x = action // 27
        big_y = (action % 27) // 9
        small_x = (action % 9) // 3
        small_y = action % 3
        
        # Create a Board instance and set its state from numpy array
        b = Board()
        
        # Fill the board with the current state
        for by in range(3):
            for bx in range(3):
                for sy in range(3):
                    for sx in range(3):
                        x = bx * 3 + sx
                        y = by * 3 + sy
                        if board[x][y] != 0:
                            b[bx][by][sx][sy] = board[x][y]
        
        # Set the last move explicitly
        if self.last_move is not None:
            b.last_move = self.last_move
        
        # Execute the move
        b.execute_move((big_x, big_y, small_x, small_y), player)
        
        # Store this move as the last move
        self.last_move = (big_x, big_y, small_x, small_y)
        
        # Convert back to numpy array
        new_board = np.zeros((9, 9), dtype=np.int8)
        for by in range(3):
            for bx in range(3):
                for sy in range(3):
                    for sx in range(3):
                        x = bx * 3 + sx
                        y = by * 3 + sy
                        new_board[x][y] = b[bx][by][sx][sy]
        
        return (new_board, -player)

    def getValidMoves(self, board, player):
        # Return a fixed size binary vector of valid moves
        valids = [0] * self.getActionSize()
        
        # Create a Board instance and set its state from numpy array
        b = Board()
        
        # Fill the board with the current state
        for by in range(3):
            for bx in range(3):
                for sy in range(3):
                    for sx in range(3):
                        x = bx * 3 + sx
                        y = by * 3 + sy
                        if board[x][y] != 0:
                            b[bx][by][sx][sy] = board[x][y]
        
        # Set the last move explicitly
        if self.last_move is not None:
            b.last_move = self.last_move
        
        # Get legal moves as (big_x, big_y, small_x, small_y) tuples
        legal_moves = b.get_legal_moves(player)
        
        if len(legal_moves) == 0:
            valids[-1] = 1  # Pass move
            return np.array(valids)
        
        # Set valid moves in the binary vector
        for big_x, big_y, small_x, small_y in legal_moves:
            action = big_x * 27 + big_y * 9 + small_x * 3 + small_y
            valids[action] = 1
        
        return np.array(valids)

    def getGameEnded(self, board, player):
        # Return 0 if game not ended, 1 if player won, -1 if player lost, small value for draw
        
        # Create a Board instance and set its state from numpy array
        b = Board()
        
        # Fill the board with the current state and reconstruct meta-board
        for by in range(3):
            for bx in range(3):
                # First fill all small boards
                for sy in range(3):
                    for sx in range(3):
                        x = bx * 3 + sx
                        y = by * 3 + sy
                        if board[x][y] != 0:
                            b[bx][by][sx][sy] = board[x][y]
                
                # Then check if small boards are won
                if b[bx][by].winner == 0:
                    b[bx][by]._is_win(1)  # Check if player 1 won
                    b[bx][by]._is_win(-1)  # Check if player -1 won
                
                # Update meta-board
                if b[bx][by].winner != 0:
                    b.meta_board[bx][by] = b[bx][by].winner
                elif b[bx][by].is_full():
                    b.meta_board[bx][by] = 2  # Mark as draw
        
        # Check if game is won at meta level
        if b.is_meta_win(player):
            return 1
        if b.is_meta_win(-player):
            return -1
        
        # Check for draw (all small boards are either won or full)
        if b.is_meta_board_full():
            return 0.000001  # Small value for draw
        
        # Game not ended
        return 0

    def getCanonicalForm(self, board, player):
        # Return state if player==1, else return -state if player==-1
        return player * board

    def getSymmetries(self, board, pi):
        # Mirror, rotational
        assert(len(pi) == self.getActionSize())
        
        # Extract the pass action
        pass_action = pi[-1]
        
        # Reshape the action probabilities to 9x9 (excluding pass)
        pi_board = np.reshape(pi[:-1], (9, 9))
        
        # Apply symmetries
        symmetries = []
        for i in range(1, 5):  # Rotations by 0, 90, 180, 270 degrees
            for j in [True, False]:  # Flipped or not
                new_b = np.rot90(board, i)
                new_pi = np.rot90(pi_board, i)
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)
                symmetries.append((new_b, list(new_pi.ravel()) + [pass_action]))
        
        return symmetries

    def stringRepresentation(self, board):
        # 9x9 numpy array (canonical board)
        return board.tostring()

    @staticmethod
    def display(board):
        n = board.shape[0]
        assert n == 9, "Board should be 9x9"
        
        print("   ", end="")
        for y in range(n):
            print(y, "", end="")
            if y % 3 == 2:
                print("  ", end="")
        print("")
        
        print("  ", end="")
        for y in range(n+3):  # +3 for the separators
            print("-", end="")
        print("")
        
        for y in range(n):
            print(y, "|", end="")  # Print row number
            for x in range(n):
                piece = board[x][y]
                if piece == -1:
                    print("X ", end="")
                elif piece == 1:
                    print("O ", end="")
                else:
                    print("- ", end="")
                if x % 3 == 2:
                    print("|", end="")  # Vertical separator for small boards
            print("")
            if y % 3 == 2:
                print("  ", end="")
                for _ in range(n+3):  # +3 for the separators
                    print("-", end="")
                print("")  # Horizontal separator for small boards