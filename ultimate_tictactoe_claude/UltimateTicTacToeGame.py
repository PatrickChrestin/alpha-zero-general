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
        
        # Rebuild the meta board based on current state
        self._rebuild_meta_board(b)
        
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
        
        # Rebuild the meta board based on current state
        self._rebuild_meta_board(b)
        
        # Set the last move explicitly if it exists
        if hasattr(self, 'last_move') and self.last_move is not None:
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
    def _rebuild_meta_board(self, b):
        """
        Rebuild the meta board by checking each small board for wins or draws.
        """
        for by in range(3):
            for bx in range(3):
                # Check if small board is won by player 1
                if b[bx][by]._is_win(1):
                    b.meta_board[bx][by] = 1
                    b[bx][by].winner = 1  # Update small board winner too
                # Check if small board is won by player -1
                elif b[bx][by]._is_win(-1):
                    b.meta_board[bx][by] = -1
                    b[bx][by].winner = -1  # Update small board winner too
                # Then check if small board is full (draw)
                elif b[bx][by].is_full():
                    b.meta_board[bx][by] = 2  # 2 represents a draw
                    b[bx][by].winner = 2  # Update small board winner too
                else:
                    b.meta_board[bx][by] = 0  # Board still in play
                    b[bx][by].winner = 0  # Update small board winner too

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
        
        # Rebuild the meta board based on current state
        self._rebuild_meta_board(b)
        
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
        # 9x9 numpy array (canonical board) + last_move information
        s = board.tostring()
        
        # Append last_move information to ensure states are differentiated
        if hasattr(self, 'last_move') and self.last_move is not None:
            # Convert last_move tuple to bytes and append to state
            last_move_str = f"{self.last_move[0]},{self.last_move[1]},{self.last_move[2]},{self.last_move[3]}"
            s += last_move_str.encode()
        
        return s

    @staticmethod
    def display(board):
        n = board.shape[0]
        assert n == 9, "Board should be 9x9"
        
        # Create a visualization of the meta board
        print("\nMeta Board Status:")
        meta_board = [[0 for _ in range(3)] for _ in range(3)]
        
        # Determine the meta board status
        for by in range(3):
            for bx in range(3):
                # Check for wins in each small board
                pieces = np.array([[board[bx*3+sx][by*3+sy] for sy in range(3)] for sx in range(3)])
                
                # Check for horizontal wins
                for y in range(3):
                    if pieces[0][y] == pieces[1][y] == pieces[2][y] != 0:
                        meta_board[bx][by] = pieces[0][y]
                        break
                
                # Check for vertical wins
                for x in range(3):
                    if pieces[x][0] == pieces[x][1] == pieces[x][2] != 0:
                        meta_board[bx][by] = pieces[x][0]
                        break
                
                # Check for diagonal wins
                if pieces[0][0] == pieces[1][1] == pieces[2][2] != 0:
                    meta_board[bx][by] = pieces[0][0]
                elif pieces[0][2] == pieces[1][1] == pieces[2][0] != 0:
                    meta_board[bx][by] = pieces[0][2]
                
                # Check if board is full but no winner
                if meta_board[bx][by] == 0:
                    is_full = True
                    for sx in range(3):
                        for sy in range(3):
                            if board[bx*3+sx][by*3+sy] == 0:
                                is_full = False
                                break
                        if not is_full:
                            break
                    if is_full:
                        meta_board[bx][by] = 2  # 2 means draw
        
        # Display meta board
        print("  0 1 2")
        print(" -------")
        for y in range(3):
            print(f"{y}|", end="")
            for x in range(3):
                symbol = "-"
                if meta_board[x][y] == 1:
                    symbol = "O"
                elif meta_board[x][y] == -1:
                    symbol = "X"
                elif meta_board[x][y] == 2:
                    symbol = "="  # Draw
                print(f"{symbol} ", end="")
            print("|")
        print(" -------")
        print(" X=won by X, O=won by O, ==draw, -=still in play\n")
        
        # Display the full board
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