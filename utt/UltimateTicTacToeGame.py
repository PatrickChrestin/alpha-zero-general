from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .UltimateTicTacToeLogic import Board
import numpy as np

class UltimateTicTacToeGame(Game):
    """
    Ultimate Tic Tac Toe game class implementing the Game interface.
    """
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    @staticmethod
    def getSquarePiece(piece):
        return UltimateTicTacToeGame.square_content[piece]

    def __init__(self):
        self.n = 3  # Size of each small board
        self.N = 3  # Number of small boards in each row/column
        self.last_move = None  # Track the last move to determine the next valid board

    def getInitBoard(self):
        # Return initial board (numpy board)
        b = Board(self.n, self.N)
        # Convert the pieces and meta_board into a single numpy array
        # We'll use the first n*N × n*N elements for the pieces
        # and append the N×N meta_board at the end
        board = np.array(b.pieces)
        meta = np.array(b.meta_board)
        return np.concatenate((board.flatten(), meta.flatten()))

    def getBoardSize(self):
        # Return (n*N, n*N) for the total board size
        return (self.n * self.N, self.n * self.N)

    def getActionSize(self):
        # Return number of actions: n*N*n*N = 81 possible moves
        # Each move is a position (row, col) in the global board
        return self.n * self.N * self.n * self.N + 1  # +1 for pass move

    def getNextState(self, board, player, action):
        # If player takes action on board, return next (board, player)
        # Action must be a valid move
        if action == self.n * self.N * self.n * self.N:  # Pass move
            return (board, -player)
            
        b = Board(self.n, self.N)
        
        # Extract pieces and meta_board from the flattened board
        total_pieces = self.n * self.N * self.n * self.N
        pieces_flat = board[:total_pieces]
        meta_flat = board[total_pieces:]
        
        # Reshape them back
        b.pieces = pieces_flat.reshape((self.n * self.N, self.n * self.N)).tolist()
        b.meta_board = meta_flat.reshape((self.N, self.N)).tolist()
        
        # Convert action to move coordinates
        move = self._action_to_move(action)
        b.execute_move(move, player)
        
        # Store last move to determine next valid board
        self.last_move = move
        
        # Recombine pieces and meta_board for the next state
        new_pieces = np.array(b.pieces).flatten()
        new_meta = np.array(b.meta_board).flatten()
        new_board = np.concatenate((new_pieces, new_meta))
        
        return (new_board, -player)


    def getValidMoves(self, board, player):
        # Return a fixed size binary vector of valid moves
        valids = [0] * self.getActionSize()
        b = Board(self.n, self.N)
        
        # Extract pieces and meta_board from the flattened board
        total_pieces = self.n * self.N * self.n * self.N
        pieces_flat = board[:total_pieces]
        meta_flat = board[total_pieces:]
        
        # Reshape them back
        b.pieces = pieces_flat.reshape((self.n * self.N, self.n * self.N)).tolist()
        b.meta_board = meta_flat.reshape((self.N, self.N)).tolist()
        
        legal_moves = b.get_legal_moves(player, self.last_move)
        
        if len(legal_moves) == 0:
            valids[-1] = 1  # Pass if no legal moves
            return np.array(valids)
            
        for move in legal_moves:
            action = self._move_to_action(move)
            valids[action] = 1
            
        return np.array(valids)

    def getGameEnded(self, board, player):
        # Return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n, self.N)
        
        # Extract pieces and meta_board from the flattened board
        total_pieces = self.n * self.N * self.n * self.N
        pieces_flat = board[:total_pieces]
        meta_flat = board[total_pieces:]
        
        # Reshape them back
        b.pieces = pieces_flat.reshape((self.n * self.N, self.n * self.N)).tolist()
        b.meta_board = meta_flat.reshape((self.N, self.N)).tolist()
        
        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves(self.last_move):
            return 0
        # Draw
        return 1e-4

    def getCanonicalForm(self, board, player):
        # For boards that include meta information, we need to handle them carefully
        total_pieces = self.n * self.N * self.n * self.N 
        pieces_part = board[:total_pieces] * player
        meta_part = board[total_pieces:]  # Meta board doesn't get flipped
        return np.concatenate((pieces_part, meta_part))

    def getSymmetries(self, board, pi):
        # Mirror, rotational symmetries
        assert len(pi) == self.getActionSize()
        
        # Reshape pi to exclude the pass move
        pi_board = np.reshape(pi[:-1], (self.n * self.N, self.n * self.N))
        
        # Get all symmetries
        symmetries = []
        for i in range(1, 5):  # 4 rotations
            for j in [True, False]:  # with/without flipping
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                symmetries.append((newB, list(newPi.ravel()) + [pi[-1]]))
        
        return symmetries

    def stringRepresentation(self, board):
        # Convert board to a string for MCTS
        return board.tostring()
        
    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def _action_to_move(self, action):
        # Convert action index to move coordinates (row, col)
        row = action // (self.n * self.N)
        col = action % (self.n * self.N)
        return (row, col)
        
    def _move_to_action(self, move):
        # Convert move coordinates to action index
        row, col = move
        return row * (self.n * self.N) + col

    @staticmethod
    def display(board):
        n = 3  # Size of each small board
        N = 3  # Number of small boards

        # Extract pieces and meta_board from the flattened board
        total_pieces = n * N * n * N
        pieces_flat = board[:total_pieces]
        pieces = pieces_flat.reshape((n * N, n * N))
        
        # Print column numbers
        print("   ", end="")
        for i in range(n * N):
            print(i % n if i % n > 0 else i // n, end=" ")
        print("")
        
        # Print horizontal line
        print("  " + "-" * (n * N * 2 + N - 1))
        
        for i in range(n * N):
            print(f"{i} |", end="")  # Print row number
            for j in range(n * N):
                piece = pieces[i][j]
                print(UltimateTicTacToeGame.square_content[piece], end=" ")
                
                # Print vertical separator between small boards
                if (j + 1) % n == 0 and j < n * N - 1:
                    print("|", end="")
            print("|")
            
            # Print horizontal separator between small boards
            if (i + 1) % n == 0 and i < n * N - 1:
                print("  " + "-" * (n * N * 2 + N - 1))
        
        # Print bottom horizontal line
        print("  " + "-" * (n * N * 2 + N - 1))
        
        # Extract and display meta_board if needed for debugging
        meta_flat = board[total_pieces:]
        meta_board = meta_flat.reshape((N, N))
        print("\nMeta Board Status:")
        for i in range(N):
            for j in range(N):
                print(UltimateTicTacToeGame.square_content[meta_board[i][j]], end=" ")
            print()