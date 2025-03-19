'''
Ultimate Tic Tac Toe Board Logic.
Board data:
  1=player1, -1=player2, 0=empty
  Each position is a tuple (global_row, global_col)
'''

class Board():
    def __init__(self, n, N):
        """
        Initialize an Ultimate Tic Tac Toe board.
        n: Size of each small board (typically 3)
        N: Number of small boards in each row/column (typically 3)
        """
        self.n = n
        self.N = N
        self.pieces = [[0 for _ in range(n * N)] for _ in range(n * N)]
        self.meta_board = [[0 for _ in range(N)] for _ in range(N)]

    def __getitem__(self, index):
        return self.pieces[index]

    def get_legal_moves(self, player, last_move=None):
        """
        Returns a list of all legal moves.
        A move is represented as a tuple (row, col).
        """
        if last_move is None:
            # If this is the first move, all positions are valid
            return [(i, j) for i in range(self.n * self.N) for j in range(self.n * self.N)]
        
        last_row, last_col = last_move
        
        # Calculate which small board we're sent to
        target_board_row = last_row % self.n
        target_board_col = last_col % self.n
        
        # Check if that small board is already won or full
        if self.is_small_board_complete(target_board_row, target_board_col):
            # If the target board is complete, player can play anywhere on any non-complete board
            moves = []
            for i in range(self.N):
                for j in range(self.N):
                    if not self.is_small_board_complete(i, j):
                        moves.extend(self.get_moves_in_small_board(i, j))
            return moves
        else:
            # Player must play in the target board
            return self.get_moves_in_small_board(target_board_row, target_board_col)

    def get_moves_in_small_board(self, board_row, board_col):
        """
        Returns all legal moves in a specific small board.
        """
        moves = []
        
        # Calculate the global position of the small board
        start_row = board_row * self.n
        start_col = board_col * self.n
        
        for i in range(self.n):
            for j in range(self.n):
                # Calculate the global position
                global_row = start_row + i
                global_col = start_col + j
                
                # Check if the position is empty
                if self.pieces[global_row][global_col] == 0:
                    moves.append((global_row, global_col))
                    
        return moves

    def is_small_board_complete(self, board_row, board_col):
        """
        Checks if a small board is complete (won or full).
        """
        # Check if the small board has a winner
        if self.meta_board[board_row][board_col] != 0:
            return True
            
        # Calculate the global position of the small board
        start_row = board_row * self.n
        start_col = board_col * self.n
        
        # Check if the small board is full
        for i in range(self.n):
            for j in range(self.n):
                if self.pieces[start_row + i][start_col + j] == 0:
                    return False
                    
        # If we get here, the board is full with no winner
        self.meta_board[board_row][board_col] = 1e-4  # Draw
        return True

    def execute_move(self, move, player):
        """
        Execute a move on the board.
        """
        row, col = move
        
        # Place the piece
        self.pieces[row][col] = player
        
        # Calculate which small board the move belongs to
        board_row = row // self.n
        board_col = col // self.n
        
        # Check if the small board has been won
        self.update_meta_board(board_row, board_col, player)

    def update_meta_board(self, board_row, board_col, player):
        """
        Update the meta board after a move.
        """
        # If the small board already has a winner, return
        if self.meta_board[board_row][board_col] != 0:
            return
            
        # Calculate the global position of the small board
        start_row = board_row * self.n
        start_col = board_col * self.n
        
        # Check rows
        for i in range(self.n):
            if all(self.pieces[start_row + i][start_col + j] == player for j in range(self.n)):
                self.meta_board[board_row][board_col] = player
                return
                
        # Check columns
        for j in range(self.n):
            if all(self.pieces[start_row + i][start_col + j] == player for i in range(self.n)):
                self.meta_board[board_row][board_col] = player
                return
                
        # Check diagonals
        if all(self.pieces[start_row + i][start_col + i] == player for i in range(self.n)):
            self.meta_board[board_row][board_col] = player
            return
            
        if all(self.pieces[start_row + i][start_col + self.n - 1 - i] == player for i in range(self.n)):
            self.meta_board[board_row][board_col] = player
            return
            
        # Check if the small board is full (draw)
        if all(self.pieces[start_row + i][start_col + j] != 0 for i in range(self.n) for j in range(self.n)):
            self.meta_board[board_row][board_col] = 1e-4  # Draw

    def is_win(self, player):
        """
        Check if the player has won the game.
        """
        # Check rows
        for i in range(self.N):
            if all(self.meta_board[i][j] == player for j in range(self.N)):
                return True
                
        # Check columns
        for j in range(self.N):
            if all(self.meta_board[i][j] == player for i in range(self.N)):
                return True
                
        # Check diagonals
        if all(self.meta_board[i][i] == player for i in range(self.N)):
            return True
            
        if all(self.meta_board[i][self.N - 1 - i] == player for i in range(self.N)):
            return True
            
        return False

    def has_legal_moves(self, last_move=None):
        """
        Check if there are any legal moves left.
        """
        return len(self.get_legal_moves(1, last_move)) > 0