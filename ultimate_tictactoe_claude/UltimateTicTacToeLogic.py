'''
Board class for the game of Ultimate TicTacToe.
Board consists of 9 smaller 3x3 TicTacToe boards arranged in a 3x3 grid.
Board data:
  1=white(O), -1=black(X), 0=empty
  
Each position is represented as (big_x, big_y, small_x, small_y) where:
  big_x, big_y: coordinates of the small board (0-2)
  small_x, small_y: coordinates within the small board (0-2)
  
The next board to play in is determined by the previous move's small coordinates.

Author: Adapted from Evgeny Tyurin's TicTacToe implementation
'''

class SmallBoard():
    def __init__(self):
        # Create 3x3 board
        self.pieces = [[0 for _ in range(3)] for _ in range(3)]
        self.winner = 0  # 0 if no winner, 1 for white, -1 for black, 2 for draw

    def __getitem__(self, index):
        return self.pieces[index]
    
    def is_full(self):
        for y in range(3):
            for x in range(3):
                if self.pieces[x][y] == 0:
                    return False
        return True
    
    def has_legal_moves(self):
        if self.winner != 0:
            return False
        return not self.is_full()
    
    def get_legal_moves(self):
        if self.winner != 0:
            return []
        
        moves = []
        for y in range(3):
            for x in range(3):
                if self.pieces[x][y] == 0:
                    moves.append((x, y))
        return moves
    
    def execute_move(self, move, color):
        x, y = move
        assert self.pieces[x][y] == 0
        self.pieces[x][y] = color
        
        # Check if this move created a win
        if self._is_win(color):
            self.winner = color
        elif self.is_full():
            self.winner = 2  # Draw
    
    def _is_win(self, color):
        # Check horizontals
        for y in range(3):
            if self.pieces[0][y] == self.pieces[1][y] == self.pieces[2][y] == color:
                return True
        
        # Check verticals
        for x in range(3):
            if self.pieces[x][0] == self.pieces[x][1] == self.pieces[x][2] == color:
                return True
        
        # Check diagonals
        if self.pieces[0][0] == self.pieces[1][1] == self.pieces[2][2] == color:
            return True
        if self.pieces[0][2] == self.pieces[1][1] == self.pieces[2][0] == color:
            return True
        
        return False

class Board():
    def __init__(self):
        # Create 3x3 grid of small boards
        self.boards = [[SmallBoard() for _ in range(3)] for _ in range(3)]
        
        # Meta board to track which small boards are won
        self.meta_board = [[0 for _ in range(3)] for _ in range(3)]
        
        # Track last move to determine next valid board
        self.last_move = None
        
        # Meta winner
        self.winner = 0  # 0 if no winner, 1 for white, -1 for black, 2 for draw
    
    def __getitem__(self, index):
        return self.boards[index]
    
    def get_board_status(self, big_x, big_y):
        return self.meta_board[big_x][big_y]
    
    def _get_next_board_coords(self):
        # If there's no last move (start of game), any board is valid
        if self.last_move is None:
            return None
        
        # Extract small coordinates from last move
        _, _, small_x, small_y = self.last_move
        
        # If the corresponding board is full or won, any board is valid
        if not self.boards[small_x][small_y].has_legal_moves():
            return None
        
        return small_x, small_y
    
    def get_legal_moves(self, color):
        if self.winner != 0:
            return []
        
        next_board = self._get_next_board_coords()
        moves = []
        
        # If next_board is None, player can play in any non-won small board
        if next_board is None:
            for big_y in range(3):
                for big_x in range(3):
                    if self.meta_board[big_x][big_y] == 0:
                        small_moves = self.boards[big_x][big_y].get_legal_moves()
                        for small_x, small_y in small_moves:
                            moves.append((big_x, big_y, small_x, small_y))
        else:
            big_x, big_y = next_board
            if self.meta_board[big_x][big_y] == 0:
                small_moves = self.boards[big_x][big_y].get_legal_moves()
                for small_x, small_y in small_moves:
                    moves.append((big_x, big_y, small_x, small_y))
        
        return moves
    
    def has_legal_moves(self):
        return len(self.get_legal_moves(1)) > 0  # Color doesn't matter here
    
    def is_meta_win(self, color):
        # Check horizontals
        for y in range(3):
            if self.meta_board[0][y] == self.meta_board[1][y] == self.meta_board[2][y] == color:
                return True
        
        # Check verticals
        for x in range(3):
            if self.meta_board[x][0] == self.meta_board[x][1] == self.meta_board[x][2] == color:
                return True
        
        # Check diagonals
        if self.meta_board[0][0] == self.meta_board[1][1] == self.meta_board[2][2] == color:
            return True
        if self.meta_board[0][2] == self.meta_board[1][1] == self.meta_board[2][0] == color:
            return True
        
        return False
    
    def is_meta_board_full(self):
        for y in range(3):
            for x in range(3):
                if self.meta_board[x][y] == 0:
                    return False
        return True
    
    def execute_move(self, move, color):
        big_x, big_y, small_x, small_y = move
        
        # Make sure the move is valid based on the last move
        next_board = self._get_next_board_coords()
        if next_board is not None:
            assert (big_x, big_y) == next_board, "Must play in the board specified by the last move"
        
        # Execute move on the small board
        self.boards[big_x][big_y].execute_move((small_x, small_y), color)
        
        # Update meta board if the small board was just won
        if self.boards[big_x][big_y].winner != 0:
            self.meta_board[big_x][big_y] = self.boards[big_x][big_y].winner
        
        # Check if the game is won on the meta level
        if self.is_meta_win(color):
            self.winner = color
        elif self.is_meta_board_full():
            self.winner = 2  # Draw
        
        # Update last move
        self.last_move = move
    
    def get_state_matrix(self):
        """Return 9x9 matrix representation of the board for neural network input"""
        state = [[0 for _ in range(9)] for _ in range(9)]
        
        for big_y in range(3):
            for big_x in range(3):
                for small_y in range(3):
                    for small_x in range(3):
                        # Convert to 9x9 coordinates
                        x = big_x * 3 + small_x
                        y = big_y * 3 + small_y
                        state[x][y] = self.boards[big_x][big_y][small_x][small_y]
        
        return state