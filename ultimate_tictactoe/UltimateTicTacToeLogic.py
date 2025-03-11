'''
Board class for the game of Ultimate TicTacToe.
Default board size is 9x9.
Board data:
  1=white(O), -1=black(X), 0=empty
  first dim is row , 2nd is column:
     pieces[0][0] is the top left square,
     pieces[9][0] is the bottom left square,
     pieces[9][9] is the bottom right square,
     pieces[0][9] is the top right square,
Squares are stored and manipulated as (x,y) tuples.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Mar 10, 2025.

Based on the board for the game of Othello by Eric P. Nichols.

'''
class Board():

    def __init__(self, n = 9):
        "Set up initial board configuration."
        self.n = n
        # Create the empty board array.
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n

        # Keep track of the last move made
        self.last_move = (None,None)

        """
        Defines the big 3x3 board
        0 is playing
        1 is white
        -1 is black
        2 is draw
        """
        self.grids = [None]*3
        for i in range(3):
            self.grids[i] = [None]*3
            for j in range(3):
                self.grids[i][j] = 0

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def get_legal_moves(self, color = 1):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        @param color not used and came from previous version.        
        """
        moves = set()  # stores the legal moves.

        if self.last_move == (None,None):
            for y in range(self.n):
                for x in range(self.n):
                    if self.pieces[x][y]==0:
                        newmove = (x,y)
                        moves.add(newmove)
            return list(moves)

        (lx,ly) = self.last_move
        # index of new board
        nx, ny = lx%3, ly%3
        bx, by = lx//3, ly//3
        if (self.grids[nx][ny] != 0):
            # can only play in empty cells of grids that are not finished
            for i in range(3):
                for j in range(3):
                    if self.grids[j][i] == 0:
                        for y in range(3*i, 3*i+3):
                            for x in range(3*j, 3*j+3):
                                if self.pieces[x][y]==0:
                                    newmove = (x,y)
                                    moves.add(newmove)
            return list(moves)
        else:
            for y in range(3*ny, 3*ny+3):
                for x in range(3*nx, 3*nx+3):
                    if self.pieces[x][y]==0:
                        newmove = (x,y)
                        moves.add(newmove)
            return list(moves)

    def is_tie(self):
        for i in range(3):
            for j in range(3):
                if self.grids[i][j] == 0:
                    return False
        return True
    
    def is_win(self, color):
        """
        Check whether the given player has collected a triplet in any direction in the grids
        @param color (1=white,-1=black)
        """
        win = 3
        # check y-strips
        for y in range(3):
            count = 0
            for x in range(3):
                if self.grids[x][y]==color:
                    count += 1
            if count==win:
                return True
        # check x-strips
        for x in range(3):
            count = 0
            for y in range(3):
                if self.grids[x][y]==color:
                    count += 1
            if count==win:
                return True
        # check two diagonal strips
        count = 0
        for d in range(3):
            if self.grids[d][d]==color:
                count += 1
        if count==win:
            return True
        count = 0
        for d in range(3):
            if self.grids[d][2-d]==color:
                count += 1
        if count==win:
            return True
        return False
    
    def has_legal_moves(self):
        if self.is_win(1) or self.is_win(-1) or self.is_tie():
            return False
        legal_moves = self.get_legal_moves()
        return len(legal_moves) > 0

    def execute_move(self, move, color):
        """Perform the given move on the board; 
        color gives the color pf the piece to play (1=white,-1=black)
        """

        (x,y) = move

        # Add the piece to the empty square.
        nx, ny = x//3, y//3
        assert self.pieces[x][y] == 0
        assert self.grids[nx][ny] == 0
        if move not in self.get_legal_moves():
            print("Illegal move")
            print(move)
            print(self.get_legal_moves())
            assert move in self.get_legal_moves()
        self.pieces[x][y] = color
        self.last_move = (x,y)

        self.grids[nx][ny] = self.get_winner_of_board(move, color)



    def get_winner_of_board(self, move, color):
        """
        Check if the board is won by the player who made the move
        """
        (x,y) = move
        nx, ny = x//3, y//3
        win = 3
        # check y-strips
        count = 0
        for i in range(3):
            if self.pieces[3*nx+i][y]==color:
                count += 1
        if count==win:
            return color
        # check x-strips
        count = 0
        for i in range(3):
            if self.pieces[x][3*ny+i]==color:
                count += 1
        if count==win:
            return color
        # check two diagonal strips
        count = 0
        for i in range(3):
            if self.pieces[3*nx+i][3*ny+i]==color:
                count += 1
        if count==win:
            return color
        count = 0
        for i in range(3):
            if self.pieces[3*nx+i][3*ny+2-i]==color:
                count += 1
        if count==win:
            return color
        # check if the board is full
        count = 0
        for i in range(3):
            for j in range(3):
                if self.pieces[3*nx+i][3*ny+j] == 0:
                    return 0
        return 2
    
    def printBoard(self):
        #pieces are given as 81 cells, print 9 per line
        # print numbers one to nine on top
        print("  ", end = "")
        for i in range(9):
            print(i, end = " ")
            if ((i+1)%3 == 0 and i != 8):
                print(" ", end = " ")
        print()
        for i in range(9):
            if (i%3 == 0 and i != 0):
                print("-----------------------")
            print(i, end = " ")
            for j in range(9):
                if self.pieces[j][i] == 1:
                    print("O", end = " ")
                elif self.pieces[j][i] == -1:
                    print("X", end = " ")
                else:
                    print(".", end = " ")
                if ((j+1)%3 == 0 and j != 0 and j != 8):
                    print("|", end = " ")
            print()

        

# write a test function to check the correctness of the code

def test():
    return
    performed_moves = set()

    b = Board()
    b.execute_move((2,0), 1)
    performed_moves.add((2,0))
    # check if none of the performed moves is in the valid moves
    assert all([move not in b.get_legal_moves() for move in performed_moves])
    b.execute_move((6,0), -1)
    performed_moves.add((6,0))
    assert all([move not in b.get_legal_moves() for move in performed_moves])
    b.execute_move((2,1), 1)
    performed_moves.add((2,1))
    assert all([move not in b.get_legal_moves() for move in performed_moves])
    b.execute_move((6,3), -1)
    performed_moves.add((6,3))
    assert all([move not in b.get_legal_moves() for move in performed_moves])
    b.execute_move((2,2), 1)
    performed_moves.add((2,2))
    assert all([move not in b.get_legal_moves() for move in performed_moves])
    # first grid is won by 1
    assert b.grids[0][0] == 1
    b.execute_move((6,6), -1)
    performed_moves.add((6,6))
    assert all([move not in b.get_legal_moves() for move in performed_moves])
    b.printBoard()
    print(len(b.get_legal_moves()))
    assert len(b.get_legal_moves()) == 69
    b.execute_move((2,3), 1)
    performed_moves.add((2,3))
    assert all([move not in b.get_legal_moves() for move in performed_moves])
    b.execute_move((6,1), -1)
    performed_moves.add((6,1))
    assert all([move not in b.get_legal_moves() for move in performed_moves])
    b.execute_move((2,4), 1)
    performed_moves.add((2,4))
    assert all([move not in b.get_legal_moves() for move in performed_moves])
    b.execute_move((6,4), -1)
    performed_moves.add((6,4))
    assert all([move not in b.get_legal_moves() for move in performed_moves])
    b.execute_move((2,5), 1)
    performed_moves.add((2,5))
    assert all([move not in b.get_legal_moves() for move in performed_moves])
    # second grid is won by 1
    b.printBoard()
    assert b.grids[0][1] == 1
    b.execute_move((6,7), -1)
    performed_moves.add((6,7))
    assert all([move not in b.get_legal_moves() for move in performed_moves])
    b.printBoard()
    b.execute_move((2,6), 1)
    performed_moves.add((2,6))
    assert all([move not in b.get_legal_moves() for move in performed_moves])
    b.execute_move((6,2), -1)
    performed_moves.add((6,2))
    assert all([move not in b.get_legal_moves() for move in performed_moves])
    b.execute_move((2,7), 1)
    performed_moves.add((2,7))
    assert all([move not in b.get_legal_moves() for move in performed_moves])
    b.execute_move((6,5), -1)
    performed_moves.add((6,5))
    assert all([move not in b.get_legal_moves() for move in performed_moves])
    b.execute_move((2,8), 1)
    performed_moves.add((2,8))
    assert all([move not in b.get_legal_moves() for move in performed_moves])
    # third grid is won by 1
    assert b.grids[0][2] == 1
    # no more moves left
    assert b.has_legal_moves() == False
    assert b.is_win(1) == True
    assert b.is_win(-1) == False
    assert b.is_tie() == False
    print("All tests passed")
test()