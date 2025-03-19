import numpy as np

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, 1)
        number_of_valid_moves = np.sum(valid_moves)
        a = None
        randomNumber = np.random.randint(number_of_valid_moves)
        for i in range(len(valid_moves)):
            if valid_moves[i]:
                if randomNumber == 0:
                    a = i
                    break
                randomNumber -= 1
        assert a is not None
        return a

class HumanUltimateTicTacToePlayer():
    def __init__(self, game, name="Human"):
        self.game = game

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, 1)
        print("Valid moves:")
        print(valid_moves)
        print("Enter your move, big row and column, then small row and column (0-2):")
        while True:
            big_row = int(input("Big row: "))
            big_col = int(input("Big column: "))
            small_row = int(input("Small row: "))
            small_col = int(input("Small column: "))
            if big_row < 0 or big_row > 2 or big_col < 0 or big_col > 2:
                print("Invalid big board coordinates (must be 0-2)")
                continue
            if small_row < 0 or small_row > 2 or small_col < 0 or small_col > 2:
                print("Invalid small board coordinates (must be 0-2)")
                continue
            # Convert to action index
            action = big_row * 27 + big_col * 3 + small_row * 9 + small_col
            if action < 0 or action >= len(valid_moves):
                print("Invalid action index")
                continue
            if valid_moves[action]:
                break
            else:
                print("Invalid move")
        return action