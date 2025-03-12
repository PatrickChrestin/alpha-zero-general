import numpy as np

"""
Human and Random players for the game of Ultimate TicTacToe.

Adapted from TicTacToePlayers.py by Evgeny Tyurin.
"""

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanUltimateTicTacToePlayer():
    def __init__(self, game, player_name="Human"):
        self.game = game
        self.player_name = player_name

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        
        # Display available moves
        print(f"\n{self.player_name}'s turn (Player {'O' if board[0][0] == -1 else 'X'}):")
        
        # Get the next board to play in based on valid moves
        next_board = self._get_next_board(valid)
        
        if next_board:
            big_x, big_y = next_board
            print(f"You must play in small board at position ({big_x}, {big_y})")
        else:
            print("You can play in any available small board")
        
        while True:
            try:
                # Get input for big board coordinates
                if next_board:
                    big_x, big_y = next_board
                else:
                    print("Enter big board coordinates (big_x big_y):")
                    big_input = input()
                    big_x, big_y = [int(x) for x in big_input.split()]
                    if big_x < 0 or big_x > 2 or big_y < 0 or big_y > 2:
                        print("Invalid big board coordinates (must be 0-2)")
                        continue
                
                # Get input for small board coordinates
                print("Enter small board coordinates (small_x small_y):")
                small_input = input()
                small_x, small_y = [int(x) for x in small_input.split()]
                
                if small_x < 0 or small_x > 2 or small_y < 0 or small_y > 2:
                    print("Invalid small board coordinates (must be 0-2)")
                    continue
                
                # Convert to action index
                action = big_x * 27 + big_y * 9 + small_x * 3 + small_y
                
                if valid[action]:
                    return action
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Invalid input format. Please enter two integers separated by space.")
            except IndexError:
                print("Invalid input. Please enter coordinates between 0 and 2.")

    def _get_next_board(self, valid_moves):
        """
        Determine which small board the player must play in
        based on the valid moves available.
        """
        if sum(valid_moves[:-1]) == 0:  # No valid moves except possibly pass
            return None
            
        # Check if all valid moves are in the same big board
        big_boards = set()
        for i in range(len(valid_moves) - 1):  # Exclude pass move
            if valid_moves[i] == 1:
                big_x = i // 27
                big_y = (i % 27) // 9
                big_boards.add((big_x, big_y))
        
        if len(big_boards) == 1:
            return list(big_boards)[0]
        
        return None