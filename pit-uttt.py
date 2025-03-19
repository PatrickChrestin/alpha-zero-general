import sys
import os
import logging
import coloredlogs
from utt.UltimateTicTacToeGame import UltimateTicTacToeGame
from Arena import Arena
from utt.UltimateTicTacToePlayers import HumanUltimateTicTacToePlayer, RandomPlayer
from utils import *
import numpy as np

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

def main():
    log.info('Loading %s...', UltimateTicTacToeGame.__name__)
    game = UltimateTicTacToeGame()
    
    # Create players
    player_type = get_player_type()
    
    if player_type == 1:
        # Human vs Human
        player1 = HumanUltimateTicTacToePlayer(game, "Player 1")
        player2 = HumanUltimateTicTacToePlayer(game, "Player 2")
        log.info('Starting Human vs Human game...')
    elif player_type == 2:
        # Human vs Random
        player1 = HumanUltimateTicTacToePlayer(game, "Human")
        player2 = RandomPlayer(game)
        log.info('Starting Human vs Random game...')
    else:
        # Random vs Random
        player1 = RandomPlayer(game)
        player2 = RandomPlayer(game)
        log.info('Starting Random vs Random game...')
    
    # Create arena
    arena = Arena(player1.play, player2.play, game, game.display)
    
    # Play a single game
    result = arena.playGame(verbose=True)
    
    # Display result
    if result == 1:
        print("Player 1 wins!")
    elif result == -1:
        print("Player 2 wins!")
    else:
        print("It's a draw!")
    
    # Ask to play again
    if input("Play again? (y/n): ").lower() == 'y':
        main()

def get_player_type():
    print("Select game mode:")
    print("1. Human vs Human")
    print("2. Human vs Random AI")
    print("3. Random AI vs Random AI")
    
    while True:
        try:
            choice = int(input("Enter your choice (1-3): "))
            if 1 <= choice <= 3:
                return choice
            else:
                print("Invalid choice. Please enter a number between 1 and 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    main()