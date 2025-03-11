import sys
import os

import logging
import coloredlogs
import ultimate_tictactoe.UltimateTicTacToeGame as Game
from ultimate_tictactoe.UltimateTicTacToeGame import UltimateTicTacToeGame
from ultimate_tictactoe.keras.NNet import NNetWrapper as nn
from utils import *
import numpy as np

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')

args = dotdict({
    'numMCTSSims': 25,
    'cpuct': 1.0,
    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp_uttt', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

def play_game():
    log.info('Loading %s...', UltimateTicTacToeGame.__name__)
    g = UltimateTicTacToeGame()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Starting the game...')

    while True:
        board = g.getInitBoard()
        player = 1
        g.display(board)
        while g.getGameEnded(board, player) == 0:
            if player == 1:
                action = get_user_action(g, board)
            else:
                action = get_model_action(nnet, g, board, player)
            board, player = g.getNextState(board, player, action)
            g.display(board)
        result = g.getGameEnded(board, 1)
        if result == 1:
            print("You win!")
        elif result == -1:
            print("You lose!")
        else:
            print("It's a draw!")
        if input("Play again? (y/n): ") != 'y':
            break

def get_user_action(game, board):
    valid = game.getValidMoves(board, 1)
    while True:
        action = int(input("Enter your move (0-8): "))
        if valid[action]:
            return action
        else:
            print("Invalid move. Try again.")

def get_model_action(nnet, game, board, player):
    canonical_board = game.getCanonicalForm(board, player)
    pi, _ = nnet.predict(canonical_board)
    valid_moves = game.getValidMoves(board, player)
    pi = pi * valid_moves  # Mask invalid moves
    action = np.argmax(pi)
    return action

if __name__ == "__main__":
    play_game()
