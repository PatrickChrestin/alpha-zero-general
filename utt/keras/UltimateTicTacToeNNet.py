import sys
sys.path.append('..')
from utils import *

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import numpy as np

"""
NeuralNet for the game of Ultimate TicTacToe.
Based on the TicTacToeNNet by Evgeny Tyurin.
"""
class UltimateTicTacToeNNet:
    def __init__(self, game, args):
        self.board_x = 9
        self.board_y = 9
        self.action_size = 82
        self.board_channels = 1  # 1 channel: current state of the board

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))

        # Reshape to fit Conv2D layer's expected input shape
        x_image = Reshape((self.board_x, self.board_y, self.board_channels))(self.input_boards)

        # Convolutional block 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(x_image)))

        # Convolutional block 2
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv1)))

        # Convolutional block 3
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv2)))

        # Convolutional block 4
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv3)))

        # Residual connections
        h_conv5 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv4))) + h_conv3
        h_conv6 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv5))) + h_conv4

        # Policy head
        h_conv_policy = Activation('relu')(BatchNormalization(axis=3)(Conv2D(2, 1, padding='same')(h_conv6)))
        h_conv_policy_flat = Flatten()(h_conv_policy)
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(h_conv_policy_flat)

        # Value head
        h_conv_value = Activation('relu')(BatchNormalization(axis=3)(Conv2D(1, 1, padding='same')(h_conv6)))
        h_conv_value_flat = Flatten()(h_conv_value)
        h_value = Dense(256, activation='relu')(h_conv_value_flat)
        self.v = Dense(1, activation='tanh', name='v')(h_value)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))
    def predict(self, board):
        # Prepare input
        board = board[np.newaxis, np.newaxis, :, :]

        # Run prediction
        pi, v = self.model.predict(board)

        return pi[0], v[0]

    def fit(self, examples):
        # Prepare data
        input_boards = np.array([example[0] for example in examples])
        target_pis = np.array([example[1] for example in examples])
        target_vs = np.array([example[2] for example in examples])

        # Train model
        self.model.fit(
            x=input_boards,
            y=[target_pis, target_vs],
            batch_size=32,
            epochs=10
        )
