###############################################################################
#
# Copyright (c) 2018, Henrique Morimitsu,
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# #############################################################################

import numpy as np
import tensorflow as tf

from game_manager import GameManager
from moku_state import MokuState
from net import (NaiveNet, ResNet)

tfe = tf.contrib.eager


class MokuManager(GameManager):
    """ Implements functions for handling logic related to the rules of the
    Moku game. Moku game is a generalization of games where two players
    alternatingly put one piece on the board, and the objective is to create
    a line of N pieces of the same color. Examples of Moku games are:
    gomoku (5-in-a-row), tic-tac-toe and also connect 4. The Moku game includes
    a parameter drop_mode, which indicates if the played pieces drop to the
    bottom (like in connect 4) or not.

    Args:
      drop_mode: boolean: indicating if the game is played in drop mode.
      moku_size: int: the number of same pieces in a line that are
        required to win the game.
      board_size: tuple: two values indicating number of rows and number of
        columns of the board.
      tf_device: string: tensorflow string for tf.device to be used.
      num_res_layers: int: number of residual blocks in the CNN.
      num_channels: int: number of channels in the CNN layers.
      ckpt_path: string or None: path to the checkpoint file to be loaded.
      replace_unloaded_resnet_by_naivenet: boolean: if True, when ckpt_path
        is None, a NaiveNet is used instead of a ResNet with random weights.
    """
    def __init__(self,
                 drop_mode,
                 moku_size,
                 board_size,
                 tf_device,
                 num_res_layers,
                 num_channels,
                 ckpt_path=None,
                 replace_unloaded_resnet_by_naivenet=True):
        self.drop_mode = drop_mode
        self.moku_size = moku_size
        self.board_size = board_size
        self.tf_device = tf_device
        self.num_res_layers = num_res_layers
        self.num_channels = num_channels
        self.initialize_resnet(
            ckpt_path, num_res_layers, num_channels,
            replace_unloaded_resnet_by_naivenet)
        self.h_filter, self.v_filter, self.d_filter = \
            self._initialize_filters()

    def augment_samples(self,
                        state_batch_tf,
                        policy_batch_tf,
                        value_prior_batch_tf):
        """ Performs data augmentation on the training samples by applying
        random flipping.

        Args:
            state_batch_tf: tf.Tensor NCHW: batch of states of the game.
            policy_batch_tf: tf.Tensor N: batch of policy labels.
            value_prior_batch_tf: tf.Tensor N: batch of value labels.

        Returns:
            tf.Tensor: the same states transformed by the augmentation.
            tf.Tensor: the same policies transformed by the augmentation.
            tf.Tensor: the same values transformed by the augmentation.
        """
        if self.drop_mode:
            xrand = tf.random_uniform([])
            state_batch_tf, policy_batch_tf = tf.cond(
                tf.less(xrand, 0.5),
                lambda: (state_batch_tf, policy_batch_tf),
                lambda: (
                    tf.reverse(state_batch_tf, axis=[2]),
                    self.board_size[1] - policy_batch_tf - 1))
        else:
            policy_row_tf = policy_batch_tf / self.board_size[1]
            policy_row_tf = tf.cast(policy_row_tf, tf.int64)
            policy_col_tf = policy_batch_tf % self.board_size[1]

            xrand = tf.random_uniform([])
            state_batch_tf, policy_row_tf, policy_col_tf = tf.cond(
                tf.less(xrand, 0.5),
                lambda: (state_batch_tf, policy_row_tf, policy_col_tf),
                lambda: (
                    tf.transpose(state_batch_tf, [0, 2, 1]),
                    policy_col_tf,
                    policy_row_tf))
            xrand = tf.random_uniform([])
            state_batch_tf, policy_row_tf = tf.cond(
                tf.less(xrand, 0.5),
                lambda: (state_batch_tf, policy_row_tf),
                lambda: (
                    tf.reverse(state_batch_tf, axis=[1]),
                    self.board_size[0] - policy_row_tf - 1))
            xrand = tf.random_uniform([])
            state_batch_tf, policy_col_tf = tf.cond(
                tf.less(xrand, 0.5),
                lambda: (state_batch_tf, policy_col_tf),
                lambda: (
                    tf.reverse(state_batch_tf, axis=[2]),
                    self.board_size[1] - policy_col_tf - 1))

            policy_batch_tf = self.board_size[1]*policy_row_tf + policy_col_tf
        return state_batch_tf, policy_batch_tf, value_prior_batch_tf

    def get_iwinner(self, state_batch):
        """ Returns the index of the winner in each state of the batch.

        Args:
          state_batch: np.array NCHW: a batch of game states.

        Returns:
          np.array: a ternary vector whose values are:
            -1 if there is no winner (it is a draw, or the game is unfinished),
            0 if player 1 won,
            1 if player 2 won.
        """
        board_batch = state_batch[:, :2]
        with tf.device(self.tf_device):
            board_tf = tf.constant(board_batch, tf.float32)
        board_tf = tf.transpose(board_tf, [0, 2, 3, 1])

        h_count = tf.reshape(tf.nn.depthwise_conv2d(
            board_tf, self.h_filter, [1, 1, 1, 1], 'VALID'),
            [board_batch.shape[0], -1, 2])
        v_count = tf.reshape(tf.nn.depthwise_conv2d(
            board_tf, self.v_filter, [1, 1, 1, 1], 'VALID'),
            [board_batch.shape[0], -1, 2])
        d_count = tf.nn.depthwise_conv2d(
            board_tf, self.d_filter, [1, 1, 1, 1], 'VALID')
        d_count = tf.reshape(
            d_count,
            [d_count.get_shape()[0], d_count.get_shape()[1],
             d_count.get_shape()[2], 2, 2])
        d_count = tf.transpose(d_count, [0, 1, 2, 4, 3])
        d_count = tf.reshape(d_count, [board_batch.shape[0], -1, 2])
        max_count = tf.cast(tf.reduce_max(tf.concat(
            [h_count, v_count, d_count], axis=1), axis=1), tf.int32)
        winner_flags = tf.where(
            tf.greater_equal(max_count, self.moku_size),
            tf.ones_like(max_count), tf.zeros_like(max_count))
        second_player_flags = 2 * winner_flags[:, 1]
        iwinners = tf.stack([winner_flags[:, 0], second_player_flags], axis=1)
        iwinners = tf.reduce_max(iwinners, axis=1) - 1
        return iwinners.numpy()

    def has_moves_remaining(self, state_batch):
        """ Indicates if each state from the batch has remaining valid moves.

        Args:
          state_batch: np.array NCHW: a batch of game states.

        Returns:
          np.array: a binary vector indicating whether there is at least
            one valid move to be played at each state.
        """
        free_positions = self.valid_moves(state_batch)
        free_positions = free_positions.reshape((free_positions.shape[0], -1))
        return np.sum(free_positions, axis=1) > 0

    def initial_state(self):
        """ Returns a state representing the initial configuration of the board
        before the game starts.

        Returns:
          MokuState: the initial state of the moku game
        """
        return MokuState(self.board_size)

    def initialize_resnet(self,
                          ckpt_path,
                          num_res_layers=None,
                          num_channels=None,
                          replace_unloaded_resnet_by_naivenet=True):
        """ Initializes a residual network.

        Args:
          ckpt_path: string: path to the checkpoint to be loaded.
          tf_device: string: tensorflow string for tf.device to be used.
          num_res_layers: int: number of residual blocks in the CNN.
          num_channels: int: number of channels in the CNN layers.
          replace_unloaded_resnet_by_naivenet: boolean: if True, when ckpt_path
            is None, a NaiveNet is used instead of a ResNet with random
            weights.

        Returns:
          tf.keras.Model: the initialized network.
        """
        if ckpt_path is not None or not replace_unloaded_resnet_by_naivenet:
            if num_channels is None:
                num_channels = self.num_channels
            if num_res_layers is None:
                num_res_layers = self.num_res_layers
            self.net = ResNet(
                self._num_possible_moves(), num_res_layers, num_channels)
            print('Created resnet with %d res layers and %d channels' %
                  (num_res_layers, num_channels))
            if ckpt_path is not None:
                try:
                    checkpoint = tfe.Checkpoint(net=self.net)
                    checkpoint.restore(ckpt_path)
                    print('Loaded resnet params from: ' + ckpt_path)
                except tf.errors.NotFoundError:
                    pass
        else:
            self.net = NaiveNet(self._num_possible_moves())
            print('Created naive net')

    def invalid_moves(self, state_batch):
        """ Indicates which of the possible moves at each state from the batch
        are illegal according to the rules of the game.

        Args:
          state_batch: np.array NCHW: a batch of game states.

        Returns:
          np.array: a binary tensor indicating if each possible move is
            invalid.
        """
        board_batch = state_batch[:, :2]
        if self.drop_mode:
            col_stones = np.sum(board_batch, axis=(1, 2))
            occupied_positions = \
                (col_stones >= board_batch.shape[2]).astype(np.uint8)
        else:
            occupied_positions = np.max(board_batch, axis=1)
        return occupied_positions

    def is_over(self, state_batch):
        """ Indicates if each state from the batch represents a finished game.

        Args:
          state_batch: np.array NCHW: a batch of game states.

        Returns:
          np.array: a binary vector indicating whether the game at each
          state has reached a finished state.
        """
        return np.maximum(self.get_iwinner(state_batch) >= 0,
                          (1 - self.has_moves_remaining(state_batch)))

    def max_num_moves(self):
        """ Returns the maximum number of moves that can be played from the start
        the end of the game. In Moku games, it is equal to the size of the
        board.

        Returns:
            int: maximum number of moves
        """
        return self.board_size[0] * self.board_size[1]

    def predict(self, state_batch):
        """ Predict the policies and values for each state in the batch.

        Args:
          state_batch: np.array NCHW: a batch of game states.

        Returns:
          np.array NHW or NW: the predicted policies, if drop_mode=True, then
            it is an NW tensor, otherwise it is an NHW tensor.
          np.array: a vector with the predicted values.
        """
        occupied_positions = self.invalid_moves(state_batch)
        with tf.device(self.tf_device):
            occupied_positions_tf = tf.constant(occupied_positions, tf.float32)
            policy_priors, value_priors = \
                self.net(tf.constant(state_batch, tf.float32), training=False)

            reduction_axis = 1
            if not self.drop_mode:
                reduction_axis = (1, 2)
                policy_priors = tf.reshape(
                    policy_priors,
                    [state_batch.shape[0], state_batch.shape[2],
                     state_batch.shape[3]])

            policy_priors = tf.exp(
                policy_priors - tf.reduce_max(
                    policy_priors, axis=reduction_axis, keepdims=True))

            policy_priors = tf.where(
                tf.equal(occupied_positions_tf, 1),
                tf.zeros_like(policy_priors), policy_priors)
            policy_priors = policy_priors / tf.reduce_sum(
                policy_priors, axis=reduction_axis, keepdims=True)
            policy_priors = tf.where(
                tf.equal(occupied_positions_tf, 1),
                -100 * tf.ones_like(policy_priors),
                policy_priors)  # Put far from zero to avoid precision problems
        policy_priors = policy_priors.numpy()
        value_priors = value_priors.numpy()
        value_priors = self._replace_winning_value_priors(
            state_batch, value_priors[:, 0])

        return policy_priors, value_priors

    def update_state(self, state, imove):
        """ Given a state and a valid move index, plays the move at the given
        state and update it according to the rules of the game.

        Args:
          state: np.array CHW: a game state.

        Returns:
          np.array CHW, the new state after the move imove is played.
        """
        if self.drop_mode:
            col = imove
            row = np.sum(state.state[:2], axis=(0, 1))[col]
        else:
            row = imove // state.state.shape[2]
            col = imove % state.state.shape[2]
        iplayer = state.state[-1, 0, 0]
        state.state[iplayer, row, col] = 1
        state.state[-1, :, :] = (iplayer + 1) % 2
        return state

    def valid_moves(self, state_batch):
        """ Indicates which of the possible moves at each state from the batch
        are legal according to the rules of the game.

        Args:
          state_batch: np.array NCHW: a batch of game states.

        Returns:
          np.array: a binary tensor indicating if each possible move is valid.
        """
        occupied_positions = self.invalid_moves(state_batch)
        free_positions = np.ones_like(occupied_positions) - occupied_positions
        return free_positions

    def _replace_winning_value_priors(self, state_batch, value_priors_batch):
        """ Checks the batch of states to find out which games have winners
        and replaces the predicted values by the expected values for the
        winner.

        Args:
          state_batch: np.array NCHW: a batch of game states.
          value_priors_batch: np.array: a vector with the predicted values.

        Returns:
          np.array: the updated vector of values.
        """
        iwinners = self.get_iwinner(state_batch)
        has_p1_won = iwinners == 0
        has_p2_won = iwinners == 1
        iplayer_batch = state_batch[:, -1, 0, 0]
        p1v = iplayer_batch == 0
        p2v = iplayer_batch == 1
        p1_wins = has_p1_won * p1v
        p2_wins = has_p2_won * p2v
        wins = np.maximum(p1_wins, p2_wins)
        p1_losses = has_p2_won * p1v
        p2_losses = has_p1_won * p2v
        losses = -1 * np.maximum(p1_losses, p2_losses)
        decided = \
            (wins + losses).astype(np.float32) * type(self).DECIDED_MULTIPLIER
        undecided_idx = np.where(decided == 0)
        decided[undecided_idx] = value_priors_batch[undecided_idx]
        return decided

    def _initialize_filters(self):
        """ Initialize the convolution kernels that are used to check if there
        a winner.

        Returns:
          tf.Tensor, kernel for finding horizontal lines.
          tf.Tensor, kernel for finding vertical lines.
          tf.Tensor, kernel for finding diagonal lines.
        """
        with tf.device(self.tf_device):
            h_filter = tf.ones([1, self.moku_size, 1, 1], tf.float32)
            h_filter = tf.concat([h_filter, h_filter], axis=2)
            v_filter = tf.ones([self.moku_size, 1, 1, 1], tf.float32)
            v_filter = tf.concat([v_filter, v_filter], axis=2)
            d1_filter = tf.scatter_nd(
                tf.constant([[i, i, 0, 0] for i in range(self.moku_size)]),
                tf.constant([1.0 for _ in range(self.moku_size)]),
                tf.constant([self.moku_size, self.moku_size, 1, 1]))
            d2_filter = tf.scatter_nd(
                tf.constant([[i, self.moku_size-i-1, 0, 0]
                             for i in range(self.moku_size)]),
                tf.constant([1.0 for _ in range(self.moku_size)]),
                tf.constant([self.moku_size, self.moku_size, 1, 1]))
            d_filter = tf.concat([d1_filter, d2_filter], axis=3)
            d_filter = tf.concat([d_filter, d_filter], axis=2)
        return h_filter, v_filter, d_filter

    def _num_possible_moves(self):
        """ Returns the total number of moves which are possible at any state
        of the game, regardless of being valid or not. E.g., there are 9
        possible moves in tic-tac-toe, because it is played on a 3x3 board.

        Returns:
          int, the number of possible moves.
        """
        if self.drop_mode:
            return self.board_size[1]
        else:
            return self.board_size[0] * self.board_size[1]
