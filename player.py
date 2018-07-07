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
import os.path as osp
import sys
import tensorflow as tf
import time

import utils
from mcts import MCTS

tfe = tf.contrib.eager


def player(player_trmanager_queue,
           plmanager_player_queue,
           ckpt_dir,
           max_simulations_per_move,
           max_seconds_per_move,
           move_temperature,
           num_relaxed_turns,
           random_move_probability,
           cpuct,
           virtual_loss,
           root_noise_weight,
           dirichlet_noise_param,
           eval_batch_size,
           game_manager_module,
           game_manager_kwargs):
    """ Starts the player process. It will continuously receive checkpoint paths
    from player_manager. Each time a checkpoint is received, a new network
    is restored and a game is played until the end based on the network
    predictions and MCTS. After the game is finished, the move order and the
    result are sent to the train_manager.

    Args:
      player_trmanager_queue: Queue: to send results to train_manager.
      plmanager_player_queue: Queue: to get checkpoint paths from the
        player_manager.
      ckpt_dir: string: the path to the directory where the checkpoint files
        are saved.
      max_simulations_per_move: int: maximum number of MCTS searches per move.
      max_seconds_per_move: float: maximum time to perform the searches per
        move.
      move_temperature: float: exponential factor to rebalance the policy and
        control its level of exploration. This value corresponds to 1 / \tau in
        [1].
      num_relaxed_turns: int: move_temperature is set to 1.0 during the first
        num_relaxed_turns turns.
      random_move_probability: float: probability of making a totally random
        move at any moment.
      cpuct: float: controls the exploration in MCTS. See c_{puct} in [1].
      virtual_loss: float: controls branch exploration when using MCTS in
        parallel mode. See [1].
      root_noise_weight: float: amount of noise to be added to policy
        predictions in the root node (first move).
      dirichlet_noise_param: float: controls the dirichlet distribution.
      eval_batch_size: int: number of leaf nodes to collect in MCTS before
        getting a prediction from the network.
      game_manager_module: list: a list with two string containing the name
        of the game manager module (file) and the name of the class inside of
        the module.
      game_manager_kwargs: dict: a dictionary of arguments and its respective
        values.

    [1] Silver, David, et al. "Mastering the game of go without human
        knowledge." Nature 550.7676 (2017): 354.
    """
    np.random.seed()

    gm_module = __import__(game_manager_module[0])
    gm_class = getattr(gm_module, game_manager_module[1])
    game_manager = gm_class(**game_manager_kwargs)

    prev_ckpt_name = -1
    while True:
        ckpt_name = plmanager_player_queue.get()
        if ckpt_name != prev_ckpt_name:
            prev_ckpt_name = ckpt_name

            ckpt_path = None
            if ckpt_name is not None:
                ckpt_path = osp.join(ckpt_dir, ckpt_name)
                game_manager.initialize_resnet(ckpt_path)
        mctss = [MCTS(
            game_manager, max_simulations_per_move, cpuct,
            virtual_loss, game_manager.initial_state(), root_noise_weight,
            dirichlet_noise_param, eval_batch_size,
            game_manager_kwargs['tf_device'])]
        moves, iwinner = play_one_game(
            game_manager, mctss, max_simulations_per_move,
            max_seconds_per_move, move_temperature, num_relaxed_turns,
            random_move_probability)
        player_trmanager_queue.put((moves, iwinner))


def play_one_game(game_manager,
                  mctss,
                  max_simulations_per_move,
                  max_seconds_per_move,
                  move_temperature,
                  num_relaxed_turns,
                  random_move_probability):
    """ Plays one game from start to end.

    Args:
      game_manager: GameManager: an object containing a manager for the current
        game.
      mctss: list: list of one or two MCTS objects to be used during the game.
        if only one is provided, both players will use the same tree.
      max_simulations_per_move: int: maximum number of MCTS searches per move.
      max_seconds_per_move: float: maximum time to perform the searches per
        move.
      move_temperature: float: exponential factor to rebalance the policy and
        control its level of exploration. See more in player.player.
      num_relaxed_turns: int: move_temperature is set to 1.0 during the first
        num_relaxed_turns turns.
      random_move_probability: float: probability of making a totally random
        move at any moment.

    Returns:
      list: list of (move_index, player_index) pairs. The list is ordered
        according to the order the moves were played in the game.
      int: index of the winning player, or -1 if there is no winner
    """
    state = game_manager.initial_state()
    for mc in mctss:
        mc.initialize_tree(state)

    iplayer = 0
    iplay = 0
    moves = []
    while not game_manager.is_over(state.state[np.newaxis])[0]:
        if iplay < num_relaxed_turns:
            turn_temperature = 1.0
        else:
            turn_temperature = move_temperature
        imc = iplayer % len(mctss)

        mctss[imc].simulate(state, max_seconds_per_move)
        imove, _ = mctss[imc].choose_move(
            turn_temperature, random_move_probability)
        moves.append((imove, iplayer))
        state = game_manager.update_state(state, imove)
        iplayer = (iplayer + 1) % 2
        for imc2 in range(len(mctss)):
            mctss[imc2].update_root(imove, state)
        iplay += 1
    iwinner = int(game_manager.get_iwinner(state.state[np.newaxis])[0])

    return moves, iwinner
