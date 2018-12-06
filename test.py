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

""" Play against a network.

Some arguments can be passed by command line (see --help) while other can be
modified in the config.py file.

To run:

$ python test.py --mode=tictactoe

Or choose another valid mode (see --help).
"""

import argparse
import numpy as np
import os.path as osp
import sys
import tensorflow as tf

import utils
from game_manager_io import GameManagerIO
from mcts import MCTS

tfe = tf.contrib.eager

config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config_proto)


def main(argv):
    args = parse_args()

    valid_modes_list = utils.get_valid_game_modes()
    valid_modes_string = utils.get_valid_game_modes_string()
    if args.mode not in valid_modes_list:
        print('Invalid game mode informed. Please inform a mode with ' +
              '--mode=mode_name, where mode_name is one of the following ' +
              '{%s}' % valid_modes_string)
        sys.exit()

    gconf = utils.get_game_config(args.mode, 'test')

    if args.game_type == 'moku':
        (game_config_string, game_manager_module, game_manager_kwargs,
            game_manager_io_module, game_manager_io_kwargs) = \
                utils.generate_moku_manager_params(
                    gconf.drop_mode, gconf.moku_size, gconf.board_size,
                    args.gpu_id, gconf.num_res_layers, gconf.num_channels)
    else:
        raise NotImplementedError(
            'Game type %s is not supported.' % args.game_type)

    train_dir = osp.join('train_files', game_config_string)

    ckpt_path = utils.get_checkpoint_path(train_dir, args.num_iters_ckpt)

    game_manager_kwargs['ckpt_path'] = ckpt_path

    gm_module = __import__(game_manager_module[0])
    gm_class = getattr(gm_module, game_manager_module[1])
    game_manager = gm_class(**game_manager_kwargs)

    gmio_module = __import__(game_manager_io_module[0])
    gmio_class = getattr(gmio_module, game_manager_io_module[1])
    game_manager_io = gmio_class(**game_manager_io_kwargs)

    state = game_manager.initial_state()

    mctss = [MCTS(game_manager, gconf.max_simulations_per_move, gconf.cpuct,
             gconf.virtual_loss, state, gconf.root_noise_weight,
             gconf.dirichlet_noise_param, gconf.eval_batch_size,
             game_manager_kwargs['tf_device'])]

    iplayer = 0
    iplay = 0
    moves = []
    last_played_imove = None
    while not game_manager.is_over(state.state[np.newaxis])[0]:
        imove = None
        if iplay < gconf.num_relaxed_turns:
            turn_temperature = 1.0
        else:
            turn_temperature = gconf.move_temperature
        imc = iplayer % len(mctss)
        print('===== New turn =====')
        game_manager_io.print_board(state, last_played_imove)
        if args.iuser == 2 or iplayer == args.iuser:
            # User types a move
            imove = game_manager_io.get_input(state)
        if imove == GameManagerIO.IEXIT:
            break
        if imove == GameManagerIO.ICOMPUTER_MOVE or \
                (args.iuser != 2 and iplayer != args.iuser):
            # Computer chooses a move
            stats = mctss[imc].simulate(state, gconf.max_seconds_per_move)
            if args.show_mcts:
                print('MCTS stats')
                game_manager_io.print_stats(stats)
                print()

            if args.show_win_prob or imove == GameManagerIO.ICOMPUTER_MOVE:
                with tf.device(game_manager_kwargs['tf_device']):
                    _, value_prior = game_manager.predict(
                        tf.constant(state.state[np.newaxis], tf.float32))
                    win_prob = (value_prior[0] + 1.0) / 2.0
                    print('Estimated win probability: %.03f\n' % win_prob)

            if args.show_move_prob or imove == GameManagerIO.ICOMPUTER_MOVE:
                print('Move probabilities:')
                game_manager_io.print_stats_on_board(stats, 1)
                print()

            if args.show_move_prob_temp:
                print('Move probabilities with temperature ' +
                      '%.1e' % turn_temperature)
                game_manager_io.print_stats_on_board(stats, turn_temperature)
                print()

            if imove == GameManagerIO.ICOMPUTER_MOVE:
                # If user asked for computer prediction,
                # escape before actually choosing a move
                continue

            imove, _ = mctss[imc].choose_move(turn_temperature)
            moves.append((imove, iplayer))
        last_played_imove = imove
        state = game_manager.update_state(state, last_played_imove)
        iplayer = (iplayer + 1) % 2
        for imc2 in range(len(mctss)):
            mctss[imc2].update_root(last_played_imove,  state)
        iplay += 1

    if imove == GameManagerIO.IEXIT:
        print('Game unfinished')
    else:
        game_manager_io.print_board(state, imove)
        iwinner = game_manager.get_iwinner(state.state[np.newaxis])
        if iwinner < 0:
            print('DRAW')
        else:
            if args.iuser == 2:
                print('Player %d WON.' % (iwinner + 1))
            elif iwinner == args.iuser:
                print('You WON!')
            else:
                print('You LOST!')


def parse_args():
    parser = argparse.ArgumentParser()
    valid_modes = utils.get_valid_game_modes_string()
    parser.add_argument(
        '--mode',
        help=('A valid game mode name. valid modes are {%s}.' % valid_modes),
        default=None
    )
    parser.add_argument(
        '--gpu_id',
        help=('GPU id to use, or -1 to use the CPU.'),
        default=0,
        type=int
    )
    parser.add_argument(
        '--game_type',
        help=('Type is a more general term which may include many game ' +
              'modes. For example, moku is the type of tictactoe, connect4 ' +
              'and gomoku modes.'),
        default='moku'
    )
    parser.add_argument(
        '--iuser',
        help=('Index of the user, 0 to play first and 1 to play second. ' +
              'Or you can also use -1 to let the computer play as both ' +
              'players or 2 if you want to play as both players.'),
        default=0,
        type=int
    )
    parser.add_argument(
        '--num_iters_ckpt',
        help=('Number of iterations in the checkpoint to load. ' +
              'e.g. if the file is called moku3_3x3_1000.ckpt, type 1000. ' +
              'Use -1 to load the latest checkpoint or 0 to use a naive ' +
              'network.'),
        default=-1,
        type=int
    )
    parser.add_argument(
        '-sm',
        '--show_mcts',
        help=('If set, the MCTS stats for the current state will ' +
              'be displayed.'),
        nargs='?',
        const=True,
        default=False,
        type=bool
    )
    parser.add_argument(
        '-sp',
        '--show_move_prob',
        help=('If set, the probabilities of playing at each position will ' +
              'be displayed.'),
        nargs='?',
        const=True,
        default=False,
        type=bool
    )
    parser.add_argument(
        '-spt',
        '--show_move_prob_temp',
        help=('If set, the probabilities of playing at each position ' +
              'rebalanced by the temperature will be displayed.'),
        nargs='?',
        const=True,
        default=False,
        type=bool
    )
    parser.add_argument(
        '-sw',
        '--show_win_prob',
        help=('If set, the winning probability estimated by the network ' +
              'will be displayed.'),
        nargs='?',
        const=True,
        default=False,
        type=bool
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    tf.app.run()
