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

""" Puts multiple networks to play against each other multiple times and keeps
win-loss-draw count. The networks positions are swapped after each challenge
round.

Some arguments can be passed by command line (see --help) while other can be
modified in the config.py file.

To run:

$ python challenge.py --mode=tictactoe

Or choose another valid mode (see --help).
"""

import argparse
import numpy as np
import os.path as osp
import sys
import tensorflow as tf

import utils
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

    gconf = utils.get_game_config(args.mode, 'challenge')

    if args.num_challenges > 0:
        gconf.num_challenges = args.num_challenges

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

    for i in range(len(args.num_iters_ckpt)):
        x = int(args.num_iters_ckpt[i])
        if x < 0:
            x = utils.get_last_checkpoint_number(train_dir)
        args.num_iters_ckpt[i] = x

    ckpt_paths = [utils.get_checkpoint_path(
        train_dir, x) for x in args.num_iters_ckpt]

    gmio_module = __import__(game_manager_io_module[0])
    gmio_class = getattr(gmio_module, game_manager_io_module[1])
    game_manager_io = gmio_class(**game_manager_io_kwargs)

    gm_module = __import__(game_manager_module[0])
    gm_class = getattr(gm_module, game_manager_module[1])
    ip1 = 0

    args.max_simulations_per_move = [
        int(x) for x in args.max_simulations_per_move]
    local_max_simulations_per_move = args.max_simulations_per_move
    if len(local_max_simulations_per_move) == 1:
        local_max_simulations_per_move = \
            local_max_simulations_per_move * len(args.num_iters_ckpt)
    elif len(local_max_simulations_per_move) != len(args.num_iters_ckpt):
        print('Number of arguments in max_simulations_per_move and ' +
              'num_iters_ckpt do not match. See --help for more information.')
        sys.exit()

    local_eval_batch_size = [0] * len(local_max_simulations_per_move)
    for i in range(len(local_max_simulations_per_move)):
        if local_max_simulations_per_move[i] < 1:
            local_max_simulations_per_move[i] = gconf.max_simulations_per_move
            local_eval_batch_size[i] = gconf.eval_batch_size
        else:
            local_eval_batch_size[i] = \
                int(local_max_simulations_per_move[i] / 100.0) + 1

    print('Running %d challenges for each pair of checkpoints.' %
          gconf.num_challenges)

    results = np.zeros(
        (2, len(args.num_iters_ckpt), len(args.num_iters_ckpt), 3), np.int32)
    for ichallenge in range(gconf.num_challenges):
        iend_ckpt1 = len(args.num_iters_ckpt) - 1
        if args.include_self_play:
            iend_ckpt1 += 1
        for ickpt1 in range(iend_ckpt1):
            istart_ckpt2 = ickpt1 + 1
            if args.include_self_play:
                istart_ckpt2 -= 1
            for ickpt2 in range(istart_ckpt2, len(args.num_iters_ckpt)):
                chal_ckpt_nums = [
                    args.num_iters_ckpt[ickpt1],
                    args.num_iters_ckpt[ickpt2]]
                chal_ckpt_paths = [ckpt_paths[ickpt1], ckpt_paths[ickpt2]]
                chal_max_simulations_per_move = [
                    local_max_simulations_per_move[ickpt1],
                    local_max_simulations_per_move[ickpt2]]
                chal_eval_batch_size = [
                    local_eval_batch_size[ickpt1],
                    local_eval_batch_size[ickpt2]]

                print('=====================================================')
                print('Checkpoint %d vs. %d' % tuple(chal_ckpt_nums))
                print('=====================================================')

                game_managers = []
                for i, ckpt in enumerate(chal_ckpt_paths):
                    print('Net %d' % (i + 1))
                    game_manager_kwargs['ckpt_path'] = ckpt
                    game_managers.append(gm_class(**game_manager_kwargs))
                print('=====================================================')
                print()

                state = game_managers[0].initial_state()
                mctss = [
                    MCTS(game_managers[i], chal_max_simulations_per_move[i],
                         gconf.cpuct, gconf.virtual_loss, state,
                         gconf.root_noise_weight, gconf.dirichlet_noise_param,
                         chal_eval_batch_size[i],
                         game_manager_kwargs['tf_device'])
                    for i in range(len(game_managers))]

                iplayer = ip1
                iplay = 0
                moves = []
                imove = None
                while not game_managers[iplayer].is_over(
                        state.state[np.newaxis])[0]:
                    if iplay < gconf.num_relaxed_turns:
                        turn_temperature = 1.0
                    else:
                        turn_temperature = gconf.move_temperature
                    imc = iplayer % len(mctss)
                    if args.show_middle_game:
                        game_manager_io.print_board(state, imove)

                        stats = mctss[imc].simulate(
                            state, gconf.max_seconds_per_move)

                        print('Net %d to play:' % (iplayer + 1))

                        if args.show_mcts:
                            print('MCTS stats')
                            game_manager_io.print_stats(stats)
                            print()

                        if args.show_win_prob:
                            with tf.device(game_manager_kwargs['tf_device']):
                                _, value_prior = \
                                    mctss[imc].game_manager.predict(
                                        tf.constant(state.state[np.newaxis],
                                                    tf.float32))
                                win_prob = (value_prior[0] + 1.0) / 2.0
                                print('Estimated win probability: %.03f\n' %
                                      win_prob)

                        if args.show_move_prob:
                            print('Move probabilities:')
                            game_manager_io.print_stats_on_board(stats, 1)
                            print()

                        if args.show_move_prob_temp:
                            print('Move probabilities with temperature ' +
                                  '%.1e' % turn_temperature)
                            game_manager_io.print_stats_on_board(
                                stats, turn_temperature)
                            print()

                    imove, _ = mctss[imc].choose_move(
                        turn_temperature)
                    moves.append((imove, iplayer))
                    state = game_managers[iplayer].update_state(state, imove)
                    iplayer = (iplayer + 1) % 2
                    for imc2 in range(len(mctss)):
                        mctss[imc2].update_root(imove,  state)
                    iplay += 1
                game_manager_io.print_board(state, imove)
                iwinner = game_managers[iplayer].get_iwinner(
                    state.state[np.newaxis])[0]

                print('Checkpoint %d vs. %d result (match %d):' %
                      tuple(chal_ckpt_nums + [ichallenge+1]))
                if iwinner < 0:
                    print('DRAW')
                    results[0, ickpt1, ickpt2, 2] += 1
                    results[1, ickpt1, ickpt2, 2] += 1
                elif iwinner == ip1:
                    print('Checkpoint %d won' % args.num_iters_ckpt[ickpt1])
                    results[ip1, ickpt1, ickpt2, 0] += 1
                    results[(ip1+1) % 2, ickpt2, ickpt1, 1] += 1
                else:
                    print('Checkpoint %d won' % args.num_iters_ckpt[ickpt2])
                    results[(ip1+1) % 2, ickpt2, ickpt1, 0] += 1
                    results[ip1, ickpt1, ickpt2, 1] += 1

                print('\nNumber of wins of the players in the rows vs. the ' +
                      'players in the columns. Missing results are draws.\n')
                print_results(np.sum(results[:, :, :, 0], axis=0),
                              args.num_iters_ckpt)

                if args.show_results_by_player:
                    print('Results when playing as player 1.\n')
                    print_results(results[0, :, :, 0], args.num_iters_ckpt)
                    print('Results when playing as player 2.\n')
                    print_results(results[1, :, :, 0], args.num_iters_ckpt)

        ip1 = (ip1 + 1) % 2


def print_results(results, ckpt_nums):
    """ Prints the table of results of the challenges.

    Args:
      results: np.array: an NxN 2D array with the number of wins between each
        pair of players. The number represents the number of wins of the player
        in the rows against the player in the columns.
      ckpt_nums: list: an N vector with the number of iterations which
        represents the "name" of each player of the table.
    """
    ckpt_nums = [str(x) for x in ckpt_nums]
    maxlen = 0
    for x in ckpt_nums:
        maxlen = max(maxlen, len(x))
    for x in results.flatten():
        maxlen = max(maxlen, len(str(x)))

    width = maxlen + 1

    for i in range(-2, len(ckpt_nums)):
        for j in range(-1, len(ckpt_nums)):
            if i == -2:
                if j == -1:
                    print(''.rjust(maxlen) + ' |', end='')
                else:
                    print(ckpt_nums[j].rjust(width), end='')
            elif i == -1:
                if j == -1:
                    repeats = maxlen + 2
                else:
                    repeats = width
                print(''.join('-' * repeats), end='')
            else:
                if j == -1:
                    print(ckpt_nums[i].rjust(maxlen) + ' |', end='')
                else:
                    print(str(results[i, j]).rjust(width), end='')
        print()
    print()


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
        '--num_challenges',
        help=('Number of games to play. If less than one, defaults to value ' +
              'in config.py.'),
        default=0,
        type=int
    )
    parser.add_argument(
        '--num_iters_ckpt',
        nargs='+',
        help=('List (separated by spaces) of number of iterations in the ' +
              'checkpoints to load. e.g. if the file is called ' +
              'moku3_3x3_1000.ckpt, put 1000 in the list. Use -1 to load ' +
              'the latest checkpoint or 0 to use a naive network.'),
        default=['0', '-1']
    )
    parser.add_argument(
        '--max_simulations_per_move',
        nargs='+',
        help=('Max number of MCTS simulations per move. This must be a ' +
              'list (separated by spaces) of either one single value or a ' +
              'list whose length is the same as the list in num_iters_ckpt. ' +
              'If a value is less than one, then this value is replaced by ' +
              'that defined in config.py.'),
        default=['0']
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
    parser.add_argument(
        '-smid',
        '--show_middle_game',
        help=('If set, shows the board after each move. Otherwise, only the ' +
              'final board of each challenge is shown.'),
        nargs='?',
        const=True,
        default=False,
        type=bool
    )
    parser.add_argument(
        '-sr',
        '--show_results_by_player',
        help=('If set, also shows the winning results separately when ' +
              'playing as player 1 and 2.'),
        nargs='?',
        const=True,
        default=False,
        type=bool
    )
    parser.add_argument(
        '-isp',
        '--include_self_play',
        help=('If set, games between the same checkpoint as both players ' +
              'are included in the challenges.'),
        nargs='?',
        const=True,
        default=False,
        type=bool
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    tf.app.run()
