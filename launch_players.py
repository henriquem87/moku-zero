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

""" Run this script to start game (sample) generation for training. Each task
will be launched in a separated process. There are three main tasks involved
in game generation:
  1) player_manager: the middleman between training and playing. It will
    receive the inputs from the train_manager, request the file_client
    to download training files, if necessary, and then forward requests
    to the players.
  2) player: responsible for actually playing the games and generating training
    samples. It loads a given checkpoint, plays a game using MCTS and produces
    a list of moves and the result, which is then given to the train_manager.
    Many players may be run in parallel.
  3) file_client: only used when the player is not in the same machine as the
    trainer. It requests new training checkpoints from the training machine.

To run:

$ python launch_players.py --mode=tictactoe --num_player_processes=3

Or choose other valid arguments (see --help).
"""

import argparse
from datetime import datetime
import multiprocessing as mp
import os
import os.path as osp
import sys
import tensorflow as tf

import utils
from config import NetworkConfig
from file_client import file_client
from multiprocessing.managers import SyncManager
from player import player
from player_manager import player_manager

config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config_proto)


def main():
    args = parse_args()

    valid_modes_list = utils.get_valid_game_modes()
    valid_modes_string = utils.get_valid_game_modes_string()
    if args.mode not in valid_modes_list:
        print('Invalid game mode informed. Please inform a mode with ' +
              '--mode=mode_name, where mode_name is one of the following ' +
              '{%s}' % valid_modes_string)
        sys.exit()

    gconf = utils.get_game_config(args.mode, 'test')

    max_ckpts_to_keep = 1
    args.gpu_id = [int(x) for x in args.gpu_id]
    if len(args.gpu_id) == 0:
        player_gpu_ids = [-1 for _ in range(args.num_player_processes)]
    elif len(args.gpu_id) == 1:
        player_gpu_ids = [
            args.gpu_id[0] for _ in range(args.num_player_processes)]
    else:
        player_gpu_ids = []
        num_repetitions = args.num_player_processes // len(args.gpu_id) + 1
        for _ in range(num_repetitions):
            player_gpu_ids += args.gpu_id
        player_gpu_ids = player_gpu_ids[:args.num_player_processes]

    print('Player gpu ids', player_gpu_ids)

    players_game_manager_kwargs = []
    for gpu_id in player_gpu_ids:
        if args.game_type == 'moku':
            (game_config_string, game_manager_module, game_manager_kwargs,
                _, _) = \
                    utils.generate_moku_manager_params(
                        gconf.drop_mode, gconf.moku_size, gconf.board_size,
                        gpu_id, gconf.num_res_layers, gconf.num_channels)
        else:
            raise NotImplementedError(
                'Game type %s is not supported.' % args.game_type)
        players_game_manager_kwargs.append(game_manager_kwargs)

    train_dir = osp.join('train_files', game_config_string)
    if not osp.exists(train_dir):
        os.makedirs(train_dir)

    netconf = NetworkConfig

    if args.games_queue_port >= 0:
        netconf.games_queue_port = args.games_queue_port
    if args.file_server_port >= 0:
        netconf.file_server_port = args.file_server_port
    if args.server_ip != '':
        netconf.server_ip = args.server_ip
    if args.authkey != '':
        netconf.authkey = args.authkey.encode('utf-8')

    client_manager = init_client_manager(
        netconf.server_ip, netconf.games_queue_port, netconf.authkey)
    trmanager_plmanager_queue = client_manager.get_trmanager_plmanager_queue()
    player_trmanager_queue = client_manager.get_player_trmanager_queue()
    plmanager_fileclient_queue = mp.Queue(1)
    fileclient_plmanager_queue = mp.Queue(1)

    plmanager_player_queue = mp.Queue(gconf.queue_capacity)

    print('%s: Launching players' % datetime.now().strftime(
        '%Y_%m_%d_%H_%M_%S'))

    file_client_p = mp.Process(
        target=file_client, args=(
            netconf.server_ip, netconf.file_server_port,
            plmanager_fileclient_queue, fileclient_plmanager_queue,))
    file_client_p.daemon = True
    file_client_p.start()

    players_p = [mp.Process(
        target=player, args=(
            player_trmanager_queue, plmanager_player_queue,
            train_dir, gconf.max_simulations_per_move,
            gconf.max_seconds_per_move, gconf.move_temperature,
            gconf.num_relaxed_turns, gconf.random_move_probability,
            gconf.cpuct, gconf.virtual_loss, gconf.root_noise_weight,
            gconf.dirichlet_noise_param, gconf.eval_batch_size,
            game_manager_module, players_game_manager_kwargs[i],
        )
    ) for i in range(args.num_player_processes)]
    for p in players_p:
        p.daemon = True
        p.start()

    player_manager_p = mp.Process(
        target=player_manager, args=(
            trmanager_plmanager_queue, plmanager_player_queue,
            plmanager_fileclient_queue, fileclient_plmanager_queue,
            train_dir, max_ckpts_to_keep,))
    player_manager_p.daemon = True
    player_manager_p.start()

    for p in players_p:
        p.join()


def init_client_manager(ip, port, authkey):
    """ Initializes a queue manager for operating over a local network.

    Args:
      ip: string: the IP address of the server (trainer) machine. Use
        'localhost' if everything is running in a single machine.
      port: int: the port opened in the server for the queue manger.
      authkey: byte string: a password to authenticate the client with the
        server.
    """
    class ServerQueueManager(SyncManager):
        pass

    ServerQueueManager.register('get_trmanager_plmanager_queue')
    ServerQueueManager.register('get_player_trmanager_queue')

    manager = ServerQueueManager(address=(ip, port), authkey=authkey)
    print('Connecting queue to %s:%d ...' % (ip, port))
    manager.connect()

    print('Connected.')
    return manager


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
        nargs='+',
        help=('List (separated by spaces) of GPU ids to use, or -1 to use ' +
              'the CPU.'),
        default=['0']
    )
    parser.add_argument(
        '--game_type',
        help=('Type is a more general term which may include many game ' +
              'modes. For example, moku is the type of tictactoe, connect4 ' +
              'and gomoku modes.'),
        default='moku'
    )
    parser.add_argument(
        '-n',
        '--num_player_processes',
        help=('Number of parallel player processes to run.'),
        default=3,
        type=int
    )
    parser.add_argument(
        '--games_queue_port',
        help=('Port opened to receive games from the ' +
              'players\' queue. If negative, defaults to value in config.py'),
        default=-1,
        type=int
    )
    parser.add_argument(
        '--file_server_port',
        help=('Port opened to tranfer files to the players. ' +
              'If negative, defaults to value in config.py.'),
        default=-1,
        type=int
    )
    parser.add_argument(
        '--authkey',
        help=('Authentication key for the communication with players\'' +
              'queue. If empty, defaults to value in config.py.'),
        default=''
    )
    parser.add_argument(
        '--server_ip',
        help=('IP address of the server machine. If empty, defaults to ' +
              'value in config.py.'),
        default=''
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
