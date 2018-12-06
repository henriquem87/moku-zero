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

""" Run this script to start the training stage. Each task will be launched in
a separated process. There are three main tasks involved in training:
  1) train_manager: the middleman between training and playing. It will
    receive games from the players, transform it into training samples,
    periodically save samples to disk and then serve training batches for the
    trainer.
  2) trainer: receives the training batches and optimizes a deep network. It
    periodically saves new checkpoints to be loaded by the players to produce
    better games.
  3) file_server: only used when the trainer is not in the same machine as the
    player. It awaits for request for checkpoint files and send them to the
    clients.

To run:

$ python launch_train.py --mode=tictactoe

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
from file_server import file_server
from multiprocessing.managers import SyncManager
from trainer import trainer
from train_manager import train_manager

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
            _, _) = \
                utils.generate_moku_manager_params(
                    gconf.drop_mode, gconf.moku_size, gconf.board_size,
                    args.gpu_id, gconf.num_res_layers, gconf.num_channels)
    else:
        raise NotImplementedError(
            'Game type %s is not supported.' % args.game_type)

    train_dir = osp.join('train_files', game_config_string)

    if not osp.exists(train_dir):
        os.makedirs(train_dir)

    ckpt_path = None
    ckpt = tf.train.get_checkpoint_state(train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = ckpt.model_checkpoint_path

    game_manager_kwargs['ckpt_path'] = ckpt_path

    netconf = NetworkConfig()

    if args.games_queue_port >= 0:
        netconf.games_queue_port = args.games_queue_port
    if args.file_server_port >= 0:
        netconf.file_server_port = args.file_server_port
    if args.authkey != '':
        netconf.authkey = args.authkey.encode('utf-8')

    server_manager = init_server_manager(
        netconf.games_queue_port, netconf.authkey, gconf.queue_capacity)
    player_trmanager_queue = server_manager.get_player_trmanager_queue()
    trmanager_plmanager_queue = server_manager.get_trmanager_plmanager_queue()

    trmanager_trainer_queue = mp.Queue(gconf.queue_capacity)
    trainer_trmanager_queue = mp.Queue(1)

    print('%s: Starting trainer' % datetime.now().strftime(
        '%Y_%m_%d_%H_%M_%S'))

    file_server_p = mp.Process(
        target=file_server, args=(
            train_dir, netconf.file_server_port))
    file_server_p.daemon = True
    file_server_p.start()

    train_manager_p = mp.Process(
        target=train_manager, args=(
            player_trmanager_queue, trmanager_plmanager_queue,
            trainer_trmanager_queue, trmanager_trainer_queue, train_dir,
            gconf.max_samples_per_result_to_train,
            gconf.num_games_per_checkpoint, gconf.train_batch_size,
            gconf.augment_training_samples, gconf.use_relative_value_labels,
            game_config_string, game_manager_module, game_manager_kwargs))
    train_manager_p.daemon = True
    train_manager_p.start()

    trainer(trmanager_trainer_queue, trainer_trmanager_queue,
            train_dir, gconf.train_batch_size, gconf.save_ckpt_interval,
            gconf.max_train_iters, gconf.initial_lr, gconf.lr_decay,
            gconf.lr_decay_steps, gconf.log_interval,
            gconf.backpropagate_losing_policies,
            gconf.keep_checkpoint_every_n_hours, game_config_string,
            game_manager_module, game_manager_kwargs)


def init_server_manager(port, authkey, queue_capacity):
    """ Initializes a queue manager for operating over a local network.

    Args:
      port: int: the port opened in the server for the queue manger.
      authkey: byte string: a password to authenticate the client with the
        server.
      queue_capacity: int: the maximum number of items that can be put
        simultaneously in the queue.
    """
    trmanager_plmanager_queue = mp.Queue(queue_capacity)
    player_trmanager_queue = mp.Queue(queue_capacity)

    class JobQueueManager(SyncManager):
        pass

    JobQueueManager.register(
        'get_trmanager_plmanager_queue',
        callable=lambda: trmanager_plmanager_queue)
    JobQueueManager.register(
        'get_player_trmanager_queue',
        callable=lambda: player_trmanager_queue)

    manager = JobQueueManager(address=('', port), authkey=authkey)
    manager.start()
    print('Started server at port %s' % port)
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    tf.app.run()
