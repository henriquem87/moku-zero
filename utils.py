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

""" Diverse helper functions. """

import config
import hashlib
import inspect
import os
import os.path as osp
import sys
import tensorflow as tf


def checksum(file_path, block_size=65536):
    """ Computes the SHA1 checksum of a file.

    Args:
      file_path: string: the path to the file.
      block_size: the size of the chunks to be read.

    Returns:
      byte string: the checksum in bytes.
    """
    h = hashlib.sha1()
    with open(file_path, 'rb') as f:
        block = f.read(block_size)
        while len(block) > 0:
            h.update(block)
            block = f.read(block_size)
    return h.digest()


def generate_moku_manager_params(drop_mode,
                                 moku_size,
                                 board_size,
                                 gpu_id,
                                 num_res_layers,
                                 num_channels,
                                 ckpt_path=None,
                                 replace_unloaded_resnet_by_naivenet=True):
    """
    Args:
      drop_mode: boolean: indicating if the game is played in drop mode.
      moku_size: int: the number of same pieces in a line that are
        required to win the game.
      board_size: tuple: two values indicating number of rows and number of
        columns of the board.
      gpu_id: int: id of the GPU to use, or -1 to use CPU.
      num_res_layers: int: number of residual blocks in the CNN.
      num_channels: int: number of channels in the CNN layers.
      ckpt_path: string or None: path to the checkpoint file to be loaded.
      replace_unloaded_resnet_by_naivenet: boolean: if True, when ckpt_path
        is None, a NaiveNet is used instead of a ResNet with random weights.
    """
    if gpu_id < 0:
        tf_device = '/cpu:0'
    else:
        tf_device = '/gpu:' + str(gpu_id)

    game_config_string = 'moku_%d_%dx%d' % (
        moku_size, board_size[0], board_size[1])
    if drop_mode:
        game_config_string += '_drop'
    game_manager_module = ['moku_manager', 'MokuManager']
    game_manager_kwargs = {
        'drop_mode': drop_mode,
        'moku_size': moku_size,
        'board_size': board_size,
        'tf_device': tf_device,
        'num_res_layers': num_res_layers,
        'num_channels': num_channels,
        'ckpt_path': ckpt_path,
        'replace_unloaded_resnet_by_naivenet':
            replace_unloaded_resnet_by_naivenet}
    game_manager_io_module = ['moku_manager_io', 'MokuManagerIO']
    game_manager_io_kwargs = {
        'drop_mode': drop_mode,
        'board_size': board_size}
    return (game_config_string, game_manager_module, game_manager_kwargs,
            game_manager_io_module, game_manager_io_kwargs)


def get_checkpoint_path(train_dir, num_iters_ckpt):
    """ Finds the checkpoint path that corresponds to the num_iters_ckpt.

    Args:
      train_dir: string: path to the directory where the checkpoints files are
        saved.
      num_iters_ckpt: int: number of training iterations performed when the
        checkpoint was saved.

    Returns:
      string: path to the corresponding checkpoint with num_iters_ckpt
        iterations.
    """
    ckpt_path = None

    if osp.exists(train_dir):
        ckpt = tf.train.get_checkpoint_state(train_dir)
        if num_iters_ckpt < 0 and ckpt and ckpt.model_checkpoint_path:
            ckpt_path = ckpt.model_checkpoint_path
        elif num_iters_ckpt == 0:
            ckpt_path = None
        else:
            ckpt_files = [f for f in os.listdir(train_dir) if '.ckpt' in f]
            ckpt_files = [f[:f.rfind('.')] for f in ckpt_files]
            ckpt_files = sorted(list(set(ckpt_files)))

            ckpt_num = [int(f.split('.')[0].split('-')[-1])
                        for f in ckpt_files]
            ckpt_dict = {n: f for n, f in zip(ckpt_num, ckpt_files)}

            if num_iters_ckpt < 0 and len(ckpt_files) == 0:
                ckpt_path = None
            else:
                if num_iters_ckpt < 0:
                    num_iters_ckpt = sorted(ckpt_num)[-1]

                ckpt_name = ckpt_dict.get(num_iters_ckpt)
                if ckpt_name is None:
                    ckpt_num_str = sorted(list(set(ckpt_num)))
                    ckpt_num_str = [str(x) for x in ckpt_num_str]
                    print('Checkpoint with ' + str(num_iters_ckpt) +
                          ' iterations cannot be found, the available ' +
                          'values are: {' + ', '.join(ckpt_num_str) + '}')
                    sys.exit()

                ckpt_path = osp.join(train_dir, ckpt_name)
    return ckpt_path


def get_game_config(game_mode, run_mode=''):
    """ Finds and loads a config object according to the given mode strings.
    The config class must be called game_mode + run_mode + 'config', e.g.,
    TicTacToeChallengeConfig. Caps are ignored.

    Args:
      game_mode: string: a string representing the name of the game, e.g.,
        tictactoe.
      run_mode: string: the running mode of the game, e.g., challenge or test.

    Returns:
      config: the corresponding config object.

    Raises:
      ValueError: if a config with the given game_mode and run_mode cannot be
        found.
    """
    for name, obj in inspect.getmembers(config):
        if inspect.isclass(obj):
            if name.lower() == (game_mode+run_mode+'config').lower():
                return getattr(config, name)
    raise ValueError(
        'Config class %sconfig not found.' % (game_mode + run_mode))


def get_last_checkpoint_number(train_dir):
    """ Finds the number of iterations corresponding to the latest checkpoint.

    Args:
      train_dir: string: path to the directory where the checkpoints files are
        saved.

    Returns:
      int: Number of iteration in thee latest checkpoint.
    """
    ckpt_files = [f for f in os.listdir(train_dir) if '.ckpt' in f]
    ckpt_files = [f[:f.rfind('.')] for f in ckpt_files]
    ckpt_files = set(ckpt_files)
    ckpt_num = sorted(
        [int(f.split('.')[0].split('-')[-1]) for f in ckpt_files])
    return ckpt_num[-1]


def get_valid_game_modes():
    """ Returns a list with the names of valid game modes. This function simply
    searches the config file to find classes whose suffix appear in three
    specific class names:
    - suffixconfig,
    - suffixtestconfig,
    - suffixchallengeconfig.
    All suffixes that satisfy the three above conditions are returned as valid
    game mode names.

    Returns:
      list: a list with the valid game mode names
    """
    class_names = []
    for name, obj in inspect.getmembers(config):
        if inspect.isclass(obj):
            class_names.append(name.lower())
    game_candidates = [
        c.replace('config', '').replace('challenge', '').replace('test', '')
        for c in class_names]
    candidates_count = {}
    for c in game_candidates:
        if candidates_count.get(c) is None:
            candidates_count[c] = 0
        candidates_count[c] += 1
    valid_game_modes = [
        c for c in candidates_count.keys() if candidates_count[c] == 3]
    return valid_game_modes


def get_valid_game_modes_string():
    """ Returns a string with the valid game mode names (just for printing).
    """
    return ', '.join(get_valid_game_modes())
