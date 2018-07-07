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

from datetime import datetime
import numpy as np
import os
import os.path as osp
import shutil
import tensorflow as tf
import time


def train_manager(player_trmanager_queue,
                  trmanager_plmanager_queue,
                  trainer_trmanager_queue,
                  trmanager_trainer_queue,
                  train_dir,
                  max_samples_per_result_to_train,
                  num_games_per_checkpoint,
                  batch_size,
                  augment_training_samples,
                  use_relative_value_labels,
                  game_config_string,
                  game_manager_module,
                  game_manager_kwargs):
    """ Starts the train manager process. It will perform the following tasks:
      1) load samples from disk (if some exists),
      2) receive games from players and convert them into training samples,
      3) periodically save samples to disk,
      4) feed training batches to trainer,
      5) get new checkpoint paths from trainer,
      6) feed checkpoint paths to player_manager.

    Args:
      player_trmanager_queue: Queue: to get games from the players.
      trmanager_plmanager_queue: Queue: to send checkpoint paths to the
        player manager.
      trainer_trmanager_queue: Queue: to get new checkpoint paths from the
        trainer.
      trmanager_trainer_queue: Queue: to sample mini-batches to the trainer.
      train_dir: string: path to the directory where training files are being
        stored in the client (player) machine.
      max_samples_per_result_to_train: int: only the latest
        max_samples_per_result_to_train samples per result are kept in memory
        and used for training. This can be used to focus the training only
        on the newest samples or the restrict memory usage.
      num_games_per_checkpoint: int: number of games that are generated per
        checkpoint file. While this number is not reached, newer checkpints are
        not forwarded to the players. When this value is reached, the samples
        are also saved to disk and the training set is updated.
      batch_size: int: mini-batch size to be used in the training.
      augment_training_samples: boolean: if True, the samples are randomly
        augmented (e.g., by flipping).
      use_relative_value_labels: boolean: if False, value labels are ternary
        values in {-1, 0, 1} representing loss, draw and win. If True, win and
        loss values are relative to the number of moves remaining until
        reaching the deciding move. A win value can be any in the interval
        [0.5, 1.0]. A win value of 0.5 means that the player will win only
        after playing the maximum number of moves of the game. Conversely,
        a win value of 1.0 means that the player will win at the next move. The
        same idea is applied for losses with the sign reversed. If the network
        is able to learn such values, MCTS can prioritize faster wins and
        longer losses.
      game_config_string: string: a name for the current game.
      game_manager_module: list: a list with two string containing the name
        of the game manager module (file) and the name of the class inside of
        the module.
      game_manager_kwargs: dict: a dictionary of arguments and its respective
        values.
    """
    np.random.seed()

    gm_module = __import__(game_manager_module[0])
    gm_class = getattr(gm_module, game_manager_module[1])
    game_manager = gm_class(**game_manager_kwargs)

    result_suffixes = ['draw', 'p1win', 'p2win']

    (states_accumulator, policy_accumulator, value_prior_accumulator,
        num_games, num_loaded_examples) = read_games_from_disk(
            train_dir, game_config_string, result_suffixes)

    dataset = None
    data_iterator = None
    if num_games > 0:
        format_str = ''
        for _ in range(len(states_accumulator)):
            format_str += '%d '
        print(('Loaded ' + format_str + 'examples (%d games) from disk') %
              tuple(list(num_loaded_examples) + [num_games]))
        dataset, data_iterator = \
            initialize_dataset(
                states_accumulator, policy_accumulator,
                value_prior_accumulator, batch_size, data_iterator,
                game_manager, augment_training_samples)

    ckpt_name = trainer_trmanager_queue.get()
    num_games_in_current_checkpoint = 0
    while True:
        if num_games_in_current_checkpoint < num_games_per_checkpoint:
            while not trmanager_plmanager_queue.full():
                trmanager_plmanager_queue.put(ckpt_name)
        else:
            if not trainer_trmanager_queue.empty():
                ckpt_name = trainer_trmanager_queue.get()
                num_games_in_current_checkpoint = 0

        if not player_trmanager_queue.empty():
            (states_accumulator, policy_accumulator,
             value_prior_accumulator) =\
                append_new_samples(
                    states_accumulator, policy_accumulator,
                    value_prior_accumulator, player_trmanager_queue,
                    game_manager, use_relative_value_labels)
            num_games += 1
            num_games_in_current_checkpoint += 1
            if num_games % 10 == 0:
                print('%s: self-played %d games' %
                      (datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
                       num_games))

            if num_games_in_current_checkpoint == num_games_per_checkpoint:
                (states_accumulator, policy_accumulator,
                 value_prior_accumulator) = remove_exceeding_samples(
                    states_accumulator, policy_accumulator,
                    value_prior_accumulator, max_samples_per_result_to_train)

                file_name = 'train_samples_%s.npz' % game_config_string
                save_games(
                    osp.join(train_dir, file_name), states_accumulator,
                    policy_accumulator, value_prior_accumulator,
                    result_suffixes, num_games)
                dataset, data_iterator = initialize_dataset(
                    states_accumulator, policy_accumulator,
                    value_prior_accumulator, batch_size, data_iterator,
                    game_manager, augment_training_samples)

        if dataset is not None and not trmanager_trainer_queue.full():
            try:
                states_batch, policy_batch, value_prior_batch = \
                    data_iterator.next()
            except StopIteration:
                data_iterator = iter(dataset)
                states_batch, policy_batch, value_prior_batch = \
                    data_iterator.next()
            # Convert into numpy for putting in the queue because
            # an EagerTensor cannot be pickled.
            # This may be a bottleneck of the pipeline, but at least some
            # simple tests moving the dataset directly to trainer did not show
            # any noticeable improvements.
            states_batch = states_batch.numpy()
            policy_batch = policy_batch.numpy()
            value_prior_batch = value_prior_batch.numpy()

            trmanager_trainer_queue.put(
                (states_batch, policy_batch, value_prior_batch))


def append_new_samples(states_accumulator,
                       policy_accumulator,
                       value_prior_accumulator,
                       player_trmanager_queue,
                       game_manager,
                       use_relative_value_labels):
    """ Gets a new game from the queue and append to the existing samples.

    Args:
      states_accumulator: np.array NCHW: a tensor with all the collected
        states.
      policy_accumulator: np.array: a vector with all the collected policy
        labels.
      value_prior_accumulator: np.array: a vector with all the collected value
        labels.
      player_trmanager_queue: Queue: to get games from the players.
      game_manager: GameManager: an object containing a manager for the current
        game.
      use_relative_value_labels: boolean: if False, value labels are ternary
        values in {-1, 0, 1}. If True, win and loss values are relative to the
        number of moves remaining until reaching the deciding move. See more
        details in the comments of train_manager.

    Returns:
      np.array NCHW: the states updated with the new samples
      np.array: the policies updated with the new samples
      np.array: the values updated with the new samples
    """
    moves, iwinner = player_trmanager_queue.get()
    states_samples, policy_samples, value_prior_samples = \
        generate_training_samples(
            moves, iwinner, game_manager, use_relative_value_labels)
    ires = iwinner + 1
    if len(states_accumulator[ires]) == 0:
        states_accumulator[ires] = states_samples
        policy_accumulator[ires] = policy_samples
        value_prior_accumulator[ires] = value_prior_samples
    else:
        states_accumulator[ires] = np.concatenate(
            (states_accumulator[ires], states_samples), axis=0)
        policy_accumulator[ires] = np.concatenate(
            (policy_accumulator[ires], policy_samples), axis=0)
        value_prior_accumulator[ires] = np.concatenate(
            (value_prior_accumulator[ires], value_prior_samples), axis=0)

    return (states_accumulator, policy_accumulator, value_prior_accumulator)


def generate_training_samples(moves,
                              iwinner,
                              game_manager,
                              use_relative_value_labels):
    """ Converts a list of moves into training samples.

    Args:
      moves: list: list of (move_index, player_index) pairs in the order they
        were played
      iwinner: index of the winning player, or -1 if there was no winner
      game_manager: GameManager: an object containing a manager for the current
        game.
      use_relative_value_labels: boolean: if False, value labels are ternary
        values in {-1, 0, 1}. If True, win and loss values are relative to the
        number of moves remaining until reaching the deciding move. See more
        details in the comments of train_manager

    Returns:
      np.array NCHW: a tensor with all the played moves
      np.array: a vector with the policy labels for all the moves
      np.array: a vector with the value labels for all the moves
    """
    states_samples = []
    policy_samples = []
    value_prior_samples = []
    state = game_manager.initial_state()
    played_moves = np.zeros(
        game_manager.board_size[0]*game_manager.board_size[1], np.int32)

    max_num_moves = game_manager.max_num_moves()
    num_game_moves = len(moves)
    for i, (imove, iplayer) in enumerate(moves):
        vp = 0.0
        if iwinner >= 0:
            vp = 1.0
            if use_relative_value_labels:
                vp = (vp +
                      (max_num_moves-num_game_moves+i)/(max_num_moves-1)) / 2.0
            if iplayer != iwinner:
                vp *= -1.0

        states_samples.append(state.state.copy())
        policy_samples.append(imove)
        value_prior_samples.append(vp)
        state = game_manager.update_state(state, imove)
        played_moves[imove] += 1

    states_samples = np.stack(states_samples)
    policy_samples = np.stack(policy_samples)
    value_prior_samples = np.stack(value_prior_samples)
    return states_samples, policy_samples, value_prior_samples


def initialize_dataset(states_accumulator,
                       policy_accumulator,
                       value_prior_accumulator,
                       batch_size,
                       data_iterator,
                       game_manager,
                       augment_training_samples):
    """ Creates a tf.data.Dataset from the np.array samples.

    Args:
      states_accumulator: np.array NCHW: a tensor with all the collected
        states.
      policy_accumulator: np.array: a vector with all the collected policy
        labels.
      value_prior_accumulator: np.array: a vector with all the collected value
        labels.
      batch_size: int: mini-batch size to be used in the training.
      data_iterator: iter: an iterator over the dataset.
      game_manager: GameManager: an object containing a manager for the current
        game.
      augment_training_samples: boolean: if True, the samples are randomly
        augmented (e.g., by flipping).
    """
    merged_states = None
    merged_policy = None
    merged_value = None
    for ires in range(len(states_accumulator)):
        if merged_states is None and len(states_accumulator[ires]) > 0:
            merged_states = states_accumulator[ires]
            merged_policy = policy_accumulator[ires]
            merged_value = value_prior_accumulator[ires]
        elif len(states_accumulator[ires]) > 0:
            merged_states = np.concatenate(
                (merged_states, states_accumulator[ires]), axis=0)
            merged_policy = np.concatenate(
                (merged_policy, policy_accumulator[ires]), axis=0)
            merged_value = np.concatenate(
                (merged_value, value_prior_accumulator[ires]), axis=0)

    dataset = tf.data.Dataset.from_tensor_slices(
        (merged_states, merged_policy, merged_value))
    dataset = dataset.shuffle(merged_states.shape[0])
    if augment_training_samples:
        dataset = dataset.map(game_manager.augment_samples)
    dataset = dataset.batch(batch_size)
    data_iterator = iter(dataset)

    return dataset, data_iterator


def read_games_from_disk(save_dir, file_name_prefix, result_suffixes):
    """ Loads games saved on disk, if any.

    Args:
      save_dir: string: path to the directory where the games are saved.
      file_name_prefix: string: prefix of the file names to be loaded.
      result_suffixes: list: list of strings containing the names of the
        result types to be loaded.

    Returns:
      np.array NCHW: a tensor with all the collected states.
      np.array: a vector with all the collected policy labels.
      np.array: a vector with all the collected value labels.
      int: number of games loaded from the file.
      np.array: a vector of integers contaning the number of loaded samples
        per result type.
    """
    states_accumulator = [np.array([]) for _ in range(len(result_suffixes))]
    policy_accumulator = [np.array([]) for _ in range(len(result_suffixes))]
    value_prior_accumulator = [
        np.array([]) for _ in range(len(result_suffixes))]

    num_games = 0
    disk_game_files = sorted(
        [osp.join(save_dir, f)
         for f in os.listdir(save_dir)
         if f.startswith('train_samples_%s' % file_name_prefix) and
         not f[-1] == '~'])
    num_loaded_examples = np.zeros(len(states_accumulator), np.int32)
    for f in disk_game_files:
        npz = np.load(f)
        for ires in range(len(states_accumulator)):
            num_loaded_examples[ires] = \
                num_loaded_examples[ires] + \
                npz['state_%s' % result_suffixes[ires]].shape[0]
            if len(states_accumulator[ires]) == 0:
                states_accumulator[ires] = \
                    npz['state_%s' % result_suffixes[ires]]
                policy_accumulator[ires] = \
                    npz['policy_%s' % result_suffixes[ires]]
                value_prior_accumulator[ires] = \
                    npz['value_prior_%s' % result_suffixes[ires]]
            else:
                states_accumulator[ires] = np.concatenate(
                    (states_accumulator[ires],
                     npz['state_%s' % result_suffixes[ires]]), axis=0)
                policy_accumulator[ires] = np.concatenate(
                    (policy_accumulator[ires],
                     npz['policy_%s' % result_suffixes[ires]]), axis=0)
                value_prior_accumulator[ires] = np.concatenate(
                    (value_prior_accumulator[ires],
                     npz['value_prior_%s' % result_suffixes[ires]]), axis=0)
        num_games += npz['num_games']

    return (states_accumulator, policy_accumulator, value_prior_accumulator,
            num_games, num_loaded_examples)


def remove_exceeding_samples(states_accumulator,
                             policy_accumulator,
                             value_prior_accumulator,
                             max_samples_per_result_to_train):
    """ Remove older samples when they exceed the maximum number of samples
    to keep.

    Args:
      states_accumulator: np.array NCHW: a tensor with all the collected
        states.
      policy_accumulator: np.array: a vector with all the collected policy
        labels.
      value_prior_accumulator: np.array: a vector with all the collected value
        labels.
      max_samples_per_result_to_train: int: number of maximum samples to keep
        per result type.

    Returns:
      np.array NCHW: the states without exceeding samples
      np.array: the policies without exceeding samples
      np.array: the values without exceeding samples
    """
    for ires in range(len(states_accumulator)):
        if len(states_accumulator[ires]) > \
                max_samples_per_result_to_train:
            diff = len(states_accumulator[ires]) - \
                max_samples_per_result_to_train
            states_accumulator[ires] = \
                states_accumulator[ires][diff:]
            policy_accumulator[ires] = \
                policy_accumulator[ires][diff:]
            value_prior_accumulator[ires] = \
                value_prior_accumulator[ires][diff:]
    return states_accumulator, policy_accumulator, value_prior_accumulator


def save_games(save_path,
               states_accumulator,
               policy_accumulator,
               value_prior_accumulator,
               result_suffixes,
               num_games):
    """ Saves games to disk.

    Args:
      save_path: string: path to file to be saved
      states_accumulator: np.array NCHW: a tensor with all the collected
        states.
      policy_accumulator: np.array: a vector with all the collected policy
        labels.
      value_prior_accumulator: np.array: a vector with all the collected value
        labels.
      result_suffixes: list: list of strings containing the names of the
        result types to be loaded.
      num_games: int: number of collected games.
    """
    npz_kwargs = {}
    for ires in range(len(states_accumulator)):
        npz_kwargs['state_%s' % result_suffixes[ires]] = \
            states_accumulator[ires]
        npz_kwargs['value_prior_%s' % result_suffixes[ires]] = \
            value_prior_accumulator[ires]
        npz_kwargs['policy_%s' % result_suffixes[ires]] = \
            policy_accumulator[ires]
    npz_kwargs['num_games'] = num_games
    num_samples_per_result = [
        states_accumulator[j].shape[0]
        for j in range(len(states_accumulator))]

    if osp.exists(save_path):
        shutil.copy(save_path, save_path+'~')
    np.savez_compressed(
        save_path, **npz_kwargs)
    print('Saved %d %d %d examples (%d games) to disk: %s' %
          (num_samples_per_result[0], num_samples_per_result[1],
           num_samples_per_result[2], num_games, save_path))
    if osp.exists(save_path+'~'):
        os.remove(save_path+'~')
