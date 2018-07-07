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

""" Central configuration file for various parts of the code. For an
explanation about each of the config arguments, check config_explain.txt"""


class NetworkConfig(object):
    # Change the values below if you want to run the trainer and players in
    # different computers in your local network.

    # Port where the trainer server will be listening to receive games
    # generated by the players.
    games_queue_port = 12345
    # Port where the file server will be listening to send/receive checkpoint
    # names and files to feed the players.
    file_server_port = 12346
    # A password to authenticate the players. It must be a byte string, so do
    # not remove the b before the string
    authkey = b'mokumcts'
    # The IP of the computer where the trainer is running.
    # Leave it as 'localhost' if the trainer and the players are running in the
    # same computer.
    server_ip = 'localhost'


class Connect4Config(object):
    drop_mode = True
    moku_size = 4
    board_size = [6, 7]
    max_simulations_per_move = 100
    eval_batch_size = 4
    num_games_per_checkpoint = 100
    num_relaxed_turns = 3
    max_samples_per_result_to_train = 2000000
    num_channels = 128
    num_res_layers = 5
    max_train_iters = 5000000
    random_move_probability = 1.0 / 80.0
    log_interval = 100
    save_ckpt_interval = 5000
    initial_lr = 0.01
    lr_decay = 0.1
    lr_decay_steps = []
    augment_training_samples = True
    backpropagate_losing_policies = False
    use_relative_value_labels = True
    keep_checkpoint_every_n_hours = 1.0
    queue_capacity = 10
    train_batch_size = 64
    max_seconds_per_move = 10
    move_temperature = 0.000001
    cpuct = 1.0
    virtual_loss = 10
    root_noise_weight = 0.25
    dirichlet_noise_param = 10.0 / (board_size[0] * board_size[1])


class Connect4TestConfig(Connect4Config):
    num_relaxed_turns = 0
    root_noise_weight = 0.0


class Connect4ChallengeConfig(Connect4Config):
    num_challenges = 30
    num_relaxed_turns = 0
    root_noise_weight = 0.0


class GomokuConfig(object):
    drop_mode = False
    moku_size = 5
    board_size = [15, 15]
    max_simulations_per_move = 100
    eval_batch_size = 4
    num_games_per_checkpoint = 200
    num_relaxed_turns = 5
    max_samples_per_result_to_train = 10000000
    num_channels = 128
    num_res_layers = 5
    max_train_iters = 2000000
    random_move_probability = 1.0 / 500.0
    log_interval = 100
    save_ckpt_interval = 5000
    initial_lr = 0.01
    lr_decay = 0.1
    lr_decay_steps = []
    augment_training_samples = True
    backpropagate_losing_policies = False
    use_relative_value_labels = True
    keep_checkpoint_every_n_hours = 1.0
    queue_capacity = 10
    train_batch_size = 64
    max_seconds_per_move = 10
    move_temperature = 0.000001
    cpuct = 1.0
    virtual_loss = 10
    root_noise_weight = 0.25
    dirichlet_noise_param = 10.0 / (board_size[0] * board_size[1])


class GomokuTestConfig(GomokuConfig):
    num_relaxed_turns = 0
    root_noise_weight = 0.0


class GomokuChallengeConfig(GomokuConfig):
    num_challenges = 20
    num_relaxed_turns = 0
    root_noise_weight = 0.0


class TicTacToeConfig(object):
    drop_mode = False
    moku_size = 3
    board_size = [3, 3]
    max_simulations_per_move = 20
    eval_batch_size = 1
    num_games_per_checkpoint = 100
    num_relaxed_turns = 1
    max_samples_per_result_to_train = 100000
    num_channels = 32
    num_res_layers = 1
    max_train_iters = 200000
    random_move_probability = 1.0 / 16.0
    log_interval = 100
    save_ckpt_interval = 5000
    initial_lr = 0.01
    lr_decay = 0.1
    lr_decay_steps = []
    augment_training_samples = True
    backpropagate_losing_policies = False
    use_relative_value_labels = True
    keep_checkpoint_every_n_hours = 1.0 / 12.0
    queue_capacity = 10
    train_batch_size = 64
    max_seconds_per_move = 10
    move_temperature = 0.000001
    cpuct = 1.0
    virtual_loss = 10
    root_noise_weight = 0.25
    dirichlet_noise_param = 10.0 / (board_size[0] * board_size[1])


class TicTacToeTestConfig(TicTacToeConfig):
    num_relaxed_turns = 0
    root_noise_weight = 0.0


class TicTacToeChallengeConfig(TicTacToeConfig):
    num_challenges = 100
    num_relaxed_turns = 1
    root_noise_weight = 0.0
