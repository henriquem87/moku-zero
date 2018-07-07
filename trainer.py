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

tfe = tf.contrib.eager


class LearningRate(object):
    """ Helper class for managing the learning rate. It current implements
    only learning rate decay at fixed step numbers.

    Arguments:
      global_step: tfe.Variable: the current step (iteration) number.
      initial_lr: float: initial value of learning rate.
      lr_decay: float: decay value to multiply at each decay step.
      lr_decay_steps: list: the step numbers at which the decay is applied.
    """
    def __init__(self, global_step, initial_lr, lr_decay, lr_decay_steps):
        self.global_step = global_step
        self.current_lr = tfe.Variable(initial_lr, dtype=tf.float32, name='lr')
        self.initial_lr = tf.constant(initial_lr, tf.float32)
        self.lr_decay = tf.constant(lr_decay, tf.float32)
        self.lr_decay_steps = lr_decay_steps
        self.last_lr_update = tfe.Variable(
            global_step, dtype=tf.int64, name='last_lr_update')

    def get_lr(self):
        """ Returns the current learning rate.

        Note that this call will activate the decay, if global_step is at a
        decay step value.

        Returns:
          tfe.Variable: the learning rate ath the current global_step
        """
        if self.global_step > self.last_lr_update and \
                int(self.global_step) in self.lr_decay_steps:
            tf.assign(self.current_lr, self.current_lr * self.lr_decay)
            tf.assign(self.last_lr_update, self.global_step)
        return self.current_lr


def trainer(tr_manager_trainer_queue,
            trainer_tr_manager_queue,
            train_dir, batch_size,
            save_ckpt_interval,
            max_train_iters,
            initial_lr,
            lr_decay,
            lr_decay_steps,
            log_interval,
            backpropagate_losing_policies,
            keep_checkpoint_every_n_hours,
            game_config_string,
            game_manager_module,
            game_manager_kwargs):
    """ Starts the training process. The network parameters will be restored
    from a checkpoint, if it exists.

    Args:
      tr_manager_trainer_queue: Queue: to get training batch samples from
        trainer_manager.
      trainer_tr_manager_queue: Queue: to put checkpoint file names to
        trainer_manager.
      train_dir: string: path to the directory where training files are
        stored.
      batch_size: int: batch size to use during training.
      save_ckpt_interval: int: number of training steps to save a new
        checkpoint.
      max_train_iters: int: number of training steps before concluding.
      initial_lr: float: initial value of learning rate.
      lr_decay: float: decay value to multiply at each decay step.
      lr_decay_steps: list: the step numbers at which the decay is applied.
      log_interval: int: number of steps to print a training log message.
      backpropagate_losing_policies: boolean: if False, ignore policy losses
        coming from the losing player.
      keep_checkpoint_every_n_hours: float: interval in hours at which a
        checkpoint is kept on disk permanently.
      game_config_string: string: a name for the current game.
      game_manager_module: list: a list with two string containing the name
        of the game manager module (file) and the name of the class inside of
        the module.
      game_manager_kwargs: dict: a dictionary of arguments and its respective
        values.
    """
    np.random.seed()

    ckpt_path = game_manager_kwargs['ckpt_path']

    game_manager_kwargs['replace_unloaded_resnet_by_naivenet'] = False
    gm_module = __import__(game_manager_module[0])
    gm_class = getattr(gm_module, game_manager_module[1])
    game_manager = gm_class(**game_manager_kwargs)

    global_step = tf.train.get_or_create_global_step()
    lr = LearningRate(global_step, initial_lr, lr_decay, lr_decay_steps)

    start_time = time.time()
    net = game_manager.net
    optimizer = tf.train.MomentumOptimizer(
        lr.get_lr(), momentum=0.9, use_nesterov=True)
    checkpoint = tfe.Checkpoint(
        net=net, optimizer=optimizer, global_step=global_step,
        current_lr=lr.current_lr)
    if ckpt_path is not None:
        print('Loading training params from: ' + ckpt_path)
        checkpoint.restore(ckpt_path)
    ckpt_name = None
    if ckpt_path is not None:
        ckpt_name = osp.split(ckpt_path)[1]
    trainer_tr_manager_queue.put(ckpt_name)
    writer = tf.contrib.summary.create_file_writer(train_dir)
    writer.set_as_default()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_reg_loss = 0.0

    exp_decay = 1.0 - 1.0/log_interval
    exp_moving_loss = -1.0
    exp_moving_policy_loss = -1.0
    exp_moving_value_loss = -1.0
    exp_moving_reg_loss = -1.0

    keep_checkpoint_every_n_seconds = keep_checkpoint_every_n_hours * 3600.0
    last_kept_checkpoint_time = time.time()
    while global_step <= max_train_iters:
        # Workaround for memory leak when using loss in Eager Execution
        # See tensorflow issue #20062
        tf.reset_default_graph()

        with tf.contrib.summary.always_record_summaries():
            states_batch, policy_batch, value_prior_batch = \
                tr_manager_trainer_queue.get()
            with tf.device(game_manager_kwargs['tf_device']):
                states_batch_tf = tf.constant(states_batch, tf.float32)
                policy_batch_tf = tf.constant(policy_batch, tf.int32)
                value_prior_batch_tf = \
                    tf.constant(value_prior_batch, tf.float32)

                with tfe.GradientTape() as tape:
                    policy_pred, value_pred = \
                        net(states_batch_tf, training=True)
                    policy_loss = tf.losses.sparse_softmax_cross_entropy(
                        policy_batch_tf, policy_pred,
                        reduction=tf.losses.Reduction.NONE)
                    if not backpropagate_losing_policies:
                        policy_loss = tf.where(
                            tf.less(value_prior_batch_tf, 0.0),
                            tf.zeros_like(policy_loss),
                            policy_loss)
                    policy_loss = tf.reduce_mean(policy_loss)
                    value_loss = tf.square(
                        value_pred[:, 0] - value_prior_batch_tf)
                    value_loss = tf.reduce_mean(value_loss)
                    reg_loss = tf.reduce_sum(net.losses)
                    loss = policy_loss + value_loss + reg_loss

                grads = tape.gradient(loss, net.variables)
                optimizer.apply_gradients(
                    zip(grads, net.variables),
                    global_step=global_step)

                total_loss += loss
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_reg_loss += reg_loss

                if exp_moving_loss < 0.0:
                    exp_moving_loss = loss
                    exp_moving_policy_loss = policy_loss
                    exp_moving_value_loss = value_loss
                    exp_moving_reg_loss = reg_loss
                else:
                    exp_moving_loss = \
                        exp_decay * exp_moving_loss + (1.0-exp_decay) * loss
                    exp_moving_policy_loss = \
                        exp_decay * exp_moving_policy_loss + \
                        (1.0-exp_decay) * policy_loss
                    exp_moving_value_loss = \
                        exp_decay * exp_moving_value_loss + \
                        (1.0-exp_decay) * value_loss
                    exp_moving_reg_loss = \
                        exp_decay * exp_moving_reg_loss + \
                        (1.0-exp_decay) * reg_loss

                if int(global_step) % log_interval == 0:
                    tf.contrib.summary.scalar(
                        'policy_loss', exp_moving_policy_loss,
                        step=global_step)
                    tf.contrib.summary.scalar(
                        'value_loss', exp_moving_value_loss, step=global_step)
                    tf.contrib.summary.scalar(
                        'regularization_loss', exp_moving_reg_loss,
                        step=global_step)
                    tf.contrib.summary.scalar(
                        'total_loss', exp_moving_loss, step=global_step)
                    tf.contrib.summary.scalar('lr', lr.get_lr(),
                                              step=global_step)

                    total_loss /= log_interval
                    total_policy_loss /= log_interval
                    total_value_loss /= log_interval
                    total_reg_loss /= log_interval
                    elapsed_time = time.time() - start_time
                    examples_per_second = \
                        (states_batch.shape[0] * float(log_interval)) / \
                        elapsed_time
                    print(
                        ('%s: Train iter: %d, loss %.04f, ' +
                         'policy-loss %.04f, value-loss %.04f, ' +
                         'regul-loss %.04f, lr %.1e, ' +
                         '%.01f examples per sec.') %
                        (datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
                         global_step, total_loss, total_policy_loss,
                         total_value_loss, total_reg_loss,
                         float(lr.get_lr().value()), examples_per_second))
                    total_loss = 0.0
                    total_policy_loss = 0.0
                    total_value_loss = 0.0
                    start_time = time.time()
                if int(global_step) % save_ckpt_interval == 0:
                    ckpt_name = '%s-%d.ckpt' % \
                        (game_config_string, global_step)
                    ckpt_path = osp.join(train_dir, ckpt_name)
                    checkpoint.save(ckpt_path)
                    ckpt_path = tf.train.get_checkpoint_state(train_dir)\
                        .model_checkpoint_path

                    # This could be done automatically if tfe.Checkpoint
                    # supported the keep_checkpoint_every_n_hours argument
                    # like tf.train.Saver does
                    ckpt_interval = time.time() - last_kept_checkpoint_time
                    if ckpt_interval > keep_checkpoint_every_n_seconds:
                        last_ckpt_files = [f for f in os.listdir(train_dir)
                                           if f.startswith(ckpt_name)]
                        for lcf in last_ckpt_files:
                            shutil.copy(
                                osp.join(train_dir, lcf),
                                osp.join(train_dir, lcf.replace(
                                    '.ckpt', '.ckpt-keep')))
                        last_kept_checkpoint_time = time.time()
                    print('%s: saved model %s' %
                          (datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
                           osp.join(train_dir,
                                    '%s-%d.ckpt' %
                                    (game_config_string, global_step))))

                    if global_step < max_train_iters:
                        ckpt_name = osp.split(ckpt_path)[1]
                        trainer_tr_manager_queue.put(ckpt_name)
