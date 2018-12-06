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
import time

from game_manager import GameManager

DEBUG = False


class MCTS(object):
    """ Implements Monte Carlo Tree Search. For more details, read [1].

    Args:
      game_manager: GameManager: an object containing a manager for the current
        game.
      max_simulations_per_move: int: maximum number of MCTS searches per move.
      cpuct: float: controls the exploration in MCTS. See c_{puct} in [1].
      virtual_loss: float: controls branch exploration when using MCTS in
        parallel mode. See [1].
      initial_state: State: a state representing the initial board
        configuration.
      root_noise_weight: float: amount of noise to be added to policy
        predictions in the root node (first move).
      dirichlet_noise_param: float: controls the dirichlet distribution.
      batch_size: int: number of leaf nodes to collect in MCTS before
        getting a prediction from the network.
      tf_device: string: tensorflow string for tf.device to be used.

    [1] Silver, David, et al. "Mastering the game of go without human
        knowledge." Nature 550.7676 (2017): 354.
    """
    def __init__(self,
                 game_manager,
                 max_simulations_per_move,
                 cpuct,
                 virtual_loss,
                 initial_state,
                 root_noise_weight,
                 dirichlet_noise_param,
                 batch_size,
                 tf_device):
        self.tf_device = tf_device

        self.game_manager = game_manager
        self.max_simulations_per_move = max_simulations_per_move
        self.cpuct = cpuct
        self.virtual_loss = virtual_loss
        self.batch_size = batch_size
        self.root_noise_weight = root_noise_weight
        self.dirichlet_noise_param = dirichlet_noise_param

        self.initialize_tree(initial_state)

    def backup(self, leaf_node, value_prior):
        """ Updates the nodes visited from the root until the leaf_node.

        Args:
          leaf_node: Node: the expanded leaf node
          value_prior: float: the result predicted from the leaf_node
        """
        iplayer = leaf_node.iplayer
        node = leaf_node
        if DEBUG:
            print('in backup', leaf_node)
        if abs(value_prior) == GameManager.DECIDED_MULTIPLIER:
            value_prior /= GameManager.DECIDED_MULTIPLIER
        while node.parent is not None:
            wp = value_prior
            if node.iplayer == iplayer:
                wp *= -1
            node.parent.edge_stats['N'][node.index] = \
                node.parent.edge_stats['N'][node.index] - self.virtual_loss + 1
            node.parent.edge_stats['W'][node.index] = \
                node.parent.edge_stats['W'][node.index] + wp
            node.parent.edge_stats['Q'][node.index] = \
                node.parent.edge_stats['W'][node.index] / \
                node.parent.edge_stats['N'][node.index]
            if DEBUG:
                print('in backup', node.parent)
            node = node.parent

    def choose_move(self, move_temperature, random_move_probability=0.0):
        """ Chooses a move (policy) according to the MCTS stats.

        Args:
          move_temperature: float: exponential factor to rebalance the policy
            and control its level of exploration. This value corresponds to
            1 / \tau in [1].
          random_move_probability: float: probability of making a totally
            random move.

        Returns:
          int: chosen move index.
          np.array: a vector with the probability of playing at each position.
        """
        x = np.random.rand()
        if x < random_move_probability:
            valid_moves = (self.root.edge_stats['N'] > 0)
            probs = valid_moves.astype(np.float32) / np.sum(valid_moves)
        else:
            exponent = 1.0 / move_temperature
            exp_N = np.power(self.root.edge_stats['N'] /
                             np.max(self.root.edge_stats['N']), exponent)
            probs = exp_N / np.sum(exp_N)
        frand = np.random.rand()
        cum_probs = np.cumsum(probs)
        imove = np.searchsorted(cum_probs, frand)
        return imove, probs

    def evaluate(self, states_list):
        """ Computes the policy and value predictions for each state.

        Args:
          states_list: list: a list of game states.

        Returns:
          np.array: a tensor with the policy predictions
          np.array. a vector with the value prediction
        """
        states_batch = []
        for st in states_list:
            states_batch.append(st.state)
        states_batch = np.stack(states_batch)

        policy_priors, value_priors = \
            self.game_manager.predict(states_batch)

        return policy_priors, value_priors

    def expand(self,
               leaf_node,
               inew_child,
               iplayer,
               policy_priors,
               value_prior):
        """ Creates a new leaf node with the predicted priors.

        Args:
          leaf_node: Node: the node which will be the parent of the new node.
          inew_child: int: the index of the new child of leaf_node.
          iplayer: int: the index of the player to move.
          policy_priors: np.array: a tensor with predicted policy for the
            leaf_node.
          value_prior: float: the predicted value for leaf_node.

        Returns:
          Node: the new leaf node.
        """
        edge_stats = self.generate_initial_edge_stats(policy_priors)
        leaf_node.children[inew_child] = Node(
            inew_child, leaf_node, iplayer, edge_stats, value_prior)
        leaf_node = leaf_node.children[inew_child]
        if value_prior == GameManager.DECIDED_MULTIPLIER:
            leaf_node.result = Node.WON
        elif value_prior == -1*GameManager.DECIDED_MULTIPLIER:
            leaf_node.result = Node.LOST
        elif np.max(leaf_node.edge_stats['P']) <= 0:
            leaf_node.result = Node.DRAW
        return leaf_node

    def generate_initial_edge_stats(self, policy_priors):
        """ Initializes an array of MCTS stats.

        Args:
          policy_priors: np.array: a tensor with the predicted policy.

        Returns:
          np.array: the initialized array of stats.
        """
        policy_priors_flat = policy_priors.flatten()
        edge_stats = np.zeros(
            policy_priors_flat.shape,
            dtype=[('N', 'i4'), ('W', 'f4'), ('Q', 'f4'), ('P', 'f4')])
        edge_stats['P'] = policy_priors_flat
        return edge_stats

    def initialize_tree(self, initial_state):
        """ Initializes the root of the tree.

        Args:
          initial_state: State: a state representing the initial board
            configuration.

        Returns:
          Node: the root of the tree.
        """
        policy_priors, value_priors = self.evaluate([initial_state])
        edge_stats = self.generate_initial_edge_stats(policy_priors)
        self.root = Node(
            0, None, initial_state.iplayer_to_move(), edge_stats,
            value_priors[0])

    def select(self, state):
        """ Navigates the tree until reaching a leaf node. The navigation is
        guided by the MCTS stats using PUCT, like in [1] (see the class
        comments for the reference).

        Args:
          state: State: the initial board state to guide the navigation.

        Returns:
          Node: the chosen leaf node.
          int: the chosen child index of the new node to be created below
            the leaf node.
          State: the board state obtained by following the path until the leaf
            node.
        """
        node = self.root
        reached_leaf = False
        while not reached_leaf:
            policy_probs = node.edge_stats['P']
            if node == self.root:
                noise = np.random.dirichlet(
                    [self.dirichlet_noise_param
                     for _ in range(len(policy_probs))])
                policy_probs = (1 - self.root_noise_weight) * \
                    policy_probs + self.root_noise_weight * noise
            U = self.cpuct * policy_probs * \
                (np.sqrt(np.sum(node.edge_stats['N'])) /
                    (1.0 + node.edge_stats['N']))
            U[np.where(policy_probs < 0)] = \
                policy_probs[np.where(policy_probs < 0)]
            Q_U = node.edge_stats['Q'] + U
            max_idx = np.where(Q_U == np.max(Q_U))
            ichild = max_idx[0][np.random.randint(0, len(max_idx[0]))]
            node.edge_stats['N'][ichild] = \
                node.edge_stats['N'][ichild] + self.virtual_loss
            if node.children[ichild] is not None:
                node = node.children[ichild]
                state = self.game_manager.update_state(state, ichild)
            else:
                reached_leaf = True
            if DEBUG:
                print('in select', node.edge_stats['P'], U, Q_U, ichild)
                print(state)

        return node, ichild, state

    def simulate(self, initial_state, max_time):
        """ Runs one complete simulation from the given initial state. This is
        the main function to update the tree, which generates the calls to
        select, evaluate, expand and backup.

        Args:
          initial_state: State: a state representing the initial board
            configuration.
          max_time: float: maximum time to perform the simulation.

        Returns:
          np.array: the MCTS stats for the root node after the simulation.
        """
        start_time = time.time()
        to_evaluate = []
        to_backup = []
        isim = 0

        while isim < self.max_simulations_per_move:
            state = initial_state.copy()
            leaf_node, inew_child, state = self.select(state)
            if leaf_node.result == Node.UNDEFINED and \
                    leaf_node.edge_stats['P'][inew_child] > 0:
                state = self.game_manager.update_state(state, inew_child)
                to_evaluate.append((state, leaf_node, inew_child))
            else:
                if leaf_node.result == Node.UNDEFINED:
                    value_prior = 0
                else:
                    value_prior = leaf_node.result
                to_backup.append((leaf_node, value_prior))
            if len(to_evaluate) >= self.batch_size or \
                    (isim + len(to_evaluate)) >= self.max_simulations_per_move:
                local_batch_size = min(self.batch_size, len(to_evaluate))
                policy_priors, value_priors = self.evaluate(
                    list(zip(*to_evaluate))[0][:local_batch_size])
                for ieval in range(local_batch_size):
                    state, leaf_node, inew_child = to_evaluate[ieval]
                    if leaf_node.children[inew_child] is None or \
                            np.sum(leaf_node.children[inew_child]
                                   .edge_stats['N']) == 0:
                        leaf_node = self.expand(
                            leaf_node, inew_child, state.iplayer_to_move(),
                            policy_priors[ieval], value_priors[ieval])
                    to_backup.append((leaf_node, value_priors[ieval]))
                to_evaluate = to_evaluate[local_batch_size:]
            for lnode, winp in to_backup:
                self.backup(lnode, winp)
                isim += 1
            to_backup = []
            elapsed_time = time.time() - start_time
            if max_time > 0 and elapsed_time > max_time:
                break
        # print('Simulated %d moves in %.01f secs' % (isim + 1, elapsed_time))

        return self.root.edge_stats

    def update_root(self, ichild, state):
        """ Moves the root the given child index and removes the unused part
        of the tree.

        Args:
          ichild: int: the index of the child to be the new root.
          state: State: a state representing the board in the new root.
        """
        if self.root.children[ichild] is not None:
            self.root = self.root.children[ichild]
            self.root.parent = None
        else:
            self.initialize_tree(state)


class Node(object):
    """ A node of the MCTS tree.

    Args:
      index: int: the index of this node in relation to its parent.
      parent: Node: the parent node.
      iplayer: the index of the player to move in this node.
      edge_stats: the MCTS stats for this node.
      value_prior: int: the value prediction for this node.
    """
    UNDEFINED = 2
    WON = 1
    LOST = -1
    DRAW = 0

    def __init__(self, index, parent, iplayer, edge_stats, value_prior):
        self.index = index
        self.parent = parent
        self.iplayer = iplayer
        self.result = Node.UNDEFINED
        self.value_prior = value_prior
        self.initialize_children(edge_stats)

    def initialize_children(self, edge_stats):
        """ Initializes the stats for the children of this node.

        Args:
          edge_stats: np.array: the MCTS for each of the children of this node.
        """
        if DEBUG:
            print(edge_stats)
        self.edge_stats = edge_stats
        self.children = [None for _ in range(edge_stats.shape[0])]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        print(self.index, self.iplayer)
        s = 'index: %d, iplayer: %d\n' % (self.index, self.iplayer)
        s += 'N\tW\tQ\tP'
        if self.edge_stats is None:
            s += '\nNone'
        else:
            for i in range(self.edge_stats.shape[0]):
                s += '\n%.0f\t%.02f\t%.02f\t%.02f' % \
                    (self.edge_stats['N'][i], self.edge_stats['W'][i],
                     self.edge_stats['Q'][i], self.edge_stats['P'][i])
        return s
