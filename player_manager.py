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

import os
import os.path as osp


def player_manager(trmanager_plmanager_queue,
                   plmanager_player_queue,
                   plmanager_fileclient_queue,
                   fileclient_plmanager_queue,
                   train_dir,
                   max_ckpts_to_keep):
    """ Starts the player manager process. It will perform the following tasks:
      1) receive checkpoint paths from train_manager,
      2) ask the file_client to download files which are not available locally,
      3) forward the checkpoint paths to the players.

    Args:
      trmanager_plmanager_queue: Queue: to get checkpoint paths from
      train_manager.
      plmanager_player_queue: Queue: to send checkpoint paths to the players.
      plmanager_fileclient_queue: Queue: to send requests to file_client for
        files to be downloaded from the server.
      fileclient_plmanager_queue: Queue: to receive messages from file_client.
      train_dir: string: path to the directory where training files are being
        stored in the client (player) machine.
      max_ckpts_to_keep: int: maximum  number of older checkpoints to keep in
        disk.
    """

    prev_ckpt_name = None
    downloaded_ckpt_names = []
    while True:
        ckpt_name = trmanager_plmanager_queue.get()

        if ckpt_name != prev_ckpt_name:
            ckpt_suffixes = ['.index', '.data-00000-of-00001']
            files_to_receive = [ckpt_name + suf for suf in ckpt_suffixes]
            if not osp.exists(osp.join(train_dir, files_to_receive[0])) or \
                    not osp.exists(osp.join(train_dir, files_to_receive[1])):
                downloaded_ckpt_names.append(ckpt_name)
                for f in files_to_receive:
                    received_correctly = False
                    while not received_correctly:
                        plmanager_fileclient_queue.put((train_dir, f))
                        received_correctly = fileclient_plmanager_queue.get()
            prev_ckpt_name = ckpt_name
            if len(downloaded_ckpt_names) > max_ckpts_to_keep:
                del_ckpt_name = downloaded_ckpt_names[0]
                del_files = [del_ckpt_name + suf for suf in ckpt_suffixes]
                for f in del_files:
                    f_path = osp.join(train_dir, f)
                    if osp.exists(f_path):
                        os.remove(f_path)
                del downloaded_ckpt_names[0]

        plmanager_player_queue.put(ckpt_name)
