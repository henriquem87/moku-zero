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

""" Code for handling file transfers in the client (player) side. """

import hashlib
import os.path as osp
import socket
from utils import checksum

PACKET_SIZE = 1024


def file_client(server_ip, server_port,
                plmanager_fileclient_queue, fileclient_plmanager_queue):
    """ Worker process that will handle file transfers in the client (player)
    side.

    Args:
      server_ip: string: IP address of the server (trainer).
      server_port: int: port number opened for file transfers in the server.
      plmanager_fileclient_queue: Queue: transfer data from player_manager to
        this process.
      fileclient_plmanager_queue: Queue: transfer data from this process to
        player_manager.
    """
    while True:
        file_dir, file_name = plmanager_fileclient_queue.get()

        s = socket.socket()
        print('Connecting file_client to %s:%d ...' % (server_ip, server_port))
        s.connect((server_ip, server_port))
        print('Connected.')
        s.send((file_name+'\n').ljust(PACKET_SIZE).encode('utf-8'))
        server_checksum = s.recv(PACKET_SIZE)
        write_file_path = osp.join(file_dir, file_name)
        print('Receiving %s ...' % write_file_path)
        with open(write_file_path, 'wb') as f:
            l = s.recv(PACKET_SIZE)
            while l:
                f.write(l)
                l = s.recv(PACKET_SIZE)
        print('Done receiving.')
        s.shutdown(socket.SHUT_WR)
        s.close()

        local_checksum = checksum(write_file_path)
        server_checksum = server_checksum[:len(local_checksum)]
        if server_checksum == local_checksum:
            print('Checksum is correct for %s' % write_file_path)
            fileclient_plmanager_queue.put(True)
        else:
            print('Incorrect checksum, file %s is corrupted' % write_file_path)
            fileclient_plmanager_queue.put(False)
