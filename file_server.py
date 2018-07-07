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

""" Code for handling file transfers in the server (trainer) side. """

import hashlib
import os.path as osp
import socket
from utils import checksum

PACKET_SIZE = 1024


def file_server(train_dir, port):
    """ Worker process that will handle file transfers in the server (trainer)
    side. A client can send a message containing the name of training file
    followed by a line break. This server then looks for the file in train_dir
    and returns: (1) the checksum of the file in one package then (2) the file
    itself in the remaining packages.

    Args:
      train_dir: string: path to directory where the trainer is storing
        training files.
      port: int: port number opened for file transfers.
    """
    s = socket.socket()
    ip = ''

    s.bind((ip, port))
    while True:
        s.listen(60)
        c, addr = s.accept()
        print('FileServer: got connection from', addr)

        message = c.recv(PACKET_SIZE).decode('utf-8')
        tokens = message.splitlines()
        file_name = tokens[0]
        file_path = osp.join(train_dir, file_name)

        if osp.exists(file_path):
            file_checksum = checksum(file_path)
            print('Sending checksum...')
            c.send(file_checksum.ljust(PACKET_SIZE))
            print('Checksum sent.')

            print('Sending file %s ...' % file_path)
            with open(file_path, 'rb') as f:
                l = f.read(PACKET_SIZE)
                while l:
                    c.send(l)
                    l = f.read(PACKET_SIZE)
            print('%s sent.' % file_path)
        else:
            print('File %s cannot be found, ignoring request.' % file_path)
        c.close()
    s.close()
