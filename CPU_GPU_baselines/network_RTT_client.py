#!/usr/bin/env python3

import socket
import numpy as np
import time
from time import perf_counter
import sys

"""
Send many queries to the server, save RTT distribution
"""

# python socket tutorial: https://realpython.com/python-sockets/#socket-api-overview

#### Change server's host IP when needed ####
HOST          = '10.1.212.73'
PORT          =  65432

BYTES_PER_QUERY = 1 
QUERY_NUM = 1000000

response_time = [] # in terms of ms


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    for i in range(QUERY_NUM):
        # t0 = perf_counter()
        t0 = time.time()
        s.sendall(b'0')
        r = s.recv(BYTES_PER_QUERY) 
        # t1 = perf_counter()
        t1 = time.time()
        response_time.append(1000 * 1000 * (t1 - t0)) # in microsecond

    response_time = np.array(response_time, dtype=np.float32)
    print("Average RTT in python socket: {} us".format(np.average(response_time)))
    np.save('./network_response_time', response_time)
