#!/usr/bin/env python3

import sys
import numpy as np
import socket

"""
Receive many queries for RTT measurement
"""

#### Change server's host IP when needed ####
HOST          = '10.1.212.73'
PORT          =  65432

BYTES_PER_QUERY = 1 
QUERY_NUM = 1000000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    print("start listening")
    s.listen()
    conn, addr = s.accept()

    with conn:
        print('Connected by', addr)
        for i in range(QUERY_NUM):
            s = conn.recv(BYTES_PER_QUERY)
            conn.sendall(b'0')

    print("testing finished")