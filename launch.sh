#!/bin/bash

python3 server.py --num_rounds 1 &
python3 client.py --client_id 1 --epoch 50 &
python3 client.py --client_id 2 --epoch 30 &
#python3 client.py --client_id 3 --epoch 30 &
wait