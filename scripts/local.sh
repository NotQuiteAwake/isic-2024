#!/bin/bash

HOST="$1"
PORT="$2"
KEYS="authorized_keys"
REMOTE_KEYS="/root/.ssh/$KEYS"

scp -P $PORT $KEYS "root@$HOST:$REMOTE_KEYS"
ssh -p $PORT "root@$HOST" "bash -s" < setup.sh
