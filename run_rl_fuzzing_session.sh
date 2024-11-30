#!/bin/bash

if [ -z "$1" ]; then
    echo "error: please provide number of episodes"
    exit 1
fi
NUM_EPISODES=$1

for((i=1; i<=NUM_EPISODES; i++))
do
    afl-fuzz -i input -o output -E 100 -- ./test_program @@
done