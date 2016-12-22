#!/bin/bash
# Arguments:
# AWS key
# AWS instance ID
# Game
# Agent
# Episode

scp -i "$1" "$2:/home/ubuntu/RL-algorithms/stats/$3-$4.npz" ~/MVA-projects/reinforcement-learning/Arcade-Learning-Environment/RL-algorithms/stats/
scp -i "$1" "$2:/home/ubuntu/RL-algorithms/video/$3-$4-$5.avi" ~/MVA-projects/reinforcement-learning/Arcade-Learning-Environment/RL-algorithms/downloads/
scp -i "$1" "$2:/home/ubuntu/RL-algorithms/weights/$3-$4-$5.pkl" ~/MVA-projects/reinforcement-learning/Arcade-Learning-Environment/RL-algorithms/downloads/
