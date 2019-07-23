#!/usr/bin/env bash

TARGET_EXP="exp"
mkdir ../cluster_experiments/$TARGET_EXP

scp -r mandrecki@robotarium.hw.ac.uk:~/code/predictive-learners/$TARGET_EXP ../cluster_experiments/$TARGET_EXP


#scp -r mandrecki@robotarium.hw.ac.uk:~/code/predictive-learners/trained_models ../cluster_experiments/$TARGET_EXP/trained_models
