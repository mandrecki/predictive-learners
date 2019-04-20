#!/usr/bin/env bash

#SEED_OFFSET=200
ENV_NAMES=( "CarRacing-v0" "Snake-ple-v0" "Tetris-v0" "PuckWorld-ple-v0" "WaterWorld-ple-v0" "PixelCopter-ple-v0" "CubeCrash-v0" "Catcher-ple-v0" "Pong-ple-v0" )
#ENV_NAMES=( "CarRacing-v0" )
RUNS=0

for ((i=0;i<${#ENV_NAMES[@]};++i));
do
    for j in `seq 0 $RUNS`
    do
#        echo "Running "${ENV_NAMES[i]}" for "${ENV_STEPS[i]}" steps. "${PADDING_TYPES[k]}" run "$j" out of $RUNS"
        python pred_learn/data/experience_collector.py --env-id "${ENV_NAMES[i]}" --file-number $j
    done
done
