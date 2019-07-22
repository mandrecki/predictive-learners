#!/usr/bin/env bash

#SEED_OFFSET=200

#ENV_NAMES=(  "ball_in_cup-catch" "finger-spin" "cheetah-run" "cartpole-balance" )
#ENV_NAMES=( "finger-spin" "cheetah-run" "cartpole-balance" "ball_in_cup-catch"  )
ENV_NAMES=( "PixelCopter-ple-v0" "Pong-ple-v0" "TetrisA-v2" "Catcher-ple-v0" )

#ENV_NAMES=( "CarRacing-v0" )

RUNS=1

for ((i=0;i<${#ENV_NAMES[@]};++i));
do
    for j in `seq 1 $RUNS`
    do
        echo "Running "${ENV_NAMES[i]}
        python ../pred_learn/data/experience_collector.py --env-id "${ENV_NAMES[i]}" --file-number $j --total-steps 1000 --rl-model-path ../trained_models/ppo/"${ENV_NAMES[i]}"-1.pt --render --extra-detail

    done
done
