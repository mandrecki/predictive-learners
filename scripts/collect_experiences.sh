#!/usr/bin/env bash

#SEED_OFFSET=200

ENV_NAMES=(  "ball_in_cup-catch" "finger-spin" "cheetah-run" "cartpole-balance" )


RUNS=1

for ((i=0;i<${#ENV_NAMES[@]};++i));
do
    for j in `seq 1 $RUNS`
    do
        echo "Running "${ENV_NAMES[i]}
#        python pred_learn/data/experience_collector.py --env-id
        python ../pred_learn/data/experience_collector.py --env-id "${ENV_NAMES[i]}" --file-number $j --total-steps 100 --rl-model-path ../trained_models/ppo/"${ENV_NAMES[i]}"-1.pt --render

    done
done
