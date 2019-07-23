#!/usr/bin/env bash

#SEED_OFFSET=200

#ENV_NAMES=(  "ball_in_cup-catch" "finger-spin" "cheetah-run" "cartpole-balance" )
#ENV_NAMES=(  "cartpole-balance" "reacher-easy"  "ball_in_cup-catch" "finger-spin" "cheetah-run" )
ENV_NAMES=( "PixelCopter-ple-v0" "Pong-ple-v0" "TetrisA-v2" "Catcher-ple-v0" )
#ENV_NAMES=( "Ant-v2" "CartPole-v0" "Pendulum-v0" "HalfCheetah-v2" "HalfCheetah-v2" )
#ENV_NAMES=( "CartPole-v0" "Pendulum-v0" "HalfCheetah-v2" )
#ENV_NAMES=( "CarRacing-v0" )

#EXP_NAME="classic-cp-rnn-1"
#EXP_NAME="car-rnn-1"
#EXP_NAME="pixelcopter-detail"
EXP_NAME="pixelcopter-model"

EXP_FOLDER="../exp/$EXP_NAME/"
#EXP_FOLDER="../clean_experiments/$EXP_NAME/"
#EXP_FOLDER="../cluster_experiments/$EXP_NAME/"
RUNS=1

STEPS=10000

for ((i=0;i<${#ENV_NAMES[@]};++i));
do
    for j in `seq 1 $RUNS`
    do
        echo "Running "${ENV_NAMES[i]}
        python ../pred_learn/data/experience_collector.py --env-id "${ENV_NAMES[i]}" --file-number $j --total-steps $STEPS --rl-model-path $EXP_FOLDER/trained_models/test/ppo/"${ENV_NAMES[i]}"-1.pt # --render
        python ../pred_learn/data/experience_collector.py --env-id "${ENV_NAMES[i]}" --file-number $j --total-steps $STEPS --rl-model-path $EXP_FOLDER/trained_models/base/ppo/"${ENV_NAMES[i]}"-1.pt --render
#        python ../pred_learn/data/experience_collector.py --env-id "${ENV_NAMES[i]}" --file-number $j --total-steps 1000 --rl-model-path $EXP_FOLDER/trained_models/test/ppo/"${ENV_NAMES[i]}"-1.pt --render --extra-detail

    done
done
