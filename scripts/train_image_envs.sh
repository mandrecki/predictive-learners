#!/usr/bin/env bash

export PYGAME_HIDE_SUPPORT_PROMPT=1
SEED_OFFSET=0
# cluster runnable
#ENV_NAMES=( "PixelCopter-ple-v0" "Pong-ple-v0" "TetrisA-v2" "Catcher-ple-v0" )
...


#ENV_NAMES=( "Pong-ple-v0"  )
#ENV_NAMES=( "cheetah-run" "Ant-v0" "CarRacing-v0" "CartPole-v1" )
#ENV_NAMES=( "Ant-v2" "CarRacing-v0" "CartPole-v0" )
#ENV_NAMES=( "PixelCopter-ple-v0"  "MountainCar-v0"  )
#ENV_NAMES=( "WaterWorld-ple-v0" )
#ENV_NAMES=( "Catcher-ple-v0" )
#ENV_NAMES=("TetrisA-v2" )
ENV_NAMES=( "CarRacing-v0" )

ENV_STEPS=( 5000 5000 5000 5000 5000 5000 )

RUNS=5

for ((i=0;i<${#ENV_NAMES[@]};++i));
do
    for j in `seq 1 $RUNS`
    do
        echo "Running "${ENV_NAMES[i]}" for "${ENV_STEPS[i]}" steps. PPO run $j out of $RUNS"
        python -W ignore ../pytorch-a2c-ppo-acktr/main.py --env-name "${ENV_NAMES[i]}" --num-env-steps "${ENV_STEPS[i]}" --seed $(($SEED_OFFSET+j)) --algo "ppo" --log-interval 1 --stats-file "exp/"${ENV_NAMES[i]}"-ppo_$j.csv" --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 #> /dev/null
        echo "Running "${ENV_NAMES[i]}" for "${ENV_STEPS[i]}" steps. A2C run $j out of $RUNS"
        python -W ignore ../pytorch-a2c-ppo-acktr/main.py --env-name "${ENV_NAMES[i]}" --num-env-steps "${ENV_STEPS[i]}" --seed $(($SEED_OFFSET+j)) --algo "a2c" --log-interval 10 --stats-file "exp/"${ENV_NAMES[i]}"-a2c_$j.csv" #> /dev/null 2>&1
    done
done