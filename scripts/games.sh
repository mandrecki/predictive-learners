#!/usr/bin/env bash

rm -rf /tmp/gym
export PYGAME_HIDE_SUPPORT_PROMPT=1
SEED_OFFSET=0
# cluster runnable
ENV_NAMES=( "Pong-ple-v0" "PixelCopter-ple-v0" "Catcher-ple-v0" ) # "TetrisA-v2" )

# deepmind suite
#ENV_NAMES=(  "cartpole-balance"   "reacher-easy" "ball_in_cup-catch" "finger-spin" "cheetah-run" )

#ENV_NAMES=( "Catcher-ple-v0"  )
#ENV_NAMES=( "cheetah-run" "Ant-v0" "CarRacing-v0" "CartPole-v1" )
#ENV_NAMES=( "Ant-v2" "CarRacing-v0" "CartPole-v0" )
#ENV_NAMES=( "PixelCopter-ple-v0"  "MountainCar-v0"  )
#ENV_NAMES=( "WaterWorld-ple-v0" )
#ENV_NAMES=( "CartPole-v0" )
#ENV_NAMES=( "TetrisA-v2" )
#ENV_NAMES=( "CarRacing-v0" )

ENV_STEPS=1000000
RUNS=3

EXP_NAME="games-recurrent1frame-images"
rm -r ../exp/$EXP_NAME/
mkdir ../exp/$EXP_NAME/
echo "Effect of random images detail on RL with 1 frame, RNN " > ../exp/$EXP_NAME/info.txt

for ((i=0;i<${#ENV_NAMES[@]};++i));
do
    for j in `seq 1 $RUNS`
    do
        echo "Running "${ENV_NAMES[i]}" for "$ENV_STEPS" steps. PPO with RNN run $j out of $RUNS"
        python -W ignore ../../pytorch-a2c-ppo-acktr/main.py --stats-file "../exp/$EXP_NAME/"${ENV_NAMES[i]}"-test_$j.csv" --env-name "${ENV_NAMES[i]}" --num-env-steps $ENV_STEPS --save-dir "../exp/$EXP_NAME/trained_models/test/" --seed $(($SEED_OFFSET+j)) --algo "ppo" --log-interval 1 --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --recurrent-policy --extra-image # > /dev/null
        echo "Running "${ENV_NAMES[i]}" for "$ENV_STEPS" steps. PPO no detail run $j out of $RUNS"
        python -W ignore ../../pytorch-a2c-ppo-acktr/main.py --stats-file "../exp/$EXP_NAME/"${ENV_NAMES[i]}"-base_$j.csv" --env-name "${ENV_NAMES[i]}" --num-env-steps $ENV_STEPS --save-dir "../exp/$EXP_NAME/trained_models/base/" --seed $(($SEED_OFFSET+j)) --algo "ppo" --log-interval 1 --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --recurrent-policy #> /dev/null
    done
done


#
#for ((i=0;i<${#ENV_NAMES[@]};++i));
#do
#    for j in `seq 1 $RUNS`
#    do
#        echo "Running "${ENV_NAMES[i]}" for "${ENV_STEPS[i]}" steps. PPO run $j out of $RUNS"
#        python -W ignore ../../pytorch-a2c-ppo-acktr/main.py --env-name "${ENV_NAMES[i]}" --num-env-steps "${ENV_STEPS[i]}" --save-dir ../trained_models/ --seed $(($SEED_OFFSET+j)) --algo "ppo" --log-interval 1 --stats-file "../exp/"${ENV_NAMES[i]}"-ppo_$j.csv" --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 #> /dev/null
#        echo "Running "${ENV_NAMES[i]}" for "${ENV_STEPS[i]}" steps. A2C run $j out of $RUNS"
#        python -W ignore ../../pytorch-a2c-ppo-acktr/main.py --env-name "${ENV_NAMES[i]}" --num-env-steps "${ENV_STEPS[i]}" --save-dir ../trained_models/ --seed $(($SEED_OFFSET+j)) --algo "a2c" --log-interval 10 --stats-file "../exp/"${ENV_NAMES[i]}"-a2c_$j.csv" #> /dev/null 2>&1
#    done
#done
