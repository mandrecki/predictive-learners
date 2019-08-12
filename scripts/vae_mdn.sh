#!/usr/bin/env bash

#SEED_OFFSET=200

#ENV_NAMES=(  "ball_in_cup-catch" "finger-spin" "cheetah-run" "cartpole-balance" )
#ENV_NAMES=(  "reacher-easy"  "cheetah-run"  "cartpole-balance" "ball_in_cup-catch" "finger-spin" )
#ENV_NAMES=( "PixelCopter-ple-v0" ) #"Catcher-ple-v0" ) # "TetrisA-v2" )
#ENV_NAMES=( "Pong-ple-v0" ) #"Catcher-ple-v0" ) # "TetrisA-v2" )
#ENV_NAMES=( "PixelCopter-ple-v0" "Catcher-ple-v0" "Pong-ple-v0" ) # "TetrisA-v2" )

ENV_NAMES=( "CarRacing-v0" "cartpole-balance" "Pong-ple-v0" "TetrisA-v2" "PixelCopter-ple-v0" "finger-spin"  "cheetah-run" "ball_in_cup-catch" "Catcher-ple-v0" )
#ENV_NAMES=( "Ant-v2" "CartPole-v0" "Pendulum-v0" "HalfCheetah-v2" "HalfCheetah-v2" )
#ENV_NAMES=( "CartPole-v0" "Pendulum-v0" "HalfCheetah-v2" )
#ENV_NAMES=( "CarRacing-v0" )

#EXP_NAME="classic-cp-rnn-1"
#EXP_NAME="car-rnn-1"
#EXP_NAME="pixelcopter-detail"
EXP_NAME="vae_mdn"

EXP_FOLDER="../exp/$EXP_NAME/"
#EXP_FOLDER="../clean_experiments/$EXP_NAME/"
#EXP_FOLDER="../cluster_experiments/$EXP_NAME/"

mkdir ../exp/$EXP_NAME/
mkdir ../exp/$EXP_NAME/trained_models/
mkdir ../exp/$EXP_NAME/trained_models/predictive/

echo "autoencoding with WM model" > ../exp/$EXP_NAME/info.txt

RUNS=1
BIT_DEPTH=3
#STEPS=10000

for ((i=0;i<${#ENV_NAMES[@]};++i));
do
    for j in `seq 1 $RUNS`
    do
        echo "Running "${ENV_NAMES[i]}
        python ../pred_learn/train/train_wm_vae.py --env-id "${ENV_NAMES[i]}" --model-path $EXP_FOLDER/trained_models/predictive/"${ENV_NAMES[i]}"-1.pt --vis --n-epochs 30 --bit-depth $BIT_DEPTH
        python ../pred_learn/train/train_wm_preds.py --env-id "${ENV_NAMES[i]}" --model-path $EXP_FOLDER/trained_models/predictive/"${ENV_NAMES[i]}"-1.pt --vis --n-epochs 30 --bit-depth $BIT_DEPTH

    done
done
