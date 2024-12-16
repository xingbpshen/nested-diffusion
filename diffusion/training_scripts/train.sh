export EXP_DIR=./results
export N_STEPS=1000
export RUN_NAME=run_0
export PRIOR_TYPE=f_phi_prior
export CAT_F_PHI=_cat_f_phi
export F_PHI_TYPE=f_phi_supervised  #f_phi_self_supervised
export MODEL_VERSION_DIR=card_onehot_conditional_results/${N_STEPS}steps/nn/${RUN_NAME}/${PRIOR_TYPE}${CAT_F_PHI}/${F_PHI_TYPE}
export LOSS=card_onehot_conditional
export TASK="$TMP_NAME"
export N_SPLITS=1
export DEVICE_ID=0
export N_THREADS=8
export PREPROCESS=grayscaled
export MLP_IDX=0
python main.py --mlp_idx ${MLP_IDX} --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config configs/${TASK}.yml --exp $EXP_DIR/$TASK/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS} --seed ${SEED} --preprocess ${PREPROCESS} --ni

export EXP_DIR=./results
export N_STEPS=1000
export RUN_NAME=run_1
export PRIOR_TYPE=f_phi_prior
export CAT_F_PHI=_cat_f_phi
export F_PHI_TYPE=f_phi_supervised  #f_phi_self_supervised
export MODEL_VERSION_DIR=card_onehot_conditional_results/${N_STEPS}steps/nn/${RUN_NAME}/${PRIOR_TYPE}${CAT_F_PHI}/${F_PHI_TYPE}
export LOSS=card_onehot_conditional
export TASK="$TMP_NAME"
export N_SPLITS=1
export DEVICE_ID=0
export N_THREADS=8
export PREPROCESS=grayscaled
export MLP_IDX=1
python main.py --mlp_idx ${MLP_IDX} --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config configs/${TASK}.yml --exp $EXP_DIR/$TASK/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS} --seed ${SEED} --preprocess ${PREPROCESS} --ni

export EXP_DIR=./results
export N_STEPS=1000
export RUN_NAME=run_2
export PRIOR_TYPE=f_phi_prior
export CAT_F_PHI=_cat_f_phi
export F_PHI_TYPE=f_phi_supervised  #f_phi_self_supervised
export MODEL_VERSION_DIR=card_onehot_conditional_results/${N_STEPS}steps/nn/${RUN_NAME}/${PRIOR_TYPE}${CAT_F_PHI}/${F_PHI_TYPE}
export LOSS=card_onehot_conditional
export TASK="$TMP_NAME"
export N_SPLITS=1
export DEVICE_ID=0
export N_THREADS=8
export PREPROCESS=grayscaled
export MLP_IDX=2
python main.py --mlp_idx ${MLP_IDX} --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config configs/${TASK}.yml --exp $EXP_DIR/$TASK/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS} --seed ${SEED} --preprocess ${PREPROCESS} --ni

export EXP_DIR=./results
export N_STEPS=1000
export RUN_NAME=run_3
export PRIOR_TYPE=f_phi_prior
export CAT_F_PHI=_cat_f_phi
export F_PHI_TYPE=f_phi_supervised  #f_phi_self_supervised
export MODEL_VERSION_DIR=card_onehot_conditional_results/${N_STEPS}steps/nn/${RUN_NAME}/${PRIOR_TYPE}${CAT_F_PHI}/${F_PHI_TYPE}
export LOSS=card_onehot_conditional
export TASK="$TMP_NAME"
export N_SPLITS=1
export DEVICE_ID=0
export N_THREADS=8
export PREPROCESS=grayscaled
export MLP_IDX=3
python main.py --mlp_idx ${MLP_IDX} --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config configs/${TASK}.yml --exp $EXP_DIR/$TASK/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS} --seed ${SEED} --preprocess ${PREPROCESS} --ni

export EXP_DIR=./results
export N_STEPS=1000
export RUN_NAME=run_4
export PRIOR_TYPE=f_phi_prior
export CAT_F_PHI=_cat_f_phi
export F_PHI_TYPE=f_phi_supervised  #f_phi_self_supervised
export MODEL_VERSION_DIR=card_onehot_conditional_results/${N_STEPS}steps/nn/${RUN_NAME}/${PRIOR_TYPE}${CAT_F_PHI}/${F_PHI_TYPE}
export LOSS=card_onehot_conditional
export TASK="$TMP_NAME"
export N_SPLITS=1
export DEVICE_ID=0
export N_THREADS=8
export PREPROCESS=grayscaled
export MLP_IDX=4
python main.py --mlp_idx ${MLP_IDX} --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config configs/${TASK}.yml --exp $EXP_DIR/$TASK/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS} --preprocess ${PREPROCESS} --ni
