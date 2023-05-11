#!/bin/bash
#SBATCH --array=1-5
#SBATCH -p rise # partition (queue)
#SBATCH --nodelist=zanino
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=6 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH -t 1-24:00 # time requested (D-HH:MM)

pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate adahess
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
cd /home/eecs/yefan0726/ww_train_repos/adahessian/transformer

root=/home/eecs/yefan0726/ww_train_repos/adahessian/transformer
cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${root}/config/txt_file/adam.txt)
SEED=$(echo $cfg | cut -f 1 -d ' ')

ARCH=transformer_iwslt_de_en_v2
PROBLEM=iwslt14_de_en

DATA_PATH=/data/yefan0726/data/nlp/mt/data-bin/iwslt14.tokenized.de-en.joined
model=transformer

OUTPUT_PATH=/data/yefan0726/checkpoints/ww_train/nlp/mt/adahessian/adam/${ARCH}_${PROBLEM}_seed${SEED}
NUM=5
mkdir -p $OUTPUT_PATH

python train.py ${DATA_PATH} \
                --seed ${SEED} \
                --adam-eps 1e-08 \
                --arch ${ARCH} --share-all-embeddings \
                --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
                --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
                --criterion label_smoothed_cross_entropy \
                --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 8000 \
                --lr 0.0015 --min-lr 1e-9 \
                --label-smoothing 0.1 --weight-decay 0.0001 \
                --max-tokens 4096 --save-dir ${OUTPUT_PATH} \
                --update-freq 1 --no-progress-bar --log-interval 50 \
                --ddp-backend no_c10d \
                --keep-last-epochs ${NUM} --max-epoch 55 \
                --restore-file ${OUTPUT_PATH}/checkpoint_best.pt \
                | tee -a ${OUTPUT_PATH}/train_log.txt

python scripts/average_checkpoints.py --inputs ${OUTPUT_PATH} --num-epoch-checkpoints ${NUM} --output ${OUTPUT_PATH}/averaged_model.pt

BEAM_SIZE=5
LPEN=1.0
TRANS_PATH=${OUTPUT_PATH}/trans
RESULT_PATH=${TRANS_PATH}/

mkdir -p $RESULT_PATH
CKPT=averaged_model.pt

python generate.py \
    ${DATA_PATH} \
    --path ${OUTPUT_PATH}/${CKPT} \
    --batch-size 128 \
    --beam ${BEAM_SIZE} \
    --lenpen ${LPEN} \
    --remove-bpe \
    --log-format simple \
    --source-lang de \
    --target-lang en \
    --quiet \
> ${RESULT_PATH}/res.txt

ssh yefan0726@watson.millennium.berkeley.edu "mkdir -p ${OUTPUT_PATH}"
scp -r ${OUTPUT_PATH}/trans yefan0726@watson.millennium.berkeley.edu:${OUTPUT_PATH}
