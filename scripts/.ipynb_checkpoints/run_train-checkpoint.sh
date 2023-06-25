#!/bin/sh


TOPSY_TURVY=
while getopts "d:t:T:e:v:o:p:" args; do
    case $args in
        d) DEVICE=${OPTARG}
        ;;
        t) TRAIN=${OPTARG}
        ;;
        T) TEST=${OPTARG}
        ;;
        e) EMBEDDING=${OPTARG}
        ;;
        v) TOPSY_TURVY="--topsy_turvy --glider-weight 0.2 --glider_thres 0.925"
        ;;
        o) OUTPUT_FOLDER=${OPTARG}
        ;;
        p) OUTPUT_PREFIX=${OPTARG}
        ;;
    esac
done
    
if [ ! -d ${OUTPUT_FOLDER} ]; then mkdir -p $OUTPUT_FOLDER; done



dscript train --train $TRAIN --test $TEST --embedding $EMBEDDING $TOPSY_TURVY \
              --o ${OUTPUT_FOLDER}/results.log \
              --save-prefix  ${OUTPUT_FOLDER}/ep_ \
              --lr 0.001 --lambda 0.05 --num-epoch 10 \
              --weight-decay 0 --batch-size 25--pool-width 9 \
              --kernel-width 7 --dropout-p 0.2 --projection-dim 100 \
              --hidden-dim 50 --kernel-width 7