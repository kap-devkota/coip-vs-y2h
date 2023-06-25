#!/bin/sh

NET=../data/networks/dscript-tt
MODELD=$NET/models
EMB=$NET/y2h-coip.h5
PREF=RESULTS-
while getopts "m:d:" args; do
    case $args in
        m) MODE=$OPTARG
        ;;
        d) DEVICE=$OPTARG
        ;;
    esac
done

case $MODE in 
    1) MODEL=$MODELD/y2h-dscript;TRAIN=$NET/y2h_train.tsv;TEST=$NET/y2h_test.tsv;OPTIONS=""
    ;;
    2) MODEL=$MODELD/y2h-tt;TRAIN=$NET/y2h_train.tsv;TEST=$NET/y2h_test.tsv;OPTIONS="-v"
    ;;    
    3) MODEL=$MODELD/coip-dsript;TRAIN=$NET/coip_train.tsv;TEST=$NET/coip_test.tsv;OPTIONS=""
    ;;
    4) MODEL=$MODELD/coip-tt;TRAIN=$NET/coip_train.tsv;TEST=$NET/coip_test.tsv;OPTIONS="-v"
    ;;
esac

echo "TRAIN: $TRAIN, TEST: $TEST, EMB: $EMB, OPTIONS: $OPTIONS, MODEL: $MODEL PREFIX:$PREF, DEVICE:$DEVICE"
./run_train.sh -t $TRAIN -T $TEST -e $EMB $OPTIONS -o $MODEL -p $PREF -d $DEVICE
