#!/bin/sh

DEVICE=0
FASTA=
while getopts "d:f:o:" args; do
    case $args in 
        d) DEVICE=${OPTARG}
        ;;
        f) FASTA=${OPTARG}
        ;;
        o) OUTPUT=${OPTARG}
        ;;
    esac
done

echo "DEVICE: $DEVICE, FASTA: $FASTA, OUTPUT: $OUTPUT"
dscript embed --seqs ${FASTA} --outfile ${OUTPUT} -d ${DEVICE}