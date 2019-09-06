#!/bin/bash
# Adapted from Anton Ragni 2019

if [ $# -ne 4 ]; then
    echo "Usage: $0 emb-size emb-dir corpus-path word2vec-dir"
    exit 100
fi

SIZE=$1
EMB=$2
CORPUS=$3
word2vec=$4

if [ -d $EMB ]; then
    echo "Target dir $EMB exists - Exit without embedding"
    exit 100
fi
mkdir -p $EMB

$word2vec -train $CORPUS -output $EMB/embedding.txt -size $SIZE -cbow 1 &> $EMB/embedding.log
