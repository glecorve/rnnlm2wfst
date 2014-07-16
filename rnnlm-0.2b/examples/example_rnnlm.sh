#!/bin/bash

#This is simple example how to use rnnlm tool for training and testing rnn-based language models
#Check 'example.output' how the output should look like
#SRILM toolkit must be installed for combination with ngram model to work properly

make clean
make

rm rnnlm.model
rm rnnlm.model.output.txt

#rnn model is trained here
time ./rnnlm -train rnnlm.train.txt -valid rnnlm.valid.txt -rnnlm rnnlm.model -hidden 80 -rand-seed 1 -debug 1 -class 100 -bptt 4 -bptt-block 10

#ngram model is trained here, using SRILM tools
ngram-count -text rnnlm.train -order 5 -lm rnnlm.arpa -kndiscount -interpolate -gt3min 1 -gt4min 1
ngram -lm rnnlm.arpa -order 5 -ppl rnnlm.test.txt -debug 2 > rnnlm.test.ppl

gcc convert.c -O2 -o convert
./convert <rnnlm.test.ppl >rnnlm.test.srilm.txt

#combination of both models is performed here
time ./rnnlm -rnnlm rnnlm.model -test rnnlm.test.txt -lm-prob rnnlm.test.srilm.txt -lambda 0.5
