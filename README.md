# STATS-507-FINAL-PROJECT

## Introduction
This is the final project for STATS 507. It contains several components and 
different python files to interact with the part of speech tagger in arabic. Different
neural network architectures are trained on the same dataset. This codebase contains a notebook
to setup a pipeline for testing and training of the neural network, a script to run all of our architectures
initially from the notebook (Pretrained AraT5V2, Simple Linear Layer, BiLSTM, and Encoder Decoder Transformer), and, finally, a cli tool named arabic_pos_tagger_cli. It's purpose is to train our
best model (in this case, a BiLSTM was chosen), tokenize arabic input, predict POS tags, and display them. The user can enter as many arabic utterances as needed, before closing the program with "end".

## arabic_pos_tagger_cli.py
To demo, run `python3.11 arabic_pos_tagger_cli.py`.

## STATS507_final_proposal.ipynb
To demo, run `python3.11 -m STATS507_final_proposal.ipynb`.

## dialects.py
To demo, run  `python3.11 dialects.py`.

## graphs.py
To demo, run  `python3.11 graphs.py`.



