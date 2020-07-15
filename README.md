# Hate Speech Machine Learning

This project seeks to answer the following questions:  
1. Is it possible to effectively classify posts on online forums as hate speech or not hate speech? 
2. Is it possible to procedurally generate interventionary responses to such instances of online hate speech?

The data this project uses comes from [A Benchmark Dataset for Learning to Intervene in Online Hate Speech](https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech).

## To Run
1. To run the classifier, run `python3 main.py` in the directory containing main.py. To see the process we used to test different classification models and hyperparameters, uncomment the block of code involving the `performance_tester` variable in main.py before running the program. 
2. To run the Textgenrnn response generator, run `python3 text_generation.py` in the same directory.
3. To run the Sequence-to-Sequence response generator, run `python3 seq2seq.py` in the same directory.

## Noteworthy best versions of Python libraries
Our code works best using Keras version 2.2.4, installed through regular Pip (i.e. not Anaconda).