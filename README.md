# Discussiontracker_lrec2020
This repository contains code for experiments on argumentation and specificity for the paper "The Discussion Tracker Corpus of Collaborative Argumentation", Olshefski, C., Lugini, L., Singh, R., Litman, D., & Godley, A.. In Proceedings of The 12th Language Resources and Evaluation Conference (pp. 1033-1043).

The Discussion Tracker corpus can be downloaded from www.discussiontracker.cs.pitt.edu.

## Usage

1. Install Speciteller (https://github.com/jjessyli/speciteller)
2. Copy the get_speciteller_features function from utils.py into speciteller.py
3. Set initial parameters in multitask_experiment.py:
    - the folder containing the Discussion Tracker corpus
    - the path to Glove embeddings
    - the experiment mode (argumentation, specificity, or multitask)
4. Run multitask_experiment.py