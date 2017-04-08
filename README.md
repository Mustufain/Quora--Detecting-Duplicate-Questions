# Quora--Detecting-Duplicate-Questions
Machine Learning Kaggle competition on detecting duplicate questions in Quora Dataset 

The goal of this competition is to predict which of the provided pairs of questions contain two questions with the same meaning. The ground truth is the set of labels that have been supplied by human experts. The ground truth labels are inherently subjective, as the true meaning of sentences can never be known with certainty. Human labeling is also a 'noisy' process, and reasonable people will disagree. As a result, the ground truth labels on this dataset should be taken to be 'informed' but not 100% accurate, and may include incorrect labeling. We believe the labels, on the whole, to represent a reasonable consensus, but this may often not be true on a case by case basis for individual items in the dataset.

step 1:

Basic Feature Engineering and Initial Exploratory Analysis reveals us that similariy of question pairs cannot merely be judged on cosmetic level as there are no absolute patterns or corelations of features which we extracted are found with duplicaity of questions so we will dig one layer deeper to analyze the semantics of the sentence and then analyze their duplacity



