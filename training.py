import flask
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model():
    # get the data
    data_pd = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    #use this logistic regression for training
    LR = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    # fit the logistic regression to your data
    # first column exited not be used in modelling just 3 cols
    # exited is the prediction

    X = data_pd[data_pd.columns[1:4]]
    y = data_pd[['exited']]
    print(X.head(5))
    print(y.head(5))
    model = LR.fit(X, y)


    #write the trained model to your workspace in a file called trainedmodel.pkl
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    pickle.dump( file = open(os.path.join(model_path, "trainedmodel.pkl"), 'wb'), obj=model)


if __name__ == '__main__':
    train_model()