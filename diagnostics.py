
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
##################Function to get model predictions
def model_predictions():
    data_pd = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    #read the deployed model and a test dataset, calculate predictions
    with open(os.path.join(prod_deployment_path, "trainedmodel.pkl"),'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(data_pd[data_pd.columns[1:4]])
    return predictions.values.toList() #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    # mean median standard
    data_pd = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    print(data_pd.columns)
    data_pd.drop(['corporation','exited'], axis=1, inplace=True)
    means = data_pd.mean()
    medians = data_pd.median()
    stds = data_pd.std()

    return [means, stds, medians]#return value should be a list containing all summary statistics

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    timing_list = [] 
    starttime = timeit.default_timer()
    os.system('python training.py')
    timing = timeit.default_timer() - starttime
    timing_list.append(timing)

    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    timing = timeit.default_timer() - starttime
    timing_list.append(timing)
    print(timing_list)
    return timing_list #return a list of 2 timing values in seconds

def missing_data():
    # gets and counts the missing data
    data_pd = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    data_pd.drop(['corporation','exited'], axis=1, inplace=True)
    missing_list = (data_pd.isna().sum() / len(data_pd))* 100
    print(missing_list)
    return missing_list

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    outdated_list = subprocess.check_output(['pip', 'list', '--outdated'])
    print(outdated_list)
    return outdated_list


if __name__ == '__main__':
    # model_predictions()
    # dataframe_summary()
    # execution_time()
    # missing_data()
    outdated_packages_list()





    
