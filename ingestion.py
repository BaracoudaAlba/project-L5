import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    pd_list = []
    files_read = []
    for present_file in os.listdir(input_folder_path):
        if ".csv" in present_file:
            files_read.append(present_file)
            read_pandas = pd.read_csv(os.path.join(input_folder_path,present_file))
            pd_list.append(read_pandas)
    concatenated_pd = pd.concat(pd_list)
    print(os.path.isdir(output_folder_path))
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)

    concatenated_pd.drop_duplicates().to_csv(os.path.join(output_folder_path,"finaldata.csv" ), index=False)

    with open('ingestedfiles.txt', 'w') as f:
        f.write(str(files_read))

if __name__ == '__main__':
    merge_multiple_dataframe()
