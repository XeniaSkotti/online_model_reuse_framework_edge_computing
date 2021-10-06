import pandas as pd
from csv import reader
import ast
from datetime import datetime as dt
import os

## The data used are publicly available at:
### - https://archive.ics.uci.edu/ml/datasets/GNFUV+Unmanned+Surface+Vehicles+Sensor+Data+Set+2 (experiment 1 & 2)
### - https://archive.ics.uci.edu/ml/datasets/GNFUV+Unmanned+Surface+Vehicles+Sensor+Data (experiment 3)

## Did some refactoring to the original data folders but you can change the code according to your system.
## Also I renamed the files of experiment 3 to follow the same convention as experiments 1 & 2

# os.chdir("..")

## Each experiment has four nodes, pi2, pi3, pi4 and pi5
nodes = ["pi" + str(i) for i in range(2,6)]
## and the following attributes are recorded by the each nodes device:
### - humidity
### - temperature
### - time of the recording

## therefore each data row should record the following
attributes = ["device", "humidity", "temperature", "time", "pi", "experiment"]

def read_data():
    experiments_data = []
    for i in range(1,4):
        experiment =  "experiment_" + str(i)
        directory = "//".join([os.getcwd(), "data", experiment])
        experiment_files = os.listdir(directory)
        data = []
        for file in experiment_files:
            name = file.split(".")[0]
            if name in nodes:
                node_directory = directory + "//" + file
                if ".csv" in file:
                    with open(node_directory, 'r') as f:
                        file_data = []
                        # pass the file object to reader() to get the reader object
                        csv_reader = reader(f)
                        # Iterate over each row in the csv using reader object
                        for row in csv_reader:
                            dictionary = ast.literal_eval(",".join(row))
                            file_data.append(dictionary)
                        ## create a dataframe from the rows of dictionaries and drop rows with NaN values
                        node_data = pd.DataFrame(file_data).dropna()
                        node_data["pi"] = name
                        node_data.time = [dt.fromtimestamp(epoch_time).strftime("%d-%m-%Y, %H:%M:%S.%f") \
                                          for epoch_time in node_data.time.values]
                        node_data.humidity = node_data.humidity.astype(int)
                        node_data.temperature = node_data.temperature.astype(int)

                elif ".xlsx" in file:
                    node_data = pd.read_excel(node_directory).dropna()
                    node_data.columns= node_data.columns.str.lower()
                    node_data = node_data.loc[~(node_data.humidity == " None")]
                    node_data = node_data.loc[~(node_data.temperature == " None")]
                    node_data.device = node_data.device.str.replace("\'", "")
                node_data.pi = node_data.pi.str.lower()
                node_data.experiment = i
                data.append(node_data)

        experiments_data.append(pd.concat(data).reset_index(drop=True))
    return pd.concat(experiments_data).reset_index(drop=True)