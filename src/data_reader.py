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

def read_data(dataset):
    if dataset == "gnfuv":
        return read_gnfuv_data()
    elif dataset == "bank":
        return read_banking_data()

def read_gnfuv_data():
    experiments_data = []
    for i in range(1,4):
        experiment =  "experiment_" + str(i)
        directory = "//".join([os.getcwd(), "data//GNFUV", experiment])
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

def read_banking_data():
    directory = "//".join([os.getcwd(), "data", "bank-marketing", "raw","bank-additional-full.csv"])
    df = pd.read_csv(directory, sep=";")

    for column in df.columns:
        column_type = str(df[column].dtype)
        if "int" not in column_type and "float" not in column_type:
            df[column] = df[column].astype("category")

    cat_columns = df.select_dtypes(['category']).columns
    df_codified = df.copy()
    df_codified[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    
    return df, df_codified

def read_sample_results(directory):
    sample_data = []
    no_data = []
    for i in range(1,101):
        filename = directory + f"/sample_{i}.csv"
        try:
            sample = pd.read_csv(filename)
            if not sample.empty:
                sample["sample"] = i
                sample_data.append(sample)
        except:
            no_data.append(f"sample_{i}")
    if len(no_data) > 0:
        print("The following samples had no similar pairs")
        [print(sample, end = ", ") for sample in no_data[:-1]]
        print(no_data[-1])
    return pd.concat(sample_data, ignore_index = True)

def read_gnfuv_sample_results():
    data = []
    for data_type in ["original", "standardised"]:
        directory = f"results/GNFUV/{data_type}"
        data.append(read_sample_results(directory))
    return data

def read_banking_sample_results():
    directory = f"results/bank-marketing/reduced"
    return read_sample_results(directory)