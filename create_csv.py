# import OS
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

def find_max_size():
    i = 0
    max_size = 0
    for i, x in enumerate(os.listdir("./data_raw")):
        filename = f"./data_raw/{x}"
        file_data = np.loadtxt(filename, delimiter=",",dtype=int).tolist()
        if len(file_data) > max_size:
            max_size = len(file_data)
        print(i)
    return max_size

def create_csv(max_size):
    csvFile = open('./result.csv','w',newline='')
    writer = csv.writer(csvFile)
    data = []
    i=0
    for i, x in enumerate(os.listdir("./data_raw")):
        file_type = int(x.split("_")[0])
        filename = f"./data_raw/{x}"
        file_data = np.loadtxt(filename, delimiter=",",dtype=int).tolist()
        #data.append(filename)
        if (file_type == 1):
            file_type = 0
        elif (file_type == 2):
            file_type = 1
        elif (file_type == 20):
            file_type = 3
        elif (file_type == 50):
            file_type = 4
        elif (file_type == 100):
            file_type = 5
        elif (file_type == 200):
            file_type = 6
        data.append(file_type)
        data.extend(file_data)

        # agregar 0s para tener misma longitud de filas
        zeros_array = np.zeros(max_size-len(file_data)).tolist()
        data.extend(zeros_array)
        writer.writerow(data)
        print(i)
        data = []

max_size = find_max_size()
create_csv(max_size)