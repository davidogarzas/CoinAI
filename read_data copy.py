# import OS
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

i=0

csvFile = open('./result.csv','w')
writer = csv.writer(csvFile)

def loadFiles():
    data = []
    for i, x in enumerate(os.listdir("./data_raw")):
        file_type = int(x.split("_")[0])
        filename = f"./data_raw/{x}"
        file_data = np.loadtxt(filename, delimiter=",",dtype=int).tolist()
        data.append(filename)
        data.append(file_type)
        data.extend(file_data)
        writer.writerow(data)
        print(i)
        data = []

df = loadFiles()
