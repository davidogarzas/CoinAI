import os
import numpy as np
import csv

#Max Size is 769024
def find_max_size():
    i = 0
    max_size = 0
    for i, x in enumerate(os.listdir("./data_raw")):
        filename = f"./data_raw/{x}"
        file_data = np.loadtxt(filename, delimiter=",",dtype=int).tolist()
        if len(file_data) > max_size:
            max_size = len(file_data)
        print(i)
    print(max_size)
    return max_size


def create_npy_file(max_size):
    csvFile = open('./result.csv','w',newline='')
    writer = csv.writer(csvFile)
    data = []
    data2d = [[]]
    i=0
    for i, x in enumerate(os.listdir("./data_raw")):
        file_type = int(x.split("_")[0])
        filename = f"./data_raw/{x}"
        file_data = np.loadtxt(filename, delimiter=",",dtype=int).tolist()

        # Maps coins types from 0 to 6
        if (file_type == 1):
            file_type = 0
        elif (file_type == 2):
            file_type = 1
        elif (file_type == 5):
            file_type = 2
        elif (file_type == 20):
            file_type = 3
        elif (file_type == 50):
            file_type = 4
        elif (file_type == 100):
            file_type = 5
        elif (file_type == 200):
            file_type = 6

        #data.append(filename)

        # Adds coin Type
        data.append(file_type)

        # Adds data
        data.extend(file_data)
        
        # Adds 0s to have same number of columns
        zeros_array = np.zeros(max_size-len(file_data)).tolist()
        data.extend(zeros_array)

        # Appends to 2d array
        #data2d.append(data)

        writer.writerow(data)

        # Empties data array
        data = []

        # Print advance
        print(i)

    # Saves .npy file
    #result_array = np.array(data2d)
    #np.save('coin_data', result_array)
    #print(result_array)

max_size = find_max_size()
create_npy_file(max_size)