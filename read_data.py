# import OS
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = []
i=0

def loadFiles():
    for i, x in enumerate(os.listdir("./data_raw")):
        file_type = int(x.split("_")[0])
        filename = f"./data_raw/{x}"
        file_data = pd.DataFrame(np.loadtxt(filename, delimiter=",",dtype=int))
        data.append((file_type, x, file_data))
        print(i)

    df = pd.DataFrame(data, columns=["Type", "Filename", "Data"])
    print(df)
    return df

def Plot(type):
    files = df[df['Type'] == type]['Data'].head(5)
    filenames = df[df['Type'] == type]['Filename'].head(5)
    i = 0
    for file in files:
        label = filenames[i]
        xs = [x/204000.0 for x in range(len(file))]
        i += 1
        plt.plot(xs, file,label=label)


    plt.legend()
    plt.xlabel("s")
    plt.ylabel("dB?")
    plt.title("dB vs s")
    plt.show()

    # Make sure to close the plt object once done
    plt.close()

df = loadFiles()
Plot(int(input("input a coin type to plot(1,5,20,50,100,200): ")))