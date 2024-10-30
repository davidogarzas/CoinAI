# import OS
import os
import matplotlib.axes as axes
import matplotlib.pyplot as plt
import pandas as pd
data = []
def loadFiles():
    for x in os.listdir("./data_raw"):
            file_type = int(x.split("_")[0])
            filename = x
            data.append((file_type, filename))
    df = pd.DataFrame(data)
    df.rename(columns={0:"Type",1:"File"}, inplace=True)
    print(df)
    return df

def Plot(type):
    files = df[df['Type'] == type]['File'].head(5)
    for file in files:
        numbers = []
        for line in open(f"./data_raw/{file}"):
            number = line.split(',')
            for i in range(len(number)):
                number[i] = float(number[i])
            numbers += number
        xs = [x/204000.0 for x in range(len(numbers))]
        plt.plot(xs, numbers, label=file)

    plt.legend()
    plt.xlabel("s")
    plt.ylabel("dB?")
    plt.title("dB vs s")
    plt.show()

    # Make sure to close the plt object once done
    plt.close()

df = loadFiles()
Plot(100)