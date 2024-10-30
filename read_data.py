# import OS
import os
import matplotlib.pyplot as plt

cent_1 = []
cent_2 = []
cent_5 = []
cent_20 = []
cent_50 = []
cent_100 = []
cent_200 = []
    
for x in os.listdir("data_raw/"):
    if x.startswith("1_"):
        cent_1 += [x]
    elif x.startswith("2_"):
        cent_2 += [x]
    elif x.startswith("5_"):
        cent_5 += [x]
    elif x.startswith("20_"):
        cent_20 += [x]
    elif x.startswith("50_"):
        cent_50 += [x]
    elif x.startswith("100_"):
        cent_100 += [x]
    elif x.startswith("200_"):
        cent_200 += [x]

print("Audios Cent 1: ", len(cent_1))
print("Audios Cent 2: ", len(cent_2))
print("Audios Cent 5: ", len(cent_5))
print("Audios Cent 20: ", len(cent_20))
print("Audios Cent 50: ", len(cent_50))
print("Audios Cent 100: ", len(cent_100))
print("Audios Cent 200: ", len(cent_200))

audios = (len(cent_1),len(cent_2),len(cent_5),len(cent_20),len(cent_50),len(cent_100),len(cent_200))
print("Audios Total: ",sum(audios))

numbers = []
file_1 = cent_1[0]
for line in open("data_raw/" + file_1):
    number = line.split(',')
    for i in range(len(number)):
        number[i] = float(number[i])
    numbers += number
    xs = [x/204000.0 for x in range(len(numbers))]
    plt.plot(xs, numbers, label=file_1)

numbers = []
file_2 = cent_1[1]
for line in open("data_raw/" + file_2):
    number = line.split(',')
    for i in range(len(number)):
        number[i] = float(number[i])
    numbers += number
    xs = [x/204000.0 for x in range(len(numbers))]
    plt.plot(xs, numbers, label=file_2)

numbers = []
file_3 = cent_1[2]
for line in open("data_raw/" + file_3):
    number = line.split(',')
    for i in range(len(number)):
        number[i] = float(number[i])
    numbers += number
    xs = [x/204000.0 for x in range(len(numbers))]
    plt.plot(xs, numbers, label=file_3)

numbers = []
file_4 = cent_1[3]
for line in open("data_raw/" + file_4):
    number = line.split(',')
    for i in range(len(number)):
        number[i] = float(number[i])
    numbers += number
    xs = [x/204000.0 for x in range(len(numbers))]
    plt.plot(xs, numbers, label=file_4)

numbers = []
file_5 = cent_1[4]
for line in open("data_raw/" + file_5):
    number = line.split(',')
    for i in range(len(number)):
        number[i] = float(number[i])
    numbers += number
    xs = [x/204000.0 for x in range(len(numbers))]
    plt.plot(xs, numbers, label=file_5)

numbers = []
file_6 = cent_1[5]
for line in open("data_raw/" + file_6):
    number = line.split(',')
    for i in range(len(number)):
        number[i] = float(number[i])
    numbers += number
    xs = [x/204000.0 for x in range(len(numbers))]
    plt.plot(xs, numbers, label=file_6)

numbers = []
file_7 = cent_1[6]
for line in open("data_raw/" + file_7):
    number = line.split(',')
    for i in range(len(number)):
        number[i] = float(number[i])
    numbers += number
    xs = [x/204000.0 for x in range(len(numbers))]
    plt.plot(xs, numbers, label=file_7)

plt.legend()
plt.xlabel("s")
plt.ylabel("dB?")
plt.title("dB vs s")
plt.show()


# Make sure to close the plt object once done
plt.close()