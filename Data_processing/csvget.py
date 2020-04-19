"""
Copyright (c) 2019, by the Authors: wenjian hao / clemson university

The following code will convert the training data got from rosbag to the suitable training set
in keras or pytorch which is to solve the problem that the frequence of image topic is higher than that of
 angle and velocity 

This script match steering angles and velocity of rc car to images rostopic by ros timestamp

Matching principle: choose the image most close to the time before the steering

"""

import csv
import numpy as np
import string

angthro = []
with open('manual_drive.csv') as angle_thro:
    anth = csv.reader(angle_thro)
    for line in anth:
        angthro.append(line)
angthro = np.array(angthro)

imas = []
with open('ima_time.csv') as imatime:
    imat = csv.reader(imatime)
    for line in imat:
        imas.append(line) 
imacopy = np.array(imas)

time_table = np.zeros([len(angthro), 1])
tab_store = np.zeros([9,1])
tab_store = []

for i in range(0, len(angthro)):
    if len(imacopy) >= 9:    
        for j in range(0,9):
            if imacopy[j,0] < angthro[i, 0]:
                tab_store.append(imacopy[j,0])

        tab_store = np.asarray(tab_store, dtype=float)
        time_table[i,0] = np.max(tab_store)
        leng = len(tab_store)
        # print(time_table[i,0])
        # print(i)
        if leng > 3:
            imacopy = np.delete(imacopy, [0,(leng-2)], 0)
        tab_store = []
    
    elif len(imacopy) < 9:
        for j in range(0,len(imacopy)):
            if imacopy[j,0] < angthro[i, 0]:
                tab_store.append(imacopy[j,0])

        tab_store = np.asarray(tab_store, dtype=float)
        time_table[i,0] = np.max(tab_store)
        leng = len(tab_store)
        if leng > 3:
            imacopy = np.delete(imacopy, [0,(leng-1)], 0)
        tab_store = []

# with open('strddd.csv', 'w') as csvfile:
#     filewriter = csv.writer(csvfile, delimiter = ',')
#     for line in time_table:
#         csvfile.writelines(str('%5d'%line) + '\n')

csv_output = []

oop =[]
oop.append(time_table[:,0])
oop = np.array(oop).T

csv_output = np.zeros([len(angthro), 3])
csv_output[:, 0] = (oop[:,0])
csv_output[:, 1] = (angthro[:, 1])
csv_output[:, 2] = (angthro[:, 2])

np.savetxt('ima_angle_th.csv', csv_output, fmt="%.2f,%.11f,%.11f")
print('time_table: ', np.shape(time_table))
print('output shape', np.shape(csv_output))

# #csv_output = np.str(csv_output)
# with open('ima_table.csv', 'w') as csvfile:
#     filewriter = csv.writer(csvfile, delimiter = ' ')
#     for line in csv_output:
#         csvfile.writelines(str(line) + '\n')

print('DONE !')


    

