# Visual-Navigation-for-Deepracer-car

This project is finished in 02/2019, Author: Wenjian Hao, Mechanical Engineering, Clemson University.\\
The basical target of this project is to make path following more effectiveby by using cnn model to get angle and velocity for the rc car in real-time.

Data collecting is completed by adopting rosbag recording: Controlling the rc car manually, recording the rostopic: image, angle, velocity, which is the model training input->images, output->angle, velocity.\\
The problem happened in data processing is that the frequence between image and control is different caused by ROS and maybe signal, i mapped the lastest image for each control(angle, velocity) and if none use the ima of the last control to solve the problem, the code is located in /Data_processing

if you wanna use this code, you need to record the rosbag first, and use the codes in /Data_processing to get images and control corresponding to each image which is a .csv file. Finally run the model_train.py to start training. you can also test the model use files in /Trained_model_test

i offered both pytorch and keras training in the code in case you need to transfer the model to tensorflow.pb file
