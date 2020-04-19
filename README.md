# Visual-Navigation-for-Deepracer-car

**This project is finished in 02/2019, Author: Wenjian Hao, Mechanical Engineering, Clemson University** <br />

__Why we did this project?__ <br />
- *The basical target of this project is to make path following more effectiveby by using cnn model to get angle and velocity for the rc car directly in real-time.*

__How to deal with raw rosbag to get training data?__<br />
- *Data collecting is completed by adopting rosbag recording: Controlling the rc car manually, recording the rostopic: image, angle, velocity, which is the model training input->images, output->angle, velocity*<br />
- *The problem happened in data processing is that the frequence between image and control is different caused by ROS and maybe signal*<br />
- *Sol: I mapped the lastest image to each control(angle, velocity) and if none use the ima of the last control to solve the problem, the code is located in /Data_processing*

__How to use this code?__<br />
- *if you wanna use this code, you need to record the rosbag first, and use the codes in /Data_processing to get images and control corresponding to each image which is a .csv file. Then put the data in folder'/input'. Finally run the model_train.py to start training. you can also test the model using codes in /Trained_model_test*

__I offered both pytorch and keras training in the code in case you need to transfer the model to tensorflow.pb file, you can choose which package to train the model in the code.__
