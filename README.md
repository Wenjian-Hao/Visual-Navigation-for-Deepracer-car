# Visual-Navigation-for-RC-car

**This project is finished in 02/2019, Author: Wenjian Hao, Mechanical Engineering, Clemson University** <br />

__Why we did this project?__ <br />
- *The initial goal of this project is to make autonomous navigation more efficient by using CNN model to get 'angle and velocity control' of the RC car directly, which leads to good real-time performance.*

__How to deal with raw rosbag to get training data?__<br />
- *Data collecting is completed by adopting rosbag recording: Controlling the RC car manually, recording the rostopic: 'image, angle, velocity', which are the model training input->images, output->'angle, velocity'*<br />
- *The problem occured in data processing is that the frequence between rostopics of images and controls are different caused by ROS*<br />
- *Sol: I mapped the lastest image to each control(angle, velocity) command and if none, use the image corresponding to the last control to solve the problem, the codes is located in '/Data_processing'*

__How to use this code?__<br />
- *if you wanna use this code, you need to record the rosbag first, and use the code in '/Data_processing' to get images and control commands corresponding to each image, the result of the data processing is outputted as a 'xxx.csv' file. Then put the data and 'xxx.csv' file in folder '/input'. Finally run the model_train.py to start the training. You can also test the model using the codes in '/Trained_model_test'*

__I offered both pytorch and keras training methods for model training in case you need to transfer the model to tensorflow.pb file, you can choose any of the  methods for training the model by setting it as 'True' in the code.__
