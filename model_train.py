'''
02/2019 Clemson University, Mechanical Engineering

    About the codes
    1. Goal: Building a cnn model to drive AWS car autonomously in a building
    2. CNN structure: 'nvidia - end to end' 
    3. Data Environment: Recording training data using rosbag inside and outside around flour daniel, clemson university
    4. One input: image; Two labels: steering angles and throttle
    5. This codes offers two train methods: pytorch and keras, which are trained with the same training data

    Training method
    1. Use rosbag to record the steering angles and throttle topics corresponding to images rostopic
    2. Preprocessing the collecting data: Matching images with 'angles and throttle from the manual command' according to rostime stamp
    3. Start training the CNN network
    
'''

import numpy as np
import cv2
import csv
import torch
import torch.optim as optim
import torch.nn as nn
import random
import keras

from torch.utils import data
from torch.utils.data import DataLoader
from torch_based_model import NvidiaModel, visualize_model
from tfkeras_based_model import TFKNvidiaModel

from imgaug import augmenters as img_aug
from images_processing import traingimages_augment, read_ima, ima_normal, expandlabel
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#########################################################
#########################################################
## choose pytorch or keras                             ##
#########################################################
train_using_torch = False                              ##
train_using_tf_keras = True                          ##
#########################################################
#########################################################


##################################################################
#----------------------------------------------------------------#
# preprocessing training data                                    #
#----------------------------------------------------------------#
##################################################################
#load and split the training data
model_output_dir='trained_model/'

trainingdata = []
with open('input/ima_angle_th.csv') as oridata:
    rou_data = csv.reader(oridata)
    for line in rou_data:
        trainingdata.append(line)

train_len = int(0.7*len(trainingdata)) + 1
valid_len = int(0.3*len(trainingdata))
length = [train_len, valid_len]
train_samples, validation_samples = data.random_split(trainingdata, length)


#####################################
#####################################
## using torch
#####################################
#####################################
while train_using_torch:
    print('will use pytorch, output .h5 file')
    class Dataset(data.Dataset):

        def __init__(self, samples, transform = None):
            self.samples = samples
            self.transform = transform

        def __getitem__(self, index):
            batch_samples = self.samples[index]
            field_angle = float(batch_samples[1])
            field_throttle = float(batch_samples[2])
            trainima = read_ima(batch_samples[0])
            video_mjpeg, field_angle, field_throttle = traingimages_augment(trainima, field_angle, field_throttle)
            video_mjpeg = ima_normal(video_mjpeg)
            return (video_mjpeg, field_angle, field_throttle)

        def __len__(self):
            return len(self.samples)

    #change data to feed it in to dataloader of torch
    params = {'batch_size': 32,
            'shuffle': True,
            'num_workers': 4}

    training_set = Dataset(train_samples)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(validation_samples)
    validation_generator = data.DataLoader(validation_set, **params) 
    ##------------------------------------------------------------------------------##
    print('Have finished processing data and start loading it :)')
    ##------------------------------------------------------------------------------## 

    ##################################################################################
    ## Define nvidia model                                                          ##
    ##################################################################################
    
    # define training process
    model = NvidiaModel()
    visualize_model(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criteria = nn.MSELoss()
    training_device = torch.device('cuda') #use clemson university dgx online gpu
    print('we will use ', training_device)

    def feed_device(data, training_device):
        ima, angle, throttle = data
        return ima.float().to(training_device), angle.float().to(training_device), throttle.float().to(training_device)

    num_epoch = 1
    train_angle_loss = 0
    train_velo_loss = 0
    
    for epoch in range(num_epoch):
        print('Current epoch is: ', epoch)
        model.to(training_device)
        
        model.train()
        train_loss_sum = 0

        for train_batch, ima_angle_throttle in enumerate(training_generator):
            ima_angle_throttle = feed_device(ima_angle_throttle, training_device)
            data = [ima_angle_throttle]
            train_loss_sum = 0

            for batch_data in data:
                ima, angle, velocity = batch_data
                out1, out2 = model(ima)
                angle_loss = criteria(out1, angle.unsqueeze(1))
                velocity_loss = criteria(out2, velocity.unsqueeze(1))
                train_loss_sum = angle_loss + velocity_loss
                optimizer.zero_grad()
                train_loss_sum.backward()
                optimizer.step()
                train_loss_sum += train_loss_sum.data.item()

            if train_batch % 200 == 0:
                print('train loss is: %.9f' % (train_loss_sum/(train_batch+1)))

        model.eval()
        valid_loss = 0
        with torch.set_grad_enabled(False):
            for valid_batch, ima_angle_throttle in enumerate(validation_generator):
                ima_angle_throttle = feed_device(ima_angle_throttle, training_device)
                data = [ima_angle_throttle]
                
                for batch_data in data:
                    ima, angle, velocity = batch_data

                    out1, out2 = model(ima)
                    angle_loss = criteria(out1, angle.unsqueeze(1))
                    velocity_loss = criteria(out2, velocity.unsqueeze(1))
                    loss = angle_loss + velocity_loss
                    valid_loss += loss.data.item()

                if valid_batch % 200 == 0:
                    print('Valid Loss is: %.9f' % (valid_loss/(valid_batch+1)))
        
    # state = {
    #     'model': model.module if training_device == 'cuda' else model,
    #         }

    torch.save(model.state_dict(), model_output_dir + 'model_torch.h5')

    print("HAVE FINISHED TRAINING, PLEASE CHECK THE MODEL: 'model_torch.h5'")
    break

    # checkpoint = {'model': model(),
    #         'state_dict': model.state_dict(),
    #         'optimizer' : optimizer.state_dict()}

    # torch.save(checkpoint, 'model_torch.pth')


#####################################
#####################################
## using tensorflow.keras
#####################################
#####################################

while train_using_tf_keras:
    # set parameters
    num_episode = 100
    batch_size = 64
    evl_size = 32

    #generate datas
    def data_generator(samples, batch_size, is_training):
        while True:
            batch_images = []
            batch_angles = []
            batch_throttle = []

            for i in range(batch_size):
                index = random.randint(0, len(samples) - 1)
                batch_samples = samples[index]
                trainima = read_ima(batch_samples[0])
                field_angle = float(batch_samples[1])
                field_throttle = float(batch_samples[2])

                if is_training:
                    images, angless, throttle = traingimages_augment(trainima, field_angle, field_throttle)
                    
                images = ima_normal(images)
                batch_images.append(images)
                batch_angles.append(angless)
                batch_throttle.append(throttle)
                
            yield(np.asarray(batch_images), {'output_1': np.asarray(batch_angles), 'output_2': np.asarray(batch_throttle)})


    # loading model and loop 
    model = TFKNvidiaModel()

    history = model.fit_generator(data_generator(train_samples, batch_size=batch_size, is_training=True),
                              steps_per_epoch=64,
                              epochs=num_episode,
                              validation_data = data_generator(validation_samples, batch_size=batch_size, is_training=True),
                              validation_steps=64,
                              verbose=1,
                              shuffle=1)

    
    model.save(model_output_dir + 'model_keras.h5')
    print("HAVE FINISHED TRAINING, PLEASE CHECK THE MODEL: 'model_keras.h5'")
    break
