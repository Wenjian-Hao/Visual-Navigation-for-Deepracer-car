'''

This script extracts images from ros bag files and stores them in specified folder 
with image names as corresponding timestamps. 

This script is the modified version of bag_to_csv script written by Nick Speal in May 2013 at McGill University's Aerospace Mechatronics Laboratory
www.speal.ca

'''




'''
10/2019 Clemson University Mechanical Engineering
  i make some changes to get the images and steering data from bag
'''

import rosbag
import sys
import csv
import time
import string
import os #for file management make directory
import shutil #for file management, copy file
from cv_bridge import CvBridge
import cv2
import numpy as np

#verify correct input arguments: 1 or 2
if len(sys.argv) > 2:
    print ("invalid number of arguments:   " + str(len(sys.argv)))
    print ("should be 2: 'bag2csv.py' and 'bagName'")
    print ("or just 1  : 'bag2csv.py'")
    sys.exit(1)
elif len(sys.argv) == 2:
    listOfBagFiles = [sys.argv[1]]
    numberOfFiles = "1"
    print ("reading only 1 bagfile: " + str(listOfBagFiles[0]))
elif len(sys.argv) == 1:
    listOfBagFiles = [f for f in os.listdir(".") if f[-4:] == ".bag"]	#get list of only bag files in current dir.
    numberOfFiles = str(len(listOfBagFiles))
    print ("reading all " + numberOfFiles + " bagfiles in current directory: \n")
    for f in listOfBagFiles:
        print (f)
    print ("\n press ctrl+c in the next 2 seconds to cancel \n")
    time.sleep(2)
else:
    print ("bad argument(s): " + str(sys.argv))	#shouldnt really come up
    sys.exit(1)

count = 0
bridge = CvBridge()
for bagFile in listOfBagFiles:
    count += 1
    print("reading file " + str(count) + " of  " + numberOfFiles + ": " + bagFile)
    #access bag
    bag = rosbag.Bag(bagFile)
    bagContents = bag.read_messages()
    bagName = bag.filename


    #create a new directory
    folder = string.rstrip(bagName, ".bag")
    try:	#else already exists
        os.makedirs(folder)
    except:
        pass
    shutil.copyfile(bagName, folder + '/' + bagName)


    #get list of topics from the bag
    listOfTopics = []
    for topic, msg, t in bagContents:
        if topic not in listOfTopics:
            listOfTopics.append(topic)

    #get images and name images
    for topicName in listOfTopics:
        #Create a new CSV file for each topic
        filename = folder + '/' + 'images_time' + '.csv'
        if topicName=='/video_mjpeg':
            print(topicName)

            with open(filename, 'w+') as csvfile:
                filewriter = csv.writer(csvfile, delimiter = ',')
                firstIteration = True	#allows header row
                for subtopic, msg, t in bag.read_messages(topicName):	# for each instant in time that has data for topicName
                    if subtopic=='/video_mjpeg':
                        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                        #print(type(t))
                        #print(str(t))
                        timestr = "%.2f" % t.to_sec()
                        csvfile.writelines(timestr+'\n')
                        cv2.imwrite(timestr+'.jpg',cv_img)

    #get steering data
    for topicName in listOfTopics:
        #Create a new CSV file for each topic
        filename = folder + '/' + 'steering' + '.csv'
        if topicName=='/manual_drive':
            print(topicName)

            with open(filename, 'w+') as csvfile:
                filewriter = csv.writer(csvfile, delimiter = ',')
                firstIteration = True	#allows header row
                for subtopic, msg, t in bag.read_messages(topicName):	# for each instant in time that has data for topicName
                    if subtopic=='/manual_drive':
                        timestr = "%.2f" % t.to_sec()
                        angle = msg.angle
                        throttle = msg.throttle
                        csvfile.writelines(timestr + ',' + str(angle) + ',' + str(throttle) +'\n')

    bag.close()
print ("Done Everything")