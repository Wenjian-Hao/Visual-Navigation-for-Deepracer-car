from keras.models import load_model
import cv2
import numpy as np
import csv
import random
import matplotlib.pyplot as plt
import operator

testable = []
angle = []
thro = []
with open('input/ima_angle_th.csv') as test:
    tb = csv.reader(test)
    for line in tb:
        testable.append(line)
test = np.array(testable)
ima = test[:,0]
angle = test[:,1]
thro = test[:,2]
angle = np.array(angle)
thro = np.array(thro)

model = load_model('keras_model/train_models_keras_debug.h5')

def read_ima(ima_name): 
    name = 'input/IMA/' + ima_name + '.jpg'
    get_images = cv2.imread(name)
    get_images = cv2.cvtColor(get_images, cv2.COLOR_BGR2RGB)

    return get_images

def ima_normal(get_images):
    height, _, _ = get_images.shape
    get_images = get_images[int(height/2):, :, :]
    get_images = cv2.GaussianBlur(get_images, (3,3), 0)
    get_images = cv2.resize(get_images, (120, 160))
    get_images = np.reshape(get_images, [1, 160, 120, 3])
    get_images = (get_images / 255) - 0.5

    return get_images


rangsize = len(angle)
angle_error = []
velo_error = []
angle_pred = []
velo_pred = []
angle_act = []
velo_act = []

for i in range(rangsize):
    k = random.randint(0, (len(angle) - 1))
    ima_name = ima[k]
    img = read_ima(ima_name)
    img = ima_normal(img)

    out1, out2 = model.predict(img)
    out1 /= 10
    out2 /= 10
    err_angle = abs(round(angle[k].astype(np.float), 4) - np.round(out1, 4))
    err_velo = abs(round(thro[k].astype(np.float), 4) - np.round(out2, 4))

    angle_error.append(err_angle)
    velo_error.append(err_velo)
    angle_pred.append(np.round(out1, 4))
    velo_pred.append(np.round(out2, 4))
    angle_act.append(round(angle[k].astype(np.float), 4))
    velo_act.append(round(thro[k].astype(np.float), 4))

#-------------------------------------------------------------
# convert results dimension from (n, 1, 1) to (n,)
angle_error1 = []
velo_error1 = []
for ele in angle_error:
    for ele1 in ele:
        for ele2 in ele1:
            angle_error1.append(ele2)

for ele in velo_error:
    for ele1 in ele:
        for ele2 in ele1:
            velo_error1.append(ele2)

angle_error = angle_error1
velo_error = velo_error1

angle_pred1 = []
velo_pred1 = []
for ele in angle_pred:
    for ele1 in ele:
        for ele2 in ele1:
            angle_pred1.append(ele2)

for ele in velo_pred:
    for ele1 in ele:
        for ele2 in ele1:
            velo_pred1.append(ele2)

angle_pred = angle_pred1
velo_pred = velo_pred1

# angle_act1 = []
# velo_act1 = []
# for ele in angle_act:
#     for ele1 in ele:
#         for ele2 in ele1:
#             angle_act1.append(ele2)

# for ele in velo_act:
#     for ele1 in ele:
#         for ele2 in ele1:
#             velo_act1.append(ele2)

# angle_act = angle_act1
# velo_act = velo_act1
#---------------------------------------------------

print('The angle error is: +-', np.array(angle_error).mean() + np.array(angle_error).std())
print('The velocity error is: +-', np.array(velo_error).mean() + np.array(velo_error).std())

fig, ax = plt.subplots(2, 2, figsize=(24,16))
ax[0,0].plot(angle_error, 'r')
ax[0,0].set_ylabel('Angle Error')
ax[0,0].set_title('Angle Error')
ax[0,1].plot(velo_error, 'r')
ax[0,1].set_ylabel('Velocity Error')
ax[0,1].set_title('Velocity Error')

ax[1,0].plot(angle_pred)
ax[1,0].plot(angle_act, 'r')
ax[1,0].set_ylabel('Angle predicted')
ax[1,0].set_title('Angle predicted (blue)')
ax[1,1].plot(velo_pred)
ax[1,1].plot(velo_act, 'r')
ax[1,1].set_ylabel('Velocity predicted (blue)')
ax[1,1].set_title('Velocity predicted')

plt.show()