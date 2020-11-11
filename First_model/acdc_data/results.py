import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import segmentation_models as sm

from segmentation_models.losses import dice_loss
from segmentation_models.metrics import iou_score

from keras.layers import Input, Conv2D
from keras.models import Model
from keras.optimizers import Adam

path = ""

x_train = np.load(path+"x_2d_train.npy")
y_train = np.load(path+"y_2d_train.npy")

x_val = np.load(path+"x_2d_val.npy")
y_val = np.load(path+"y_2d_val.npy")

x_test = np.load(path+"x_2d_test.npy")
y_test = np.load(path+"y_2d_test.npy")

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

### standardization of train
for i in range(x_train.shape[0]):
    x_train[i,:,:,0] = x_train[i,:,:,0]/(np.max(x_train[i,:,:,0]))

### standardization of val
for i in range(x_val.shape[0]):
    x_val[i,:,:,0] = x_val[i,:,:,0]/(np.max(x_val[i,:,:,0]))

### standardization of test
for i in range(x_test.shape[0]):
    x_test[i,:,:,0] = x_test[i,:,:,0]/(np.max(x_test[i,:,:,0]))

dim1,dim2,dim3,_ = y_train.shape
y_train_new = np.zeros(shape = (dim1,dim2,dim3,4))


for i in range(y_train.shape[0]):
    y_train_new[i,:,:,0] = (y_train[i,:,:,0]==1)
    y_train_new[i,:,:,1] = (y_train[i,:,:,0]==2)
    y_train_new[i,:,:,2] = (y_train[i,:,:,0]==3)
    y_train_new[i,:,:,3] = (y_train[i,:,:,0]==0)


dim1,dim2,dim3,_ = y_val.shape
y_val_new = np.zeros(shape = (dim1,dim2,dim3,4))


for i in range(y_val.shape[0]):
    y_val_new[i,:,:,0] = (y_val[i,:,:,0]==1)
    y_val_new[i,:,:,1] = (y_val[i,:,:,0]==2)
    y_val_new[i,:,:,2] = (y_val[i,:,:,0]==3)
    y_val_new[i,:,:,3] = (y_val[i,:,:,0]==0)



dim1,dim2,dim3,_ = y_test.shape
y_test_new = np.zeros(shape = (dim1,dim2,dim3,4))

for i in range(y_test.shape[0]):
    y_test_new[i,:,:,0] = (y_test[i,:,:,0]==1)
    y_test_new[i,:,:,1] = (y_test[i,:,:,0]==2)
    y_test_new[i,:,:,2] = (y_test[i,:,:,0]==3)
    y_test_new[i,:,:,3] = (y_test[i,:,:,0]==0)


y_train = y_train_new
y_val = y_val_new
y_test = y_test_new



bb = "vgg16"
input_shape = (256,256,3)
c = 4
enc_weights = "imagenet"
activation = "softmax"
base_model = sm.Unet(backbone_name = bb, input_shape = input_shape, classes = c, activation=activation, encoder_weights = enc_weights)

# define number of channels
N = x_train.shape[-1] # = 1,
inp = Input(shape=(None, None, N))
l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
out = base_model(l1)

modelUnet = Model(inp, out, name=base_model.name)

lr = 0.0001
optim = Adam(lr)

dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 0.5]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

modelUnet.compile(optim, total_loss, metrics)

modelUnet.load_weights("modelUnet_acdc_100epochs.keras")



def back_to_1_channel_mask(img, alpha=0.5):
    yy = np.zeros(shape=(1,256,256,1))
    yy += 1.0*(img[:,:,:,0:1]>=alpha)
    yy += 2.0*(img[:,:,:,1:2]>=alpha)
    yy += 3.0*(img[:,:,:,2:3]>=alpha)
    return yy

def plot_three(n=0):
    y_pred_train = modelUnet.predict(x_train[n:(n+1),:,:,:])
    y_pred_val = modelUnet.predict(x_val[n:(n+1),:,:,:])
    y_pred_test = modelUnet.predict(x_test[n:(n+1),:,:,:])

    alpha = 0.5
    y_pred_train = back_to_1_channel_mask(y_pred_train,alpha)
    y_pred_val = back_to_1_channel_mask(y_pred_val,alpha)
    y_pred_test = back_to_1_channel_mask(y_pred_test,alpha)

    ytrain = back_to_1_channel_mask(y_train[n:(n+1),:,:,:],alpha)
    yval = back_to_1_channel_mask(y_val[n:(n+1),:,:,:],alpha)
    ytest = back_to_1_channel_mask(y_test[n:(n+1),:,:,:],alpha)

    plt.figure()
    plt.subplot(3,3,1)
    tit = "Train Image num " + str(n)
    plt.title(tit)
    plt.imshow(x_train[n,:,:,0], 'gray', interpolation='none')
    plt.subplot(3,3,2)
    plt.title("True Mask")
    plt.imshow(x_train[n,:,:,0], 'gray', interpolation='none')
    plt.imshow(ytrain[0,:,:,0], 'jet', interpolation='none', alpha=0.7)
    plt.subplot(3,3,3)
    plt.title("Pred Mask")
    plt.imshow(x_train[n,:,:,0], 'gray', interpolation='none')
    plt.imshow(y_pred_train[0,:,:,0], 'jet', interpolation='none', alpha=0.7)

    plt.subplot(3,3,4)
    tit = "Val Image num " + str(n)
    plt.title(tit)
    plt.imshow(x_val[n,:,:,0], 'gray', interpolation='none')
    plt.subplot(3,3,5)
    plt.title("True Mask")
    plt.imshow(x_val[n,:,:,0], 'gray', interpolation='none')
    plt.imshow(yval[0,:,:,0], 'jet', interpolation='none', alpha=0.7)
    plt.subplot(3,3,6)
    plt.title("Pred Mask")
    plt.imshow(x_val[n,:,:,0], 'gray', interpolation='none')
    plt.imshow(y_pred_val[0,:,:,0], 'jet', interpolation='none', alpha=0.7)

    plt.subplot(3,3,7)
    tit = "Test Image num " + str(n)
    plt.title(tit)
    plt.imshow(x_test[n,:,:,0], 'gray', interpolation='none')
    plt.subplot(3,3,8)
    plt.title("True Mask")
    plt.imshow(x_test[n,:,:,0], 'gray', interpolation='none')
    plt.imshow(ytest[0,:,:,0], 'jet', interpolation='none', alpha=0.7)
    plt.subplot(3,3,9)
    plt.title("Pred Mask")
    plt.imshow(x_test[n,:,:,0], 'gray', interpolation='none')
    plt.imshow(y_pred_test[0,:,:,0], 'jet', interpolation='none', alpha=0.7)
    #plt.show()
    plot_name = "plot/plot_" + str(n) + ".png"
    plt.savefig(plot_name)
#
# print("here")
# n=0
# y_pred_train = modelUnet.predict(x_train[n:(n+1),:,:,:])
# print(np.shape(y_pred_train))


k = 5
for i in range(k):
        m = i
        print("plotting ", m+1 , " out of ", k )
        plot_three(i)
