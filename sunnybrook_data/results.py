import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import segmentation_models as sm
from segmentation_models.losses import dice_loss
from segmentation_models.metrics import iou_score

from keras.layers import Input, Conv2D
from keras.models import Model

DATA_NAME = 'Data'
TRAIN_SOURCE = 'Train'
TEST_SOURCE = 'Test'

WORKING_DIR = os.getcwd()

TRAIN_DATA_DIR = os.path.join(WORKING_DIR, DATA_NAME, TRAIN_SOURCE)
TEST_DATA_DIR = os.path.join(WORKING_DIR, DATA_NAME, TEST_SOURCE)

def main():

	class GetData():
		def __init__(self, data_dir):
		    images_list =[]
		    labels_list = []

		    self.source_list = []

		    examples = 0
		    print("loading images")
		    label_dir = os.path.join(data_dir, "Labels")
		    image_dir = os.path.join(data_dir, "Images")
		    for label_root, dir, files in os.walk(label_dir):
		        for file in files:
		            if not file.endswith((".png", ".jpg", ".gif")):
		                continue
		            try:
		                folder = os.path.relpath(label_root, label_dir)
		                image_root = os.path.join(image_dir, folder)


		                image = imageio.imread(os.path.join(image_root, file))
		                label = imageio.imread(os.path.join(label_root, file))

		                image = image[...,0][...,None]/255

		                label = label[...,0]>1
		                label = label[...,None]
		                label = label.astype(np.int64)

		                images_list.append(image)
		                labels_list.append(label)
		                examples = examples + 1
		            except Exception as e:
		                print(e)
		    print("finished loading images")
		    self.examples = examples
		    print("Number of examples found: ", examples)
		    self.images = np.array(images_list)
		    self.labels = np.array(labels_list)

    train_data = GetData(TRAIN_DATA_DIR)
    test_data = GetData(TEST_DATA_DIR)
    print(train_data.images.shape)
    print(train_data.labels.shape)


    np.random.seed(10)
    index = np.random.permutation(526)
    #print(index)
    ### 526 -> 450 train, 76 val

    x_train = train_data.images[index[0:450],:,:,:]
    y_train = train_data.labels[index[0:450],:,:,:]
    print(x_train.shape)
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)


    x_val = train_data.images[index[450:],:,:,:]
    y_val = train_data.labels[index[450:],:,:,:]
    x_val = np.array(x_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)


    x_test = test_data.images
    y_test = test_data.labels
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)


    bb = "vgg16"
    input_shape = (256,256,3)
    c = 1
    enc_weights = "imagenet"
    base_model = sm.Unet(backbone_name = bb, input_shape = input_shape, classes = c, activation='sigmoid', encoder_weights = enc_weights)

    # define number of channels
    N = x_train.shape[-1] # = 1
    inp = Input(shape=(None, None, N))
    l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
    out = base_model(l1)

    modelUnet = Model(inp, out, name=base_model.name)
    modelUnet.compile('Adam', loss=dice_loss, metrics=[iou_score])

    print("loading model" )
    modelUnet.load_weights("modelUnet_sunnybrook_100epochs.keras")
    print("model loaded")

    def plot_three(n=0):
        y_pred_train = modelUnet.predict(x_train[n:(n+1),:,:,:])
        y_pred_val = modelUnet.predict(x_val[n:(n+1),:,:,:])
        y_pred_test = modelUnet.predict(x_test[n:(n+1),:,:,:])

        plt.figure()
        plt.subplot(3,3,1)
        tit = "Train Image num " + str(n)
        plt.title(tit)
        plt.imshow(x_train[n,:,:,0], 'gray', interpolation='none')
        plt.subplot(3,3,2)
        plt.title("True Mask")
        plt.imshow(x_train[n,:,:,0], 'gray', interpolation='none')
        plt.imshow(y_train[n,:,:,0], 'jet', interpolation='none', alpha=0.7)
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
        plt.imshow(y_val[n,:,:,0], 'jet', interpolation='none', alpha=0.7)
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
        plt.imshow(y_test[n,:,:,0], 'jet', interpolation='none', alpha=0.7)
        plt.subplot(3,3,9)
        plt.title("Pred Mask")
        plt.imshow(x_test[n,:,:,0], 'gray', interpolation='none')
        plt.imshow(y_pred_test[0,:,:,0], 'jet', interpolation='none', alpha=0.7)
        #plt.show()
        plot_name = "plot/plot_" + str(n) + ".png"
        plt.savefig(plot_name)

    k = 76 #number from 0 to 75
    for i in range(k):
        print("plotting ", i+1, " out of ", k )
        plot_three(i)

if __name__ == '__main__':
    main()
