import os
import numpy as np
import imageio
from matplotlib import pyplot as plt
import tensorflow as tf
import nrrd
import sys
import nibabel as nib
import time

import segmentation_models as sm
from segmentation_models.losses import DiceLoss
from segmentation_models.metrics import IOUScore

from keras.layers import Input, Conv2D, Dropout
from keras.models import Model
from keras.optimizers import Adam

import cv2
import itk

def padding_image_and_standardize(img):
    (dim1,dim2,dim3) = img.shape
    new_img = (img - np.min(img))/(np.max(img)-np.min(img))
    new_image = np.zeros( shape=(256,256,dim3) )
    new_image[0:dim1,0:dim2,0:dim3] = new_img
    return new_image

def str_fun(i):
    if i<10:
        return "0"+str(i)
    else:
        return str(i)

def load_segmentation_model(dataloader):

    bb = "vgg16"
    input_shape = (256,256,3)
    c = 1
    enc_weights = "imagenet"
    base_model = sm.Unet(backbone_name = bb, input_shape = input_shape, classes = c, activation='sigmoid', encoder_weights = enc_weights)

    # define number of channels
    N = 1
    inp = Input(shape=(None, None, N))
    l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
    #now put drop rate = 0!
    drop = Dropout(0.)(l1)
    bm = base_model(drop)
    out = Dropout(0.)(bm)

    lr = 0.0001
    optim = Adam(lr)

    w = np.array([1,1])
    loss = DiceLoss(class_weights=w)
    metrics = IOUScore(class_weights=w)
    modelUnet = Model(inp, out, name=base_model.name)
    modelUnet.compile(optim, loss=loss, metrics=[metrics])

    modelUnet.load_weights("models_result/vgg16_100epochs_batch4/modelUnet_ring_dropout_vgg16_best.h5")

    y_pred = modelUnet.predict(dataloader)
    return y_pred



class Single_Patient_4D:
    def __init__(self, path, patient_name):
        self.patient_name = patient_name

        x_test = []
        y_test = []

        image_4D = nib.load(os.path.join(path, patient_name, patient_name+".nii.gz"))
        image_4D = image_4D.get_fdata()

        #shape of image_4d = (x,y,z,time)

        self.number_of_slices = image_4D.shape[2]
        self.number_of_frame = image_4D.shape[3]
        self.total_n = self.number_of_slices*self.number_of_frame

        for t in range(self.number_of_frame):
            for i in range(self.number_of_slices):
                x = image_4D[:,:,i,t] #(174,208)
                x = np.expand_dims(x, axis=2) #(174,208,1)
                x_test.append(padding_image_and_standardize(x)) #(256,256,1)
                y_test.append(np.zeros(shape=(256,256,1)))

        self.dataset_x = x_test
        self.dataset_y = y_test


    def __getitem__(self, i):
        # get data
        image = self.dataset_x[i]
        mask = self.dataset_y[i]
        return image, mask

    def __len__(self):
        return len(self.dataset_x)

class Dataloder(tf.keras.utils.Sequence):
    """Load data from dataset and form batches
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        # transpose list of lists
        batch = tuple([np.stack(samples, axis=0) for samples in zip(*data)])
        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


patient_name = "patient008_4d"
path = "volume"
patient = Single_Patient_4D(path, patient_name)

print("Select patient : ", patient.patient_name)
print("Total_number : ", patient.total_n)
print("Frames : ", patient.number_of_frame)
print("Slices : ", patient.number_of_slices)

print("load data...")
dataloader_patient = Dataloder(patient, 1, False)


print("load model")
print("prediction...")
y_pred = load_segmentation_model(dataloader_patient)

print("done")

yy_pred = (255.*y_pred).astype(np.uint8)

endo = np.zeros(shape = (patient.number_of_slices,256,256))
epi = np.zeros(shape = (patient.number_of_slices,256,256))


result = os.path.join("volume", patient_name)
#os.makedirs(result)


for tt in range(patient.number_of_frame):
    t = tt
    for i in range(t*patient.number_of_slices, (t+1)*patient.number_of_slices):
        th, im_th = cv2.threshold(yy_pred[i,:,:,0], 127, 255, cv2.THRESH_BINARY)

        # Copy the thresholded image
        im_floodfill = im_th.copy()
        # Mask used to flood filling.
        # NOTE: the size needs to be 2 pixels bigger on each side than the input image
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground
        im_out = im_th | im_floodfill_inv

        if(cv2.countNonZero(im_floodfill_inv) == 0):
            print("problem in slice ", i%patient.number_of_slices, " in frame t = ", tt)
            print("fixing...")

            y = yy_pred[i,:,:,0].copy()
            circles = cv2.HoughCircles(y,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
            cv2.circle(y,(round(circles[0][0][0]),round(circles[0][0][1])),round(circles[0][0][2]),255,2)

            th, im_th = cv2.threshold(y, 127, 255, cv2.THRESH_BINARY)

            # Copy the thresholded image
            im_floodfill = im_th.copy()
            # Mask used to flood filling.
            # NOTE: the size needs to be 2 pixels bigger on each side than the input image
            h, w = im_th.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)

            # Floodfill from point (0, 0)
            cv2.floodFill(im_floodfill, mask, (0,0), 255)

            # Invert floodfilled image
            im_floodfill_inv = cv2.bitwise_not(im_floodfill)

            # Combine the two images to get the foreground
            im_out = im_th | im_floodfill_inv

            kernel = np.ones((5,5),np.uint8)
            opening = cv2.morphologyEx(im_floodfill_inv, cv2.MORPH_OPEN, kernel)
            im_floodfill_inv = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

            opening = cv2.morphologyEx(im_out, cv2.MORPH_OPEN, kernel)
            im_out = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            if(cv2.countNonZero(im_floodfill_inv) == 0):
                print("not solved")
            else:
                print("solved")



        epi[i%patient.number_of_slices,:,:] = im_out
        endo[i%patient.number_of_slices,:,:] = im_floodfill_inv

    epi /= 255.0
    endo /= 255.0

    name_epi = "epi.nrrd"
    nrrd.write(os.path.join(result, name_epi), epi)
    name_endo = "endo.nrrd"
    nrrd.write(os.path.join(result, name_endo), endo)

    ########################################################################################################

    #---------------------------------------#
    #           EXTRACT CONTOUR             #
    #---------------------------------------#
    from volume.contour_extract import Extract_Contour

    ex = Extract_Contour(os.path.join(result,name_endo), os.path.join(result,name_epi),
                                        os.path.join(result, 'points_corrected.npz'))
    ex.extract()

    #---------------------------------------#
    #             MRI CORRECTOR             #
    #---------------------------------------#
    # from volume.mricorrector import MRICORRECTOR
    #
    # corr = MRICORRECTOR(os.path.join(result, 'points.npz'), os.path.join(result, 'points_corrected.npz'))
    #
    # correct_flag = True
    #
    # if correct_flag:
    #     corr.correct()
    # else:
    #     corr.copy()


    #---------------------------------------#
    #             Create Mesh               #
    #---------------------------------------#
    from volume.create_mesh import mesh_creator

    mesh = mesh_creator(os.path.join(result, 'points_corrected.npz'),
                        os.path.join(result, 'endo_mesh.vtk'),
                        os.path.join(result, 'epi_mesh.vtk'),
                        os.path.join(result, 'points_final.npz')
                        )

    mesh.create_mesh()


    #---------------------------------------#
    #             Add Apex                  #
    #---------------------------------------#
    from volume.addapex import GenerateApex


    npzfile = np.load(os.path.join(result, 'points_final.npz'))
    sampling_endo = npzfile['sampling_endo']
    sampling_epi = npzfile['sampling_epi']

    GenerateApex(os.path.join(result, 'endo_mesh.vtk'), os.path.join(result, 'endo_mesh_capped_' + str_fun(t)+'.vtk'), sampling_endo)
    GenerateApex(os.path.join(result, 'epi_mesh.vtk'), os.path.join(result, 'epi_mesh_capped_'+str_fun(t)+'.vtk'), sampling_epi)

    time.sleep(5)
