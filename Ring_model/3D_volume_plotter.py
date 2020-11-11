import os
import numpy as np
import imageio
from matplotlib import pyplot as plt
import tensorflow as tf
import nrrd
import sys

import segmentation_models as sm
from segmentation_models.losses import DiceLoss
from segmentation_models.metrics import IOUScore

from keras.layers import Input, Conv2D, Dropout
from keras.models import Model
from keras.optimizers import Adam

import cv2
import itk

# patients_for_test_sb = ("SC-HF-I-01", "SC-HF-I-40", "SC-HF-NI-03", "SC-HYP-01", "SC-N-02")
#
# patients_for_test_acdc = (
# "patient000","patient008","patient017","patient020","patient033",
# "patient036","patient040","patient056","patient077","patient078",
# "patient091","patient100","patient110","patient124","patient135",
# "patient140","patient151","patient166","patient176","patient187")

def load_segmentation_model(n, dataloader):

    bb = "resnet50"
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

    name = "prova" + str(n) + "_model"
    path = ""

    for root, dirs, files in os.walk(os.path.join(path,name)):
        for file in sorted(files):
            if file.endswith("best.h5")==True:
                modelUnet.load_weights(os.path.join(path,root,file))

    y_pred = modelUnet.predict(dataloader)
    return y_pred



class Single_Patient:
    def __init__(self, path_x, path_y, patient_name):
        self.patient_name = patient_name

        x_test = []
        y_test = []

        for root, dirs, files in os.walk(path_x):
            for file in sorted(files):
                if file.endswith(".png"):
                    if file.startswith(patient_name):
                        x = imageio.imread(os.path.join(path_x, file))
                        x = x/255.
                        y = imageio.imread(os.path.join(path_y, file))
                        y = y/255.
                        x_test.append(x)
                        y_test.append(y)

        for i in range(len(x_test)):
            x_test[i] = np.expand_dims(x_test[i], axis=2)
            y_test[i] = np.expand_dims(y_test[i], axis=2)

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


patient_name = "patient008"
path_x = "acdc_data/result_images"
path_y = "acdc_data/result_labels"

patient = Single_Patient(path_x, path_y, patient_name)

print("Select patient : ", patient.patient_name)
print("Number of slices : ", len(patient))

print("load data...")
dataloader_patient = Dataloder(patient, 1, False)


n = 3
print("load model weight = ",n)
print("prediction...")
y_pred = load_segmentation_model(n, dataloader_patient)

print("done")

yy_pred = (255.*y_pred).astype(np.uint8)

endo = np.zeros(shape = (len(patient),256,256))
epi = np.zeros(shape = (len(patient),256,256))


result = os.path.join("volume", patient_name)
#os.makedirs(result)

for i in range(len(patient)):
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

    epi[i,:,:] = im_out
    endo[i,:,:] = im_floodfill_inv

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
                                    os.path.join(result, 'points.npz'))
ex.extract()

#---------------------------------------#
#             MRI CORRECTOR             #
#---------------------------------------#
from volume.mricorrector import MRICORRECTOR

corr = MRICORRECTOR(os.path.join(result, 'points.npz'), os.path.join(result, 'points_corrected.npz'))

correct_flag = True

if correct_flag:
    corr.correct()
else:
    corr.copy()


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

GenerateApex(os.path.join(result, 'endo_mesh.vtk'), os.path.join(result, 'endo_mesh_capped.vtk'), sampling_endo)
GenerateApex(os.path.join(result, 'epi_mesh.vtk'), os.path.join(result, 'epi_mesh_capped.vtk'), sampling_epi)
