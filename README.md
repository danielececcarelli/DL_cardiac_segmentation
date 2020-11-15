# Deep Learning for cardiac segmentation
***Deep Learning for Cardiac images Segmentation: semi-automatic method to build Left Ventricle mesh***

![alt text](https://github.com/danielececcarelli/DL_cardiac_segmentation/blob/main/img/4D_segment.gif)

## Project Outline
- 2D automatic segmentation for Short Axis MRI Endocardium
- 2D automatic segmentation for S-A MRI Myocardium
- 3D Mesh building and shape refining
- 4D segmentation

**All the detail and result can be found in the presentation:** ![pdf](https://github.com/danielececcarelli/DL_cardiac_segmentation/blob/main/Deep%20Learning%20for%20Cardiac%20Segmentation.pdf)

## Short-Axis MRI Dataset
Two datasets used:
- **Sunnybrook dataset** ![(source)](https://www.cardiacatlas.org/studies/sunnybrook-cardiac-data/), 100 patients, 2 different time (ED and ES) -> 200 3D patients images -> transformed in 2D slice images. Contour: Inner endocardium of Left Ventricle (and also outer epicardium for ED)
- **ACDC Dataset** ![(source)](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html), (result in) 95 patient at ED and ES -> 190 3D patients images. Contour: Inner LV, Outer LV and RV

Below an example from Sunnybrook data. We can see all the patient's slices through the short axis and the relatively mask of the Inner endocardium of LV

<img src="https://github.com/danielececcarelli/DL_cardiac_segmentation/blob/main/img/esempio2_short_axis.png" width="1000" title="Example">


ACDC Dataset provide also 4D images, with the addition of temporal dimension over a whole cardiac cycle

## First Model: segmentation on Sunnybrook and ACDC separately
All the models are built using the python library "Segmentation Models" ![SM](https://github.com/qubvel/segmentation_models)
- **Segmentation of LV Endocardium on sunnybrook dataset using UNet with VGG16** ![python notebook](https://github.com/danielececcarelli/DL_cardiac_segmentation/blob/main/First_model/sunnybrook_data/train_sunnybrook.ipynb)
- **Segmentation of LV Endocardium, LV Epicardium and RV on ACDC dataset using UNet with VGG16** ![python notebook in colab](https://github.com/danielececcarelli/DL_cardiac_segmentation/blob/main/First_model/acdc_data/train_acdc_with_google_colab.ipynb)

UNet model architecure 
<img src="https://github.com/danielececcarelli/DL_cardiac_segmentation/blob/main/img/U-net.png" width="500" title="UNet architecture">

Source: ![UNet](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

## Ring Model: segmentation on Sunnybrook and ACDC together
Segmentation of LV Myocardium on a dataset with both Sunnybrook and ACDC data

Data augmentation using ![Albumentation library](https://github.com/albumentations-team/albumentations) :
- Random-sized crop
- Horizontal and Vertical flip
- Random rotation with an angle in [-90°,90°]
- Blur the image
- Random change of brightness

Example of generated augmented images and relatively augmented masks

<img src="https://github.com/danielececcarelli/DL_cardiac_segmentation/blob/main/img/augm_image.png" width=400/> <img src="https://github.com/danielececcarelli/DL_cardiac_segmentation/blob/main/img/augm_mask.png" width=400/> 

#### Result:
<img src="https://github.com/danielececcarelli/DL_cardiac_segmentation/blob/main/img/result_example.png" width="800" title="Examples of ring segmentation from train/val/test set">

## 3D Mesh building and Shape Refinement
Pipeline for **shape refinement and building volume**:
1) Extract the endo and epi from ring images using OpenCV
2) Extract contours of both endo and epi from the slices
3) (Optionally) Shift correction for the contours
4) Create the volume
5) Add the apex if required

**All the code for points 2 to 5 comes from https://github.com/cbutakoff/tools/tree/master/Python/mri_mesh_from_contours**

Result examples from a test set patient

<img src="https://github.com/danielececcarelli/DL_cardiac_segmentation/blob/main/img/example_shape_refinement.png" width="1000" title="Result examples from a test set patient">

Comparison between true volume and predicted with our Cardiac Deep Learning

<img src="https://github.com/danielececcarelli/DL_cardiac_segmentation/blob/main/img/comparison_pred_true_volume_TEST_patient008_ed.png" width="1000" title="Comparison between true volume and predicted with Deep Learning">

## 4D Segmentation
Finally, we can segment the volume of LV during a **complete cardiac cycle** to see the **cardiac dynamic** of the Ventricle. Below an example from ACDC Dataset:


<img src="https://github.com/danielececcarelli/DL_cardiac_segmentation/blob/main/img/4D_segment.gif" width="1000" title="4D segmentation">

