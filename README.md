# DL_cardiac_segmentation
Deep Learning for Cardiac images Segmentation: semi-automatic method to build Left Ventricle mesh
![alt text](https://github.com/danielececcarelli/DL_cardiac_segmentation/blob/main/img/4D_segment.gif)

## Project Outline
- 2D automatic segmentation for Short Axis MRI Endocardium
- 2D automatic segmentation for S-A MRI Myocardium
- 3D Mesh building and shape refining
- 4D segmentation

## Short-Axis MRI Dataset
Two datasets used:
- Sunnybrook dataset ![(source)](https://www.cardiacatlas.org/studies/sunnybrook-cardiac-data/), 100 patients, 2 different time (ED and ES) -> 200 3D patients images -> transformed in 2D slice images. Contour: Inner endocardium of Left Ventricle (and also outer epicardium for ED)
- ACDC Dataset ![(source)](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html), (result in) 95 patient at ED and ES -> 190 3D patients images. Contour: Inner LV, Outer LV and RV

ACDC Dataset provide also 4D images

## 4D Segmentation

