# TITLE

Forked from [Self-supervision with Superpixels: Training Few-shot Medical Image Segmentation without Annotation](https://arxiv.org/abs/2007.09886v2)

### 1. Dependencies

Please install essential dependencies (see `requirements.txt`) 

```
dcm2niix
json5
jupyter
nibabel
numpy
opencv-python
Pillow
sacred
scikit-image
SimpleITK
torch
torchvision
```

### 2. Data pre-processing 

**Abdominal MRI**

0. Download [Combined Healthy Abdominal Organ Segmentation dataset](https://chaos.grand-challenge.org/) and put the `/MR` folder under `./data/CHAOST2/` directory

1. Converting downloaded data (T2 fold) to `nii` files in 3D for the ease of reading

run `./data/CHAOST2/dcm_img_to_nii.sh` to convert dicom images to nifti files.

run `./data/CHAOST2/png_gth_to_nii.ipynp` to convert ground truth with `png` format to nifti.

2. Pre-processing downloaded images

run `./data/CHAOST2/image_normalize.ipynb`

**Abdominal CT**

0. Download [Synapse Multi-atlas Abdominal Segmentation dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) and put the `/img` and `/label` folders under `./data/SABS/` directory

1. Intensity windowing 

run `./data/SABS/intensity_normalization.ipynb` to apply abdominal window.

2. Crop irrelavent emptry background and resample images

run `./data/SABS/resampling_and_roi.ipynb` 

**Shared steps**

3. Build class-slice indexing for setting up experiments

run `./data/<CHAOST2/SABS>class_slice_index_gen.ipynb`

`
You are highly welcomed to use this pre-processing pipeline in your own work for evaluating few-shot medical image segmentation in future. Please consider citing our paper (as well as the original sources of data) if you find this pipeline useful. Thanks! 
`

<!--- ### 3. Pseudolabel generation

run `./data_preprocessing/pseudolabel_gen.ipynb`. You might need to specify which dataset to use within the notebook.

### 4. Running training and evaluation

run `./examples/train_ssl_abdominal_<mri/ct>.sh` and `./examples/test_ssl_abdominal_<mri/ct>.sh`

### Acknowledgement

This code is based on vanilla [PANet](https://github.com/kaixin96/PANet) (ICCV'19) by [Kaixin Wang](https://github.com/kaixin96) et al. The data augmentation tools are from Dr. [Jo Schlemper](https://github.com/js3611). Should you have any further questions, please let us know. Thanks again for your interest.

-->

Attention Guided Reconstruction + Transformation Prediciton as a regularizer
