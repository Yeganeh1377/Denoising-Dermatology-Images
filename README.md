# Project in Denoising The Dermatology Images: CNN-based Stacked Denoising Autoencoder in Dermatology Domain

This is the repository for my Bachelor's Thesis at the Technical University of Denmark. This project is a proof of concept, investigating the CNN-based Stacked Denoising Autoencoder in improving the brightness of dermoscopic and cell-phone images. The brightness is jittered using linear interpolation with black image with varying alpha value chosen uniformly from the range of [0.2,1.8].<br />

The repository consists of Two files. In the Experiment file the related modules used to train the model and test the model and analysis of the results is provided. The DataAnalysis file consists of the script of data cleaning and data analysis.<br />

![fig1Report (3)](https://user-images.githubusercontent.com/59656248/182412954-139de1fd-f1a6-4193-9a77-e0e3665de308.jpg)


# Data

The project uses two datasets, Imagine Dataset and a subset of The International Skin Imaging Collaboratio (ISIC) Corpus. The Imagine Dataset is a private dataset containing cell-phone images. The link to ISIC dataset is provided below.  

https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic

This project uses a subset of Psoriasis class from Imagine Dataset and the following classes from ISIC dataset:<br />
* basal cell carcinoma<br />
* melanoma<br />
* Pigmented Benign Keratosis<br />
* Nevus<br />

# Training

In order to train the model in the respective dataset, please use the following command for ISIC experiment,

```python
python TrainingModuleISIC.py 
```
And the following command for the Imagine experiment.
```python
python TrainingModuleImagine.py 
```

This project does not have any hyper-parameter optimisation, and the hyper parameters are chosen based on the literature and pilot experiments. The training hyper-parameters can be changed in the respective modules in each of the training files TrainingModuleImagine.py TrainingModuleISIC.py.<br />

