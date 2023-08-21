# Project in Denoising The Dermatology Images: CNN-based Stacked Denoising Autoencoder in Dermatology Domain

This is the repository for my Bachelor's Thesis at the Technical University of Denmark. <br />

The repository consists of Two files. In the Experiment file the related modules used to train the model and test the model and analysis of the results is provided. The DataAnalysis file consists of the script of data cleaning and data analysis.<br />

![fig1Report (3)](https://user-images.githubusercontent.com/59656248/182412954-139de1fd-f1a6-4193-9a77-e0e3665de308.jpg)


## Abstract
Clinical diagnosis of skin diseases is challenging, since the processes are prone to misdiagnosis and inaccuracies due to doctors’ subjectivity\cite{abuzaghleh2015}. As the result tele-dermatology becomes a very popular alternative, especially after the COVID-19 pandemic. However, the patient-recorded images are often taken in poor natural lighting, causing potentially quality problems which presents challenges to the Artificial Intelligence (AI)-based or Dermatologist's diagnosis. \cite{Amouroux2017} A solution is an offline light improvement model which could be adapted to adjust the light condition while preserving the main features of the skin disease.<br />

**Method:** This project is a proof of concept, aiming to explore and evaluate the capability of an CNN-based denoising autoencoder model in adjusting the brightness of dermatology images. The bad light condition is defined as too bright, too dark or existence of camera flash. I evaluate the model trained and tested on a subset of Dermoscopy dataset The International Skin Imaging Collaboration (ISIC) and a subset of the Omhu company's tele-dermatology cell-phone dataset "Imagine". A linear interpolation method is used to simulate bad light condition on these images.<br />

**Results:** The quantitative results suggest that the model is capable at removing the added noise in both datasets, if the added noise is large enough. This is indicated by an increase in both SSIM and PSNR value across different noise levels. The greatest improvement belongs to the darkened images with high level of noise. In the ISIC dataset, the model improves the quality by 0.62 for SSIM score and 16.95 for PSNR score. These values are 0.42 and 9.45 for the Imagine dataset, respectively. The visual results on both of the datasets highlights the capability of the model in removing the noise, and recovering the lesion features. The model trained on ISIC dataset is even capable of removing some artifacts in the dataset. The model trained on Imagine dataset has good generalisation ability across different skin tones and the images with naturally bad light conditions. One shortcoming of the model on tele-dermatology dataset is the lack of model's ability in adjusting the saturation and contrast in some images. The results show that this architecture is suitable for adjusting the light of the dermatology images with bad light conditions in the use case of Omhu company. With more training and fine-tuning the model limitations could be improved and eventually the model could be implemented to the current pipeline of the Omhu mobile app.<br />

## Data
The project uses two datasets, Imagine Dataset and a subset of The International Skin Imaging Collaboratio (ISIC) Corpus. The Imagine Dataset is a private dataset containing cell-phone images. The link to ISIC dataset is provided below.  

https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic

This project uses a subset of Psoriasis class from Imagine Dataset and the following classes from ISIC dataset:<br />
* basal cell carcinoma<br />
* melanoma<br />
* Pigmented Benign Keratosis<br />
* Nevus<br />

## Training
In order to train the model in the respective dataset, please use the following command for ISIC experiment,

```python
python TrainingModuleISIC.py 
```
And the following command for the Imagine experiment.
```python
python TrainingModuleImagine.py 
```

This project does not have any hyper-parameter optimisation, and the hyper parameters are chosen based on the literature and pilot experiments. The training hyper-parameters can be changed in the respective modules in each of the training files TrainingModuleImagine.py TrainingModuleISIC.py.<br />

