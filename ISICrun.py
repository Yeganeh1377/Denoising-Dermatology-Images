#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import pandas as pd
import numpy as np
import os
import time
import re 
import random
import glob
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.io import read_image
from pathlib import Path
from PIL import Image
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as tdata
import torch
import pandas as pd
import numpy as np
import os
import re 
import random
import glob
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss as mse

import pickle # extra
from datetime import datetime
import torchvision.models as models
from kornia.losses import SSIMLoss, PSNRLoss
from kornia.metrics import psnr, ssim

import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
torch.set_printoptions(linewidth=120)

from torchsummary import summary

#Todo: make sure that you dont need the rest of the files to run in remote
#from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

#pca TSNE visualisations
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

import pickle
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#torch.manual_seed(123)
#torch.cuda.manual_seed(123)
#np.random.seed(123)
#random.seed(123)

'''
def prepare_labels(image_path, label_file):

    input: one image path, file of labels: csv
    output: the label of the image, torch.long

    x = re.search(r"ISIC_\d+", image_path) #It is syntax of the regular expression in python
    img_file_name = x.group()

    df_labels = label_file.loc[label_file.image == img_file_name, :].drop("image", axis=1)
    target_index = list(df_labels.values[0]).index(1)
    target_index =  torch.tensor(target_index).type(torch.long)
    return target_index
'''
#Todo: recheck the label after fixing the splits
class costumeDataset(Dataset):
    """
    data_path: path of all the images as a list, each element in the list is path of an image
    label_map: dataframe of corresponding labels that is loaded: maps label to image-id
    class_map: a dictionary of classes: maps labels to encoded labels
    patient_map: a dataframe that maps patient_id to frequency of each label among the images of that patient
    label_col: the column in label file where the labels are saved: here is diagnosis
    transform: the type of transform. As default it is none, meaning that the images will only be transformed to tensors
    it can also have augmentations
    noisy_transform: the type of noise implemented in the image. default: none, no noisy data
    val_split: val_train ratio. default: None, no validation split.
    
    Return Dataset class representing our data set
    """
    def __init__(self, 
                 data_dir, 
                 label_map, 
                 class_map, 
                 #patient_map,
                 label_col, 
                 augment = None, 
                 noisy_transform= None, 
                 #test = False, extra
                 #test_NEP = False, not included in the pipeline
                 image_size = (240,240)
                ):
        
        
        #def __init__(self, data_path, label_file, transform = None, noisy_transform= None, test = False):
      
        
        
        #self.img_dir = Path(img_dir) #Todo: check if this is correct and what PAth does and what is data-dir 
        self.data_dir = data_dir
        #self.label_map = pd.read_csv(label_map)
        self.label_map = label_map #loaded it in the loader once
        self.augment = augment
        self.noisy_transform = noisy_transform
        #self.test = test extra = augment false
        self.class_map = class_map
        self.label_col = label_col
        #self.patient_map = patient_map
        #self.test_NEP = test_NEP
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        
    def __len__(self): 
        '''returns size of the dataset'''
        return len(self.label_map)

    def __getitem__(self, index): 
        'Generates one sample of data'  
        img_path = self.data_dir+"/"+self.label_map.loc[index, self.label_col]
        img = Image.open(img_path)
        #print(img.format, img.size, img.mode)
        label = self.class_map[self.label_map.loc[index, "label"]]
  
        if self.transform: 
            image = self.transform(img)
            #print("imageId: ", self.label_map.loc[index, self.label_col], "index: ", index, "encoded label: ", label, "true label: ", self.label_map.loc[index, "majority"], "image type: ", type(image))
            #print("==========================================================================================")
        
        if self.augment: #train valid
            augment_image = self.augment(image)
            if self.noisy_transform: #here it is a mix of all five transforms
                #noisy_image = self.noisy_transform(image)
                noisy_image = self.noisy_transform(augment_image)
                return augment_image, noisy_image, label

        else: #test
            if self.noisy_transform: #here it is a mix of all five transforms
                #noisy_image = self.noisy_transform(image)
                noisy_image = self.noisy_transform(image)
                return image, noisy_image, label
            
                
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__() #Todo: check what this means cause it is not the inheretence
        #define convolusions
        # Encoder
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=1) #
    
        # Decoder
        #self.deconv4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, output_padding=1)#
        self.deconv4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128*2, 64, 4, stride=2, padding=1)#
        #self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64*2, 32, 4, stride=2, padding=1)     # we need the channels to be doubled in input from the previous output because of skip connections
        self.deconv1 = nn.ConvTranspose2d(32*2, 3, 4, stride=2, padding=1)

        self.dropout = nn.Dropout(p=0.15)
        self.act_fn = nn.LeakyReLU(negative_slope=0.2)
        # self.act_fn = nn.LeakyReLU()
        self.out_fn = nn.Sigmoid()
        #self.batchnorm = nn.BatchNorm1d(15*15*256)
        # Classifier network
        self.classifier = nn.Sequential(
        nn.Flatten(),
        # standardize by adding BN as first layer for the classifier
        #nn.BatchNorm1d(37*52*256),#
        #nn.BatchNorm1d(15*15*256),
        #nn.Linear(37*52*256, 256),#
        nn.Linear(15*15*256, 256),
        #nn.BatchNorm1d(256),#
        nn.BatchNorm1d(256),
        self.act_fn,
        self.dropout,
        #nn.Linear(256, 128),#
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),#
        #nn.BatchNorm1d(64),
        self.act_fn,
        self.dropout,
        #nn.Linear(128, n_classes),# n_classes is 3 bc we also use this model in validation and test in test there are 3 classes
        #nn.Linear(64, n_classes),
        nn.Linear(128, 4), #must be 4 now
        )
        
    def forward(self, x):        # x: 3x240x240
        #AE architecture
        # Encoder
        #print(x.size())
        z1 = self.conv1(x)       
        z1 = self.act_fn(z1)
        z1 = self.dropout(z1)  
        #print(z1.size())
        z2 = self.conv2(z1)        
        z2 = self.act_fn(z2)
        z2 = self.dropout(z2)
        #print(z2.size())
        z3 = self.conv3(z2)  #     
        z3 = self.act_fn(z3)#
        z3 = self.dropout(z3)#
        #z3 = self.conv3(z2) 
        #print(z3.size())
        #z = self.act_fn(z3) # 128x37x52
        #print(z.size())

        z4 = self.conv4(z3)  #      
        z = self.act_fn(z4) #z.shape = 256, 37, 52#
        #z = self.batchnorm(z) #Mauricio's idea instead of in the classifier
        #print(z.size())
        # Classification
        #y_hat = self.classifier(z) #just to save the label for evaluating encoder performance
        # Decoder
        x_hat = self.deconv4(z)# #Todo: figure out why deconv4 returns 31*31 and not 30*30 when paddingoutput=1 in img_dim240*240 and not the other
        #print("deconv4", x_hat.size())
        x_hat = self.act_fn(x_hat)#
        #print("deconv4", x_hat.size())
        x_hat = torch.cat((x_hat,z3),1)  #
        x_hat = self.deconv3(x_hat)#
        x_hat = self.act_fn(x_hat)#
        
        #x_hat = self.deconv3(z)
        #x_hat = self.act_fn(x_hat)
        #print(x_hat.size())
        x_hat = torch.cat((x_hat,z2),1)     # take also z2 and feed it to the middle deconv layer
        x_hat = self.deconv2(x_hat)
        x_hat = self.act_fn(x_hat)
        #print(x_hat.size())
        x_hat = torch.cat((x_hat,z1),1)     # take also z1 and feed it to the last deconv layer
        x_hat = self.deconv1(x_hat)
        x_hat = self.out_fn(x_hat)
        #print(x_hat.size())
        #return {'z': z, 'x_hat': x_hat, 'y_hat': y_hat}
        return {'z': z, 'x_hat': x_hat}

"___________evaluate______________"
def evaluate(model= None, data_loader_val=None, stage = 'training', gamma=None, alpha=None, classifier=True, clf_loss_function= None, l1_loss_function= None, ssim_loss_function= None, psnr_loss_function= None, tb= None):
    model.eval() # so dropout is turned off, and I think so is batch norms
    running_loss = 0.0 #trainingloss per batch
    running_loss_cl = 0.0
    running_loss_r = 0.0
    running_loss_ssim = 0.0
    running_loss_l1 = 0.0
    running_correct = 0.0
    running_psnr = 0.0
    running_bacc = 0.0
    n_samples = 0.0
    feature_maps = np.array([])
    xs = np.array([])
    noisy_xs = np.array([])
    x_hats = np.array([])
    labels = []
    predictions = []
    experiment = {}
    count_val = 0
    with torch.no_grad():
        for x, noisy_x, y in data_loader_val:
            x, noisy_x, y= x.cuda(), noisy_x.cuda(), y.cuda()
            outputs = model(noisy_x)
            x_hat = outputs['x_hat'].cuda()
            #y_hat = outputs['y_hat'].cuda()
            z = outputs['z'].cuda()
            
            # compute loss
            ssimlosstmp = ssim_loss_function(x_hat, x) # BxCxHxW
            loss_ae1 = ssimlosstmp.mean() #1
            ssimloss = ssimlosstmp.mean(-1).mean(-1).mean(-1) #B
            print(loss_ae1, ssimlosstmp.size())
            
            loss_ae2 = l1_loss_function(x_hat, x) #MAE LOSS
            loss_ae = alpha * loss_ae1 + (1 - alpha) * loss_ae2
            if classifier:
                loss_clf = clf_loss_function(y_hat, y) #CRos entropy loss (classification loss): not for benchmark
                loss = loss_ae+ gamma*loss_clf
            else:
                loss = loss_ae
            #compute psnr
            #psnrloss = psnr_loss_function(x_hat, x)
            #psnr_score = -1* psnrloss.item()
            #save in tmp 
            #ssimxxhattmp = ssim(x,x_hat, 5) #same dimension as x and noisy_x BxCxHxW
            #ssimxxhat = ssimxxhattmp.mean(-1).mean(-1).mean(-1) # size = B
            
            #psnrnoisyx = -1 * psnr_loss_fct(x,noisy_x)
            #psnrxxhat = -1 * psnr_loss_fct(x, x_hat)
            #msenoisyxtmp = mse(x,noisy_x, reduction = "none") # BxCxHxW
            msexhatxtmp = mse(x,x_hat, reduction = "none")# BxCxHxW
            #print("msenoisyxtmp", msenoisyxtmp.size())
            print("msexhatxtmp", msexhatxtmp.size())
            #msenoisyx = msenoisyxtmp.mean(-1).mean(-1).mean(-1) #B
            msexhatx = msexhatxtmp.mean(-1).mean(-1).mean(-1) #B
            #print("msenoisyx",msenoisyx.size())
            print("msexhatx",msexhatx.size())
            #psnrnoisyx = 10.0 * torch.log10(1.0 ** 2 / msenoisyx) #B
            psnr = 10.0 * torch.log10(1.0 ** 2 / msexhatx) #B
            
            
            running_loss +=loss.item()
            running_loss_r += loss_ae.item()
            running_loss_ssim += ssimloss.sum(0)
            running_loss_l1 += loss_ae2.item()
            running_psnr += psnr.sum(0)
           
            if classifier:
                running_loss_cl += loss_clf.item()
                
                #_, predicted = torch.max(y_hat, dim=1)
                #running_correct += (predicted == y).sum().item()
                #n_samples += y.size(0)
                #running_bacc += balanced_accuracy_score(y.tolist(), predicted.tolist())
            
            grid4 = make_grid(x)
            grid5 = make_grid(noisy_x)
            grid6 = make_grid(x_hat)
            tb.add_image("Test/images", grid4)
            tb.add_image("Test/NoisyImages", grid5)
            tb.add_image("Test/ReconstructedImages", grid6)
           # Show progress
            '''
            if count % 100 == 0:

                print('[{}/{}] validation loss: {:.8}, ssim loss: {:.8}'.format(count, len(data_loader_val), loss.item()), loss_ae1.item())  
            count += 1'''
            
            if stage == 'validation': #testing
                #move images back to cpu and save them
                #save x
                #tb = SummaryWriter(f'runs/hpomain/{now}_experiment={experiment_name}_Evaluation')
                
                #tb.close()
                x = torch.reshape(x,(x.size()[0],-1)).detach().cpu().numpy()
                #x = torch.reshape(x,(x.size()[0],-1)).detach().numpy()
                if not len(xs):
                    xs = x
                else:
                    xs = np.vstack([xs,x])
                #save noisy_x
                noisy_x = torch.reshape(noisy_x,(noisy_x.size()[0],-1)).detach().cpu().numpy()
                #noisy_x = torch.reshape(noisy_x,(noisy_x.size()[0],-1)).detach().numpy()
                if not len(noisy_xs):
                    noisy_xs = noisy_x
                else:
                    noisy_xs = np.vstack([noisy_xs,noisy_x])
                #save xhat
                x_hat = torch.reshape(x_hat,(x_hat.size()[0],-1)).detach().cpu().numpy()
                #x_hat = torch.reshape(x_hat,(x_hat.size()[0],-1)).detach().numpy()
                if not len(x_hats):
                    x_hats = x_hat
                else:
                    x_hats = np.vstack([x_hats,x_hat])
                #save labels and yhat
                #save y
                labels += y.to('cpu').numpy().tolist()
                # make losses in test data
                
                #make yhat in form of y and save
                #if classifier: 
                    #_, predicted = torch.max(y_hat, dim=1)
                    #predictions += predicted.to('cpu').numpy().tolist()
                #predictions += predicted.numpy().tolist()
                #save embeddings
                z = torch.reshape(z,(z.size()[0],-1)).detach().cpu().numpy()
                #z = torch.reshape(z,(z.size()[0],-1)).detach().numpy()
                if not len(feature_maps):
                    feature_maps = z
                else:
                    feature_maps = np.vstack([feature_maps,z])

            if count_val % 33 == 0:
                print('[{}/{}] loss: {:.8}'.format(count_val, len(data_loader_val), loss.item()))  
            count_val += 1
                
    # valid learning curve plots
    model.train() 
    n_val = 339
    loss_eval_val = running_loss/len(data_loader_val)
    r_loss_eval_val = running_loss_r/len(data_loader_val)

    l1_loss_eval_val = running_loss_l1/len(data_loader_val)
    ssim_loss_eval_val = running_loss_ssim.item()/n_val
    # psnr learning curve
    psnr_eval_val = running_psnr.item()/n_val
    
    if classifier:
        cl_loss_eval_val = running_loss_cl/len(data_loader_val)
        acc_eval_val = running_correct/n_samples
        bacc_eval_val = running_bacc/len(data_loader_val)
        
    if stage=='training':
        if classifier:
            print("valid loss: ", loss_eval_val,"r loss: ", r_loss_eval_val, "l1 loss: ", l1_loss_eval_val,"ssim loss: ", ssim_loss_eval_val, "cl loss: ", cl_loss_eval_val, "acc: ", acc_eval_val, "psnr: ", psnr_eval_val, "bacc: ", bacc_eval_val)
            return (loss_eval_val,r_loss_eval_val,l1_loss_eval_val,ssim_loss_eval_val, cl_loss_eval_val, acc_eval_val, bacc_eval_val, psnr_eval_val)
        else:
            print("valid loss: ", loss_eval_val,"r loss: ", r_loss_eval_val, "l1 loss: ", l1_loss_eval_val,"ssim loss: ", ssim_loss_eval_val, "psnr: ", psnr_eval_val)
            return (loss_eval_val,r_loss_eval_val , l1_loss_eval_val, ssim_loss_eval_val, 0.0, 0.0, 0.0,psnr_eval_val) 
            
    if stage=='validation':
        experiment['ys'] = np.array(labels)
        experiment["xs"] = xs
        experiment["noisy_xs"] = noisy_xs
        experiment["x_hats"] = x_hats
        #if classifier:
            #experiment['y_hats'] = np.array(predictions)
        experiment['feature_maps'] = feature_maps
        
        if classifier:
            return (experiment, loss_eval_val,r_loss_eval_val, l1_loss_eval_val, ssim_loss_eval_val, cl_loss_eval_val, acc_eval_val,bacc_eval_val,psnr_eval_val)
        else:
            return (experiment, loss_eval_val,r_loss_eval_val, l1_loss_eval_val, ssim_loss_eval_val, 0.0, 0.0,0.0, psnr_eval_val) 
           
"___________train_______________"
def train(model=None, train_loader=None,valid_loader=None, optimizer=None, num_epochs=None, experiment_name=None, gamma=None, alpha=None, classifier=True,
         clf_loss_function= None, l1_loss_function= None, ssim_loss_function= None, psnr_loss_function=None):
    # criterion is not inpu instead is gamma and alpha
    #Create folder to save the results
    exp_folder = './' + experiment_name
    if not Path(exp_folder).exists():
        Path(exp_folder).mkdir(parents=True, exist_ok=True)
    # Set you model to Train
    model.train()
    #mine
    count = 0
    count_val = 0
    correct_train = 0
    
    # Early stopping
    the_last_loss = 100
    patience = 5
    trigger_times = 0
    history = {'epoch':[], 'loss_tr':[], "cl_loss_tr":[], "r_loss_tr":[], "l1_loss_tr":[], "ssim_loss_tr":[], 'accuracy_tr':[],  'psnr_score_tr':[],
                    'loss_val':[], 'accuracy_val':[],"balanced_accuracy_val":[],"cl_loss_val":[], "r_loss_val":[], "l1_loss_val":[], "ssim_loss_val":[], 'psnr_score_val':[]}
    now = str(datetime.now()).replace(':', '-').replace(' ', '_')
    tb = SummaryWriter(f'runs/BENCHMARKCORRECTED/{now}_model_EPOCHS={num_epochs}_alpha={alpha}_experiment={experiment_name}')
    for epoch in range(1, num_epochs+1):
        running_loss = 0.0 #trainingloss per batch
        running_loss_cl = 0.0
        running_loss_r = 0.0
        running_loss_ssim = 0.0
        running_loss_l1 = 0.0
        running_correct = 0.0
        running_psnr = 0.0
        n_samples = 0.0

        start_time = time.time()
        count = 0
        for x, noisy_x, y in train_loader:
            x, noisy_x, y = x.cuda(), noisy_x.cuda(), y.cuda()
            # Reset gradients
            optimizer.zero_grad()
            #forward propogation
            outputs = model(noisy_x)
            x_hat = outputs['x_hat'].cuda()
            #y_hat = outputs['y_hat'].cuda()
            z = outputs['z'].cuda()
            
           
            # Back propagation
            ssimlosstmp = ssim_loss_function(x_hat, x) # BxCxHxW
            loss_ae1 = ssimlosstmp.mean() #1 for backprop
            ssimloss = ssimlosstmp.mean(-1).mean(-1).mean(-1) #B for epoch
            print(loss_ae1, ssimlosstmp.size())
            loss_ae2 = l1_loss_function(x_hat, x) #MAE LOSS
            loss_ae = alpha * loss_ae1 + (1 - alpha) * loss_ae2 
            #if classifier:
                #loss_clf = clf_loss_function(y_hat, y) #CRos entropy loss (classification loss): not for benchmark
                #loss = loss_ae+ gamma*loss_clf
            #else:
            loss = loss_ae
            #compute psnr
            msexhatxtmp = mse(x,x_hat, reduction = "none")# BxCxHxW
            #print("msenoisyxtmp", msenoisyxtmp.size())
            print("msexhatxtmp", msexhatxtmp.size())
            #msenoisyx = msenoisyxtmp.mean(-1).mean(-1).mean(-1) #B
            msexhatx = msexhatxtmp.mean(-1).mean(-1).mean(-1) #B
            #print("msenoisyx",msenoisyx.size())
            print("msexhatx",msexhatx.size())
            #psnrnoisyx = 10.0 * torch.log10(1.0 ** 2 / msenoisyx) #B
            psnr = 10.0 * torch.log10(1.0 ** 2 / msexhatx) #B
            
            
            running_loss +=loss.item()
            running_loss_r += loss_ae.item()
            running_loss_ssim += ssimloss.sum(0)
            running_loss_l1 += loss_ae2.item()
            running_psnr += psnr.sum(0)
           
            loss.backward() 
            
            # Update parameters
            optimizer.step()
            # visualise
            if count == len(train_loader)-1:
                print("SSIM", loss_ae1)
                print("L1", loss_ae2)
                print("loss_ae", loss_ae)
                print("loss", loss)
                #print("psnr_score", psnr_score)
            
           
            if classifier:
                running_loss_cl += loss_clf.item()
                _, predicted = torch.max(y_hat, dim=1)
                running_correct += (predicted == y).sum().item()
                n_samples += y.size(0)
            # Show progress
            if count % 100 == 0:
                print('[{}/{}, {}/{}] loss: {:.8}'.format(epoch, num_epochs, count, len(train_loader), loss.item()))  
            count += 1
                
        # Training plots
        n= 1189
        loss_eval_tr = running_loss/len(train_loader)
        r_loss_eval_tr = running_loss_r/len(train_loader)
        
        l1_loss_eval_tr = running_loss_l1/len(train_loader)
        ssim_loss_eval_tr = running_loss_ssim.item()/n
        psnr_eval_tr = running_psnr.item()/n
        if classifier:
            cl_loss_eval_tr = running_loss_cl/len(train_loader)
            acc_eval_tr = running_correct/n_samples
       
        if classifier:
            loss_eval_val, r_loss_eval_val, ssim_loss_eval_val, l1_loss_eval_val, cl_loss_eval_val, acc_eval_val, bacc_eval_val, psnr_eval_val = evaluate(model, valid_loader, stage = 'training', gamma=gamma, alpha=alpha, classifier= True, clf_loss_function=clf_loss_function, l1_loss_function=l1_loss_function, ssim_loss_function=ssim_loss_function, psnr_loss_function=psnr_loss_function, tb=tb)
        else: 
            loss_eval_val, r_loss_eval_val, ssim_loss_eval_val, l1_loss_eval_val, _,_,_, psnr_eval_val = evaluate(model, valid_loader, stage = 'training', gamma=gamma, alpha=alpha, classifier= False, clf_loss_function = clf_loss_function, l1_loss_function = l1_loss_function, ssim_loss_function= ssim_loss_function, psnr_loss_function=psnr_loss_function, tb=tb)

        # log in history
        history['epoch'].append(epoch)
        #training 
        history['loss_tr'].append(loss_eval_tr)
        history['r_loss_tr'].append(r_loss_eval_tr)
        history['ssim_loss_tr'].append(ssim_loss_eval_tr)
        history['l1_loss_tr'].append(l1_loss_eval_tr)
        history["psnr_score_tr"].append(psnr_eval_tr)
        if classifier:
            history['cl_loss_tr'].append(cl_loss_eval_tr)
            history['accuracy_tr'].append(acc_eval_tr)
        
        #validation
        history['loss_val'].append(loss_eval_val)
        history['r_loss_val'].append(r_loss_eval_val)
        history['ssim_loss_val'].append(ssim_loss_eval_val)
        history['l1_loss_val'].append(l1_loss_eval_val)
        history["psnr_score_val"].append(psnr_eval_val)
        if classifier:
            history['cl_loss_val'].append(cl_loss_eval_val)
            history['accuracy_val'].append(acc_eval_val)
            history['balanced_accuracy_val'].append(bacc_eval_val)
            
        # log in Tensorboard  : todo: check it saves correctly
        #valid
        tb.add_scalars("Loss/valid", {"total loss": loss_eval_val}, epoch)
        if classifier:
            tb.add_scalars("Loss/valid", {"cross Entropy loss": cl_loss_eval_val}, epoch)
        tb.add_scalars("Loss/valid", {"reconstruction loss": r_loss_eval_val}, epoch)
        tb.add_scalars("Loss/valid", {"l1 loss": l1_loss_eval_val}, epoch)
        tb.add_scalars("Loss/valid", {"ssim loss": ssim_loss_eval_val}, epoch)
        #train
        tb.add_scalars("Loss/train", {"total loss": loss_eval_tr}, epoch)
        if classifier:
            tb.add_scalars("Loss/train", {"cross Entropy loss": cl_loss_eval_tr}, epoch)
        tb.add_scalars("Loss/train", {"reconstruction loss": r_loss_eval_tr}, epoch)
        tb.add_scalars("Loss/train", {"l1 loss": l1_loss_eval_tr}, epoch)
        tb.add_scalars("Loss/train", {"ssim loss": ssim_loss_eval_tr}, epoch)
        
        #all
        tb.add_scalars("Loss/total", {"total training loss": loss_eval_tr}, epoch)
        tb.add_scalars("Loss/total", {"total valid loss": loss_eval_val}, epoch)
        tb.add_scalars("Evaluation metrics/PSNR", {"valid": psnr_eval_val}, epoch)
        tb.add_scalars("Evaluation metrics/PSNR", {"train": psnr_eval_tr}, epoch)
        if classifier:
            tb.add_scalars("Evaluation metrics/Accuracy", {"valid": acc_eval_val}, epoch)
            tb.add_scalars("Evaluation metrics/Accuracy", {"train": acc_eval_tr}, epoch)
            tb.add_scalars("Evaluation metrics/Balanced Accuracy", {"train": bacc_eval_val}, epoch)
            

        with open(exp_folder + '/' + experiment_name + '.pickle','wb') as handle: 
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        end_time = time.time()
        
        print(f'Epoch = {epoch},loss_tr = {loss_eval_tr:.4f}, loss_val = {loss_eval_val:.4f}, psnr_tr = {psnr_eval_tr},psnr_val = {psnr_eval_val},  time ={(end_time-start_time)/60:.2f} mins')
        # Early stopping
        the_current_loss = r_loss_eval_val
        print('The current loss:', the_current_loss)

        if the_last_loss <= the_current_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!')
                #return model
                torch.save(model, exp_folder + '/' + experiment_name + f'epoch_{epoch}') 
                #plot last batch of the last epoch
                grid1 = make_grid(x)
                grid2 = make_grid(noisy_x)
                grid3 = make_grid(x_hat)
                tb.add_image("train/images", grid1)
                tb.add_image("train/NoisyImages", grid2)
                tb.add_image("train/ReconstructedImages", grid3)
                break

        else:
            print('trigger times: 0')
            trigger_times = 0

        the_last_loss = the_current_loss
        if epoch % 10 == 0:
            grid1 = make_grid(x)
            grid2 = make_grid(noisy_x)
            grid3 = make_grid(x_hat)
            tb.add_image("train/images", grid1)
            tb.add_image("train/NoisyImages", grid2)
            tb.add_image("train/ReconstructedImages", grid3)
            torch.save(model, exp_folder + '/' + experiment_name + f'epoch_{epoch}') 

    grid1 = make_grid(x)
    grid2 = make_grid(noisy_x)
    grid3 = make_grid(x_hat)
    tb.add_image("train/images", grid1)
    tb.add_image("train/NoisyImages", grid2)
    tb.add_image("train/ReconstructedImages", grid3)
    tb.close()
    torch.save(model, exp_folder + '/' + experiment_name + f'epoch_{epoch}') 

        
    # Save the best model #todo
    '''
    if (epoch > 100): #todo: check what this is
        if acc_eval_val > acc_saving: #change to ssim
            torch.save(model, exp_folder + '/' + experiment_name + 'best')
        # Save models every single epochs
        if (epoch % 20) == 0:
            torch.save(model, exp_folder + '/' + experiment_name + f'epoch_{epoch}')  '''          
    return history

"_______________set experiment__________________"
def set_experiment(params):
    #returns all the required input for train and eval modules
    # This function initializes all the modules needed to train the model given a set of parameters.
    # Transform
    composed = {
        'augment': transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=(10), scale=(0.8,1.2)),
            transforms.ToTensor()
            ]),
        
        'noisy': transforms.Compose([
            transforms.ToPILImage(),            
            #transforms.RandomApply([transforms.GaussianBlur((21, 21), params["sigma"])],p=params["p"]), #todo: check how to do the proper HPO
            transforms.RandomApply([transforms.ColorJitter(brightness=params["brightness"])],p=params["p"]),
            transforms.ToTensor()
            ])
    }
    data_dir = "ISIC/data"
    train_path = "ISIC/labels_train.csv"
    test_path = 'ISIC/labels_test.csv'
    valid_path = 'ISIC/labels_valid.csv'
    class_map = {"melanoma":0,"nevus":1, "pigmented benign keratosis":2, "basal cell carcinoma":3}
    label_col = "imageId"
    train_file, valid_file, test_file = pd.read_csv(train_path), pd.read_csv(valid_path), pd.read_csv(test_path)
    train_file = train_file.loc[:, ~train_file.columns.str.contains('Unnamed')]
    valid_file = valid_file.loc[:, ~valid_file.columns.str.contains('Unnamed')]
    test_file = test_file.loc[:, ~test_file.columns.str.contains('Unnamed')]
    print(train_file)
    print(valid_file)
    print(test_file)
    # load the dataset
    train_dataset = costumeDataset(data_dir = data_dir, 
             label_map = train_file, 
             class_map = class_map, 
             label_col = label_col, 
             augment = composed["augment"], 
             noisy_transform= composed["noisy"], 
             #test = False,
             #test_NEP = True,
             image_size = params['image_size'])
    
    test_dataset = costumeDataset(data_dir = data_dir, 
             label_map = test_file, 
             class_map = class_map, 
             label_col = label_col, 
             augment = None, 
             noisy_transform= composed["noisy"], 
             #test = False,
             #test_NEP = True,
             image_size = params['image_size'])
    valid_dataset = costumeDataset(data_dir = data_dir, 
             label_map = valid_file, 
             class_map = class_map, 
             label_col = label_col, 
             augment = None, 
             noisy_transform= composed["noisy"], 
             #test = False,
             #test_NEP = True,
             image_size = params['image_size'])
   
   

    #set data iterators
    train_loader = DataLoader(
        train_dataset, batch_size=params['batch_size'], shuffle=True, 
        num_workers=4, pin_memory=False,
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=params['batch_size'], shuffle=True, 
        num_workers=4, pin_memory=False,
    )
    
    test_loader =  DataLoader(
        test_dataset, batch_size=params['batch_size'], shuffle=True, 
        num_workers=4, pin_memory=False,
    )

    #n_classes = 3, p = params["dropout_p"], negative_slope= params["negative_slope"]
    
    # Get model in gpu
    model = Autoencoder()
    model = model.to(device)

    #get optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    #criterion = torch.nn.CrossEntropyLoss()   
    criterion = {"classifier": torch.nn.CrossEntropyLoss(), "l1loss": nn.L1Loss(), "ssimloss": SSIMLoss(5, reduction="none"), "psnrloss": PSNRLoss(1)}
    #eval
    #metrics = {"psnr": PSNRLoss(1.)}
    # Get experiment modules
    experiment_modules ={
        'model' : model,
        'augment' : composed["augment"],
        'noisy': composed["noisy"],
        'train_loader' : train_loader, 
        'valid_loader' : valid_loader,
        'test_loader': test_loader,
        'optimizer': optimizer,
        'classifier_loss' : criterion["classifier"],
        'l1loss': criterion["l1loss"],
        'ssimloss': criterion["ssimloss"],
        'psnrloss': criterion["psnrloss"]
    }
    return experiment_modules

"_______________ config and parameters __________________"
config = {
    "gamma":[0.1,0.2,0.3], #weight of classifier #todo: find values from research
    "p":[0.1,0.3, 0.5], #p of noise
    "sigma":[1,5, 7.5], #std of gaussion
    "classifier":[True, False],
    "brightness": [0.4,0.2]
}

shared_params = {
 'num_epochs': 80,
 'batch_size': 15,
 'random_seed': 1996,
 'optimizer_name': 'adamW',
 'learning_rate': 0.0001, #Alfia: 0.00001
 'image_size': (240,240),
 'degrees': (-10, 10),
 'translate': (0.0, 0.5),
 'scale': (0.5, 0.95),
 'dropout_p': 0.5,# dropout probability
 'negative_slope': 0.2, # negative slope for LeakyRelu
 'n_classes': 4,
 'alpha':1, #weight of ssim
 'weight_decay':1e-3 # L2 regularization
}
def run_experiment(gamma, p , sigma, brightness, classifier):
    #must change the experiment name classifier / benchmark manually
    

    params = { 
        'experiment_name': f'MAIN_BENCHMARK_CORRECTEDMEAN_NOISE_GAMMA{gamma}_P{p}_BRIGHTNESS{brightness}',
        'classifier': classifier, #one value
        "gamma": gamma,
        "p": p,
        "sigma":sigma,
        "brightness": brightness
    }

    params = {**shared_params, **params} 
    experiment_modules = set_experiment(params)
    history = train(experiment_modules['model'],
                    experiment_modules['train_loader'],
                    experiment_modules['test_loader'], #FOR NOISE EXPERIMENT
                    experiment_modules['optimizer'],
                    num_epochs=params['num_epochs'],
                    experiment_name=params['experiment_name'],
                    gamma=params["gamma"],
                    alpha=params["alpha"],
                    classifier= params["classifier"],
                    clf_loss_function = experiment_modules['classifier_loss'],
                    l1_loss_function = experiment_modules['l1loss'],
                    ssim_loss_function = experiment_modules['ssimloss'],
                    psnr_loss_function= experiment_modules["psnrloss"]
                   )
    
    return history   
 
'''================≈≈===≈=≈==≈=≈≈≈≈==================== Evaluation ============================================================'''

def get_evaluation(experiment_modules, params, experiment_name = '', experiment_name_print = '', valid = True, classifier = True):
    # Returns a experiment dict as in Cnn reporting which contains model evaluation of the model on the validation set. 
    
    #model = experiment_modules['model']
    experiment_folder = './' + experiment_name + '/'
    # We chose tha last saved model.
    model_file = experiment_folder + experiment_name + 'epoch_200'
    print(model_file)
    # loading the weights
    #model.load_state_dict(torch.load(model_file))
    model = torch.load(model_file)
    # Set evaluation to training stage to get experiment dict.
    #valid loader for HPO and test loader for final 
    now = str(datetime.now()).replace(':', '-').replace(' ', '_')
    tb = SummaryWriter(f'runs/hpomain/{now}_experiment={experiment_name}_Evaluation')
    if valid:
        experiment,_,_,_,_,_,_,_ = evaluate(model, experiment_modules['valid_loader'], stage = 'validation', gamma=params["gamma"], alpha=1, classifier= classifier, clf_loss_function = experiment_modules['classifier_loss'], l1_loss_function = experiment_modules['l1loss'],ssim_loss_function = experiment_modules['ssimloss'], psnr_loss_function= experiment_modules["psnrloss"], tb=tb)
        tb.close()
        experiment['experiment_name'] = experiment_name_print
        return experiment
    else: 
        if classifier:
            experiment,loss, r_loss, l1_loss, ssim_loss, cl_loss, acc, psnr = evaluate(model, experiment_modules['test_loader'], stage = 'validation', gamma=params["gamma"], alpha=0.7, classifier= classifier, clf_loss_function = experiment_modules['classifier_loss'], l1_loss_function = experiment_modules['l1loss'],ssim_loss_function = experiment_modules['ssimloss'], psnr_loss_function= experiment_modules["psnrloss"], tb=tb)
            experiment['experiment_name'] = experiment_name_print
            tb.close()
            return experiment,loss, r_loss, l1_loss, ssim_loss, cl_loss, acc, psnr
        else:
            experiment,loss, r_loss, l1_loss, ssim_loss, _ , _ , psnr = evaluate(model, experiment_modules['test_loader'], stage = 'validation', gamma=params["gamma"], alpha=0.7, classifier= classifier, clf_loss_function = experiment_modules['classifier_loss'], l1_loss_function = experiment_modules['l1loss'],ssim_loss_function = experiment_modules['ssimloss'], psnr_loss_function= experiment_modules["psnrloss"], tb=tb)
            experiment['experiment_name'] = experiment_name_print
            tb.close()
            return experiment,loss, r_loss, l1_loss, ssim_loss, psnr
        
        
def get_experiment(gamma, p, sigma, classifier, valid=True):
    params = {
        'experiment_name': f'HPO_MAIN_benchmark_CORRECTED_MEAN_GAMMA{gamma}_P{p}_SIGMA{sigma}',
        'classifier': classifier, #one value
        "gamma": gamma,
        "p": p, 
        "sigma": sigma
        }
    params = {**shared_params, **params}
    experiment_modules = set_experiment(params)    
    
    experiment = get_evaluation(experiment_modules,params, experiment_name = params['experiment_name'],experiment_name_print = params['experiment_name'], valid = valid, classifier=classifier)
   
    return experiment

'''
def visualise_performance(gamma, p, sigma, classifier, valid =True):
    
    experiment = get_experiment(gamma, p, sigma, classifier, valid = True)
   
    if valid:
        grid1 = make_grid(experiment["xs"])
        grid2 = make_grid(experiment["noisy_xs"])
        grid3 = make_grid(experiment["x_hats"])
        tb.add_image("valid/images", grid1)
        tb.add_image("valid/NoisyImages", grid2)
        tb.add_image("valid/ReconstructedImages", grid3)
    
    return'''
    
'''========================================================= MAIN ============================================================='''
#run_experiment(gamma, p , sigma, brightness, classifier):
#Classifier HPO: hpoclassifiertest logs in experiment_name file
#historyp1 = run_experiment(0.05, 0.6 , 1, 0.4, False) 
#historyp2 = run_experiment(0.05, 0.6 , 1, 0.6, False) 
historyp3 = run_experiment(0.05, 0.6 , 1, 0.8, False) 
#historyp4 = run_experiment(0.05, 0.6 , 1, 0.2, False) 


'''historyp2 = run_experiment(config["gamma"][0], config["p"][2] , 1, False) 
historyp1 = run_experiment(config["gamma"][0], config["p"][1] , 1, False) 
historyp0 = run_experiment(config["gamma"][0], config["p"][0] , 1, False) '''
'''
classhistorygamma2 = run_experiment(config["gamma"][2], 0.5 , 5, True) 
classhistorygamma1 = run_experiment(config["gamma"][1], 0.5 , 5, True) 
classhistorygamma0 = run_experiment(config["gamma"][0], 0.5 , 5, True) 
'''
'''
load_model_filename = './HPO_MAIN_BENCHMARK_GAMMA0.1_P0.5_SIGMA5/HPO_MAIN_BENCHMARK_GAMMA0.1_P0.5_SIGMA5epoch_200'
model = torch.load(load_model_filename)
print(model)
params = {
        'experiment_name': f'HPO_MAIN_BENCHMARK_GAMMA{0.1}_P{0.1}_SIGMA{5}',
        'classifier': False, #one value
        "gamma": 0.1,
        "p": 0.1, 
        "sigma": 5
        }
params = {**shared_params, **params}
experiment_modules = set_experiment(params)    
x, noisy_x, y = next(iter(experiment_modules["valid_loader"]))
now = str(datetime.now()).replace(':', '-').replace(' ', '_')
tb = SummaryWriter(f'runs/hpomain/{now}_experiment{params["experiment_name"]}_EvaluationTest')
model.eval()
with torch.no_grad():
    x, noisy_x, y= x.cuda(), noisy_x.cuda(), y.cuda()

    #move output to cuda: check if really need to
    outputs = model(noisy_x) 
    #y_hat = outputs['y_hat'].cuda()
    x_hat = outputs['x_hat'].cuda()
    #z = outputs['z'].cuda()
    grid1 = make_grid(x)
    grid2 = make_grid(noisy_x)
    grid3 = make_grid(x_hat)
    tb.add_image("valid/x", grid1)
    tb.add_image("valid/noisy", grid2)
    tb.add_image("valid/reconstructed", grid3)
tb.close()
'''

'''
benchmark = False
train = True

if benchmark:
    if train:
        benchhistoryp0 = run_experiment(config["gamma"][0], config["p"][0], config["sigma"][0], config["classifier"][1]) 
        benchhistoryp1 = run_experiment(config["gamma"][0], config["p"][1], config["sigma"][0], config["classifier"][1]) 
        benchhistoryp2 = run_experiment(config["gamma"][0], config["p"][2], config["sigma"][0], config["classifier"][1]) 
    else:
        #for i in range(len(config["p"])):
        experiment = get_experiment(config["gamma"][0], config["p"][0], config["sigma"][0], config["classifier"][1], valid=True)


else:
    if train:
        #poptim = congi["p"][0]
        poptim = 0.5
        classhistorygamma0 = run_experiment(config["gamma"][2], poptim, config["sigma"][0], False) 
        #classhistorygamma1 = run_experiment(config["gamma"][1], poptim, config["sigma"][0], config["classifier"][0]) 
        #classhistorygamma2 = run_experiment(config["gamma"][2], poptim, config["sigma"][0], config["classifier"][0]) 
        
'''
"__NOTES___"
# gamma not important in the classifier doesnt matter which value
# p is variable in benchmark and fixed to the best in classifier
#sigma is ALWAYS fixed to 5 

