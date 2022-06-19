#Author: Yeganeh Ghamary
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

import pickle # extra
from datetime import datetime
import torchvision.models as models
from kornia.losses import SSIMLoss, PSNRLoss
from kornia.metrics import psnr, ssim

import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
torch.set_printoptions(linewidth=120)

from torchsummary import summary
import pickle
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)

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
                 label_col, 
                 augment = None, 
                 noisy_transform= None, 
                 image_size = (224,224)
                ):

        self.data_dir = data_dir
        self.label_map = label_map #loaded it in the loader once
        self.augment = augment
        self.noisy_transform = noisy_transform
        self.class_map = class_map
        self.label_col = label_col
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
        
        label = self.class_map[self.label_map.loc[index, "class_label"]]
  
        if self.transform: 
            image = self.transform(img)
   
        if self.augment: 
            augment_image = self.augment(image)
            if self.noisy_transform: 
                noisy_image = self.noisy_transform(augment_image)
                return augment_image, noisy_image, label

        else: #test
            if self.noisy_transform: 
                noisy_image = self.noisy_transform(image)
                return image, noisy_image, label
            
                
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__() 
        # Encoder
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=1) 
    
        # Decoder
        self.deconv4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128*2, 64, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64*2, 32, 4, stride=2, padding=1)    
        self.deconv1 = nn.ConvTranspose2d(32*2, 3, 4, stride=2, padding=1)

        self.dropout = nn.Dropout(p=0.50)
        self.act_fn = nn.LeakyReLU(negative_slope=0.2)
        self.out_fn = nn.Sigmoid()
        
    def forward(self, x):        # x: 3x240x240
        #AE architecture
        # Encoder
        z1 = self.conv1(x)       
        z1 = self.act_fn(z1)
        z1 = self.dropout(z1)  
        z2 = self.conv2(z1)        
        z2 = self.act_fn(z2)
        z2 = self.dropout(z2)
        z3 = self.conv3(z2)      
        z3 = self.act_fn(z3)
        z3 = self.dropout(z3)

        z4 = self.conv4(z3)       
        z = self.act_fn(z4)
        # Decoder
        x_hat = self.deconv4(z)
        x_hat = self.act_fn(x_hat)
        x_hat = torch.cat((x_hat,z3),1)  
        x_hat = self.deconv3(x_hat)
        x_hat = self.act_fn(x_hat)
        
        x_hat = torch.cat((x_hat,z2),1)     
        x_hat = self.deconv2(x_hat)
        x_hat = self.act_fn(x_hat)
        x_hat = torch.cat((x_hat,z1),1)     
        x_hat = self.deconv1(x_hat)
        x_hat = self.out_fn(x_hat)
        return {'z': z, 'x_hat': x_hat}

"___________evaluate______________"
def evaluate(model= None, data_loader_val=None, stage = 'training', gamma=None, alpha=None, classifier=True, clf_loss_function= None, l1_loss_function= None, ssim_loss_function= None, psnr_loss_function= None, tb= None):
    model.eval() # so dropout is turned off,
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
            z = outputs['z'].cuda()
            
            # compute loss
            loss_ae1 = ssim_loss_function(x_hat, x)
            #loss_ae2 = l1_loss_function(x_hat, x) #MAE LOSS
            loss_ae = alpha * loss_ae1 + (1 - alpha) * loss_ae2
            loss = loss_ae
            #compute psnr
            psnrloss = psnr_loss_function(x_hat, x)
            psnr_score = -1* psnrloss.item()
            #save in tmp 
            running_loss +=loss.item()
            running_loss_r += loss_ae.item()
            running_loss_ssim += loss_ae1.item()
            running_loss_l1 += loss_ae2.item()
            running_psnr += psnr_score
                
            grid4 = make_grid(x)
            grid5 = make_grid(noisy_x)
            grid6 = make_grid(x_hat)
            tb.add_image("Test/images", grid4)
            tb.add_image("Test/NoisyImages", grid5)
            tb.add_image("Test/ReconstructedImages", grid6)
          
            
            if stage == 'validation': 
                x = torch.reshape(x,(x.size()[0],-1)).detach().cpu().numpy()
               
                if not len(xs):
                    xs = x
                else:
                    xs = np.vstack([xs,x])
                #save noisy_x
                noisy_x = torch.reshape(noisy_x,(noisy_x.size()[0],-1)).detach().cpu().numpy()
                
                if not len(noisy_xs):
                    noisy_xs = noisy_x
                else:
                    noisy_xs = np.vstack([noisy_xs,noisy_x])
                #save xhat
                x_hat = torch.reshape(x_hat,(x_hat.size()[0],-1)).detach().cpu().numpy()
                
                if not len(x_hats):
                    x_hats = x_hat
                else:
                    x_hats = np.vstack([x_hats,x_hat])
                #save labels and yhat
                #save y
                labels += y.to('cpu').numpy().tolist()
                # make losses in test data
                
                #save embeddings
                z = torch.reshape(z,(z.size()[0],-1)).detach().cpu().numpy()
                
                if not len(feature_maps):
                    feature_maps = z
                else:
                    feature_maps = np.vstack([feature_maps,z])

            if count_val % 33 == 0:
                print('[{}/{}] loss: {:.8} psnr: {:.8}'.format(count_val, len(data_loader_val), loss.item(), psnr_score))  
            count_val += 1
                
    # valid learning curve plots
    model.train() 
    loss_eval_val = running_loss/len(data_loader_val)
    r_loss_eval_val = running_loss_r/len(data_loader_val)

    l1_loss_eval_val = running_loss_l1/len(data_loader_val)
    ssim_loss_eval_val = running_loss_ssim/len(data_loader_val)
    # psnr learning curve
    psnr_eval_val = running_psnr/len(data_loader_val)
    
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
    tb = SummaryWriter(f'runs/BENCHMARK/{now}_model_EPOCHS={num_epochs}_alpha={alpha}_experiment={experiment_name}')
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
            z = outputs['z'].cuda()
            
           
            # Back propagation
            loss_ae1 = ssim_loss_function(x_hat, x)
            loss_ae2 = l1_loss_function(x_hat, x) #MAE LOSS
            loss_ae = alpha * loss_ae1 + (1 - alpha) * loss_ae2
            loss = loss_ae
            #compute psnr
            psnrloss = psnr_loss_function(x_hat, x)
            psnr_score = -1* psnrloss.item()
           
            loss.backward() 
            
            # Update parameters
            optimizer.step()
            # visualise
            if count == len(train_loader)-1:
                print("SSIM", loss_ae1)
                print("L1", loss_ae2)
                print("loss_ae", loss_ae)
                print("loss", loss)
                print("psnr_score", psnr_score)
            #save in tmp 
            running_loss +=loss.item()
            running_loss_r += loss_ae.item()
            running_loss_ssim += loss_ae1.item()
            running_loss_l1 += loss_ae2.item()
            running_psnr += psnr_score
         
            # Show progress
            if count % 100 == 0:
                print('[{}/{}, {}/{}] loss: {:.8} psnr: {:.8}'.format(epoch, num_epochs, count, len(train_loader), loss.item(), psnr_score))  
            count += 1
                
        # Training plots
        loss_eval_tr = running_loss/len(train_loader)
        r_loss_eval_tr = running_loss_r/len(train_loader)
        
        l1_loss_eval_tr = running_loss_l1/len(train_loader)
        ssim_loss_eval_tr = running_loss_ssim/len(train_loader)
        psnr_eval_tr = running_psnr/len(train_loader)
        
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
    return history

"_______________set experiment__________________"
def set_experiment(params):
    #returns all the required input for train and eval modules
    # This function initializes all the modules needed to train the model given a set of parameters.
    # Transform
    composed = {
        'augment': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=(10), scale=(0.8,1.2)),
            transforms.ToTensor()
            ]),
        
        'noisy': transforms.Compose([
            transforms.ToPILImage(),            
            transforms.RandomApply([transforms.ColorJitter(brightness=params["brightness"])],p=params["p"]),
            transforms.ToTensor()
            ])
    }
    
    data_dir = "../imagine/images"
    train_path = "../imagine/train.pickle"
    test_path = '../imagine/test.pickle'
    valid_path = '../imagine/valid.pickle'
    class_map = {"psoriasis":0}
    label_col = "image_id"

    with open(train_path, 'rb') as f:
        x = pickle.load(f)

    train_file = x.loc[x.class_label.isin(["psoriasis"]), :]
    train_file.reset_index(inplace=True)
    train_file = train_file.loc[:, ~train_file.columns.str.contains('^index')]
  

    with open(test_path, 'rb') as f:
        x = pickle.load(f)
  
    test_file = x.loc[x.class_label.isin(["psoriasis"]), :]
    test_file.reset_index(inplace=True)
    test_file = test_file.loc[:, ~test_file.columns.str.contains('^index')]
    

    with open(valid_path, 'rb') as f:
        x = pickle.load(f)
   
    valid_file = x.loc[x.class_label.isin(["psoriasis"]), :]
    valid_file.reset_index(inplace=True)
    valid_file = valid_file.loc[:, ~valid_file.columns.str.contains('^index')]
    

    # load the dataset
    train_dataset = costumeDataset(data_dir = data_dir, 
             label_map = train_file, 
             class_map = class_map, 
             label_col = label_col, 
             augment = composed["augment"], 
             noisy_transform= composed["noisy"], 
             image_size = params['image_size'])
    
    test_dataset = costumeDataset(data_dir = data_dir, 
             label_map = test_file, 
             class_map = class_map, 
             label_col = label_col, 
             augment = None, 
             noisy_transform= composed["noisy"], 
             image_size = params['image_size'])
    valid_dataset = costumeDataset(data_dir = data_dir, 
             label_map = valid_file, 
             class_map = class_map, 
             label_col = label_col, 
             augment = None, 
             noisy_transform= composed["noisy"], 
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
    
    # Get model in gpu
    model = Autoencoder()
    model = model.to(device)

    #get optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
   
    criterion = {"classifier": torch.nn.CrossEntropyLoss(), "l1loss": nn.L1Loss(), "ssimloss": SSIMLoss(5), "psnrloss": PSNRLoss(1)}
    #eval
    
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
    "gamma":[0.1,0.2,0.3], #weight of classifier 
    "p":[0.1,0.3, 0.5], #p of noise
    #"sigma":[1,5, 7.5], #std of gaussion
    "classifier":[True, False], # in This project always false
    "brightness": [0.4,0.2]
}

shared_params = {
 'num_epochs': 180,
 'batch_size': 30,
 'random_seed': 1996,
 'optimizer_name': 'adamW',
 'learning_rate': 0.0001, 
 'image_size': (224,224),
 'degrees': (-10, 10),
 'translate': (0.0, 0.5),
 'scale': (0.5, 0.95),
 'dropout_p': 0.5,# dropout probability
 'negative_slope': 0.2, # negative slope for LeakyRelu
 'n_classes': 1,
 'alpha':1, #weight of ssim
 'weight_decay':1e-3 # L2 regularization
}
def run_experiment(gamma, p , sigma, brightness, classifier):
    #must change the experiment name classifier / benchmark manually
   
    params = { 
        'experiment_name': f'MAIN_BENCHMARK_LRSLOW_DROPLARGE_NOISE_EPOCHNUMBERLARGE_GAMMA{gamma}_P{p}_BRIGHTNESS{brightness}',
        'classifier': classifier, #one value
        "gamma": gamma,
        "p": p,
        "sigma":sigma, #not used in this project
        "brightness": brightness
    }

    params = {**shared_params, **params} 
    experiment_modules = set_experiment(params)
    history = train(experiment_modules['model'],
                    experiment_modules['train_loader'],
                    experiment_modules['test_loader'], 
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
 
'''========================================================= MAIN ============================================================='''
history = run_experiment(0.05, 0.6 , 1, 0.8, False) 
