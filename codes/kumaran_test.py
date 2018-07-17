# Importing torch, numpy, image and machine learning packages
import torch
import torch.nn as nn
import torchvision 
from torchvision import datasets,models,transforms
import numpy as np
from PIL import Image 
import glob
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score,roc_curve
import argparse
import pandas as pd
import logging
import cv2

def predict_class(model,img_path,data_transforms):
	# Read the image do data transform and pass it to model and find the class with maximum value
	# img = Image.open(img_path)
	cv_img = cv2.imread(img_path)
	cv2_im = cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
	img = Image.fromarray(cv2_im)
	transformed_img = torch.unsqueeze(data_transforms(img),0)
	print (transformed_img.shape)
	output_prob = model(transformed_img)
	max_prob = np.exp(torch.max(output_prob).item())
	output = torch.argmax(output_prob).item()

	return output,max_prob

if __name__ == "__main__":

    trained_weight_path  = '/media/htic/NewVolume3/Balamurali/classes/models/final.pt'
    image_path = '/media/htic/NewVolume3/Balamurali/classes/validate/up/right_up120.png'

	# Normalization 
    data_transforms = transforms.Compose([transforms.Resize(256),
										  transforms.CenterCrop(224),
										  transforms. transforms.ToTensor(),
										  transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

	# No of classes
    no_classes = 3

	# Model initiliazation 
    model_ft = models.resnet101(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs,no_classes),nn.LogSoftmax())
	
	# Loading the pretrained weight and setting it to eval mode
	
    model_ft.load_state_dict(torch.load(trained_weight_path))
    model_ft.eval()

	
    output,confidence = predict_class(model_ft,image_path,data_transforms)
    print(output,confidence)
