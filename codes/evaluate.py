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


def predict_class(model,img_path,device,data_transforms):
    
	# Read the image do data transform and pass it to model and find the class with maximum value
	img = Image.open(img_path)
	transformed_img = torch.unsqueeze(data_transforms(img),0)
	transformed_img	= transformed_img.to(device)
	output_prob = model(transformed_img)
	output = torch.argmax(output_prob).item()

	return output

if __name__ == "__main__":

	# Device initilalization
	device = torch.device("cuda:0")

	# Normalization 
	data_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

	# No of classes
	no_classes = 3

	# Model initiliazation 
	model_ft = models.resnet18(pretrained=False)
	num_ftrs = model_ft.fc.in_features
	model_ft.fc = nn.Linear(num_ftrs,no_classes)
	model_ft.to(device)
	
	# Loading the pretrained weight and setting it to eval mode
	trained_weight_path = '/media/htic/NewVolume1/murali/GE_project/status/weights/1.pt'
	model_ft.load_state_dict(torch.load(trained_weight_path))
	model_ft.eval()

	# Evaluation when the images of each class are placed in their corresponding folder
	val_path = '/media/htic/NewVolume1/murali/GE_project/status/after_process/val/'
	folders  = os.listdir(val_path) #['close','open','unknown']
	img_ext  = 'png'

	# Storing the predicted and groundTruth
	predicted = []
	groundTruth = []
	
	# Iterating over the images 
	for ind,folder in enumerate(folders):
    		
		img_path = os.path.join(val_path,folder,'*.'+img_ext)

		for each in tqdm(glob.glob(img_path)):
		    
			output, = predict_class(model_ft,each,device,data_transforms)
			predicted.append(output)
			groundTruth.append(ind)
	
	# Calculating the classification metrics
	predicted = np.array(predicted)
	groundTruth = np.array(groundTruth)
	print ("Classes:")
	print (','.join(folders))
	conf_matrix  = confusion_matrix(groundTruth,predicted)#,labels =folders)
	class_report = classification_report(groundTruth,predicted) 
	acc_score    = accuracy_score(groundTruth,predicted)
	auc_score    = roc_auc_score(groundTruth,)


	print ("Confusion-matrix:\n",conf_matrix)
	print ("Classification report:\n",class_report)
	print ("Accuracy:",acc_score)

