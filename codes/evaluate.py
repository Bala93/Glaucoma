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
	output_prob = torch.nn.functional.softmax(output_prob)    
	max_prob = torch.max(output_prob).item()
	output = torch.argmax(output_prob).item()

	return output,max_prob

if __name__ == "__main__":



	device = torch.device("cuda:0")
	trained_weight_path = '/media/htic/NewVolume1/murali/Glaucoma/models/Combined_RimOne_Origa_Normalized/3.pt' 
	val_path = '/media/htic/NewVolume1/murali/Glaucoma/PretrainDataSets/ForValidation/Normalized'
	img_ext  = 'jpg'


	# Normalization 
	data_transforms = transforms.Compose([transforms.Resize(256),
										  transforms.CenterCrop(224),
										  transforms. transforms.ToTensor(),
										  transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


	# No of classes
	no_classes = 2

	# Model initiliazation 
	model_ft = models.resnet101(pretrained=False)
	num_ftrs = model_ft.fc.in_features
	model_ft.fc = nn.Linear(num_ftrs,no_classes)
	model_ft.to(device)
	
	# Loading the pretrained weight and setting it to eval mode
	
	model_ft.load_state_dict(torch.load(trained_weight_path))
	model_ft.eval()

	# Evaluation when the images of each class are placed in their corresponding folder

	folders  = os.listdir(val_path) #['close','open','unknown']
	

	# Storing the predicted and groundTruth
	predicted = []
	groundTruth = []
	scores = []
	
	# Iterating over the images 
	for ind,folder in enumerate(folders):
    		
		img_path = os.path.join(val_path,folder,'*.'+img_ext)

		for each in tqdm(glob.glob(img_path)):
		    
			output,confidence = predict_class(model_ft,each,device,data_transforms)
			predicted.append(output)
			groundTruth.append(ind)
			scores.append(confidence)
	
	# Calculating the classification metrics
	predicted = np.array(predicted)
	groundTruth = np.array(groundTruth)
	scores = np.array(scores)

	ind_ = np.where(predicted == 0)
	scores[ind_] = 1 - scores[ind_]

	print ("Classes:")
	print (','.join(folders))
	conf_matrix  = confusion_matrix(groundTruth,predicted)#,labels =folders)
	class_report = classification_report(groundTruth,predicted) 
	acc_score    = accuracy_score(groundTruth,predicted)
	auc_score    = roc_auc_score(groundTruth,scores)


	print ("Confusion-matrix:\n",conf_matrix)
	print ("Classification report:\n",class_report)
	print ("Accuracy:",acc_score)
	print ("AUC:",auc_score)

