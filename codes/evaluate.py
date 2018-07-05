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

	## Parsing and assigning values

	parser = argparse.ArgumentParser('Evaluating image classification')
	parser.add_argument(
		'--model_path',
		required = True,
		type = str,
		help = 'path of the model file'
	)
	parser.add_argument(
		'--val_path',
		required = True,
		type = str,
		help = 'Path to validation images'
	)

	parser.add_argument(
		'--img_ext',
		required = True,
		type = str,
		help = 'Image extension'
	)

	parser.add_argument(
		'--csv_path',
		required = True,
		type = str,
		help = 'Path to csv'
	)

	parser.add_argument(
		'--cuda_no',
		required = True,
		type = str,
		help = 'Specify the cuda id'
	)


	
	'''
	Example command:
	python evaluate.py --model_path --val_path --img_ext --csv_path --cuda_no
	'''

	opt = parser.parse_args()
	trained_weight_path = opt.model_path
	val_path   = opt.val_path
	img_ext    = opt.img_ext
	csv_path   = opt.csv_path
	cuda_no    = opt.cuda_no
	device = torch.device("cuda:{}".format(cuda_no))	
	log_path = os.path.join(os.path.dirname(csv_path),'metrics.log')
	logging.basicConfig(filename=log_path,level=logging.INFO)

	# print ("#########")
	# print ("Settings:")
	logging.info(vars(opt))
	# print ("#########")
	#############################

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
	img_names = []	

	# Iterating over the images 
	for ind,folder in enumerate(folders):
    		
		img_path = os.path.join(val_path,folder,'*.'+img_ext)

		for each in tqdm(glob.glob(img_path)):
			img_name = os.path.basename(each)
			output,confidence = predict_class(model_ft,each,device,data_transforms)
			predicted.append([output])
			groundTruth.append([ind])
			scores.append([confidence])
			img_names.append([img_name])
	
	# Calculating the classification metrics
	predicted = np.array(predicted)
	groundTruth = np.array(groundTruth)
	scores = np.array(scores)
	img_names = np.array(img_names)
	

	# To calculate AUC, the function needs the confidence score of the positive class.
	ind_ = np.where(predicted == 1)
	scores[ind_] = 1 - scores[ind_]
	scores = np.round(scores,decimals=1)
	result = np.hstack([img_names,scores])
	df = pd.DataFrame(result)
	df.to_csv(csv_path,header=['FileName','Glaucoma Risk'],index=False)


	logging.info("Classes:")
	logging.info(','.join(folders))
	conf_matrix  = confusion_matrix(groundTruth,predicted)#,labels =folders)
	class_report = classification_report(groundTruth,predicted) 
	acc_score    = accuracy_score(groundTruth,predicted)

	if no_classes == 2:
		auc_score = roc_auc_score(1 - groundTruth,scores)

	logging.info ("Confusion-matrix:\n{}".format(conf_matrix))
	logging.info ("Classification report:\n{}".format(class_report))
	logging.info ("Accuracy:{}".format(acc_score))

	if no_classes == 2:
		logging.info ("AUC:{}".format(auc_score))