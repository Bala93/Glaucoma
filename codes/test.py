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
	#output_prob = torch.nn.functional.softmax(output_prob)    
	#max_prob = torch.max(output_prob).item()
	max_prob = np.exp(torch.max(output_prob).item())
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
    '--test_path',
    required = True,
    type = str,
    help = 'Path to test images'
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

    opt = parser.parse_args()
    weight_path = opt.model_path
    test_path   = opt.test_path
    img_ext    = opt.img_ext
    test_path = os.path.join(test_path,'*.' + img_ext)
    csv_path   = opt.csv_path
    cuda_no    = opt.cuda_no
    device = torch.device("cuda:{}".format(cuda_no))	

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
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs,no_classes),nn.LogSoftmax())
    model_ft.to(device)
    model_ft.load_state_dict(torch.load(weight_path))
    model_ft.eval()



    predicted = []
    scores = []
    img_names = []
    for each in tqdm(glob.glob(test_path)):
        img_name = os.path.basename(each)
        output,confidence = predict_class(model_ft,each,device,data_transforms)
        predicted.append([output])
        scores.append([confidence])
        img_names.append([img_name])

    predicted = np.array(predicted)
    scores = np.array(scores)
    img_names = np.array(img_names)

    ind_ = np.where(predicted == 1)
    scores[ind_] = 1 - scores[ind_]
    scores = np.round(scores,decimals=1)
    result = np.hstack([img_names,scores,predicted])
    df = pd.DataFrame(result)
    df.to_csv(csv_path,header=['FileName','Glaucoma Risk','Predicted'],index=False)