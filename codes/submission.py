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
from model_custom import ModelSelect


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
    '--weight_path',
    required = True,
    type = str,
    help = 'path of the model file'
    )

    parser.add_argument(
    '--src_test_path',
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
    '--src_csv_path',
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

    # python submission.py --weight_path /media/htic/Balamurali/Glaucoma_models --src_csv_path /media/htic/Balamurali/Glaucoma_models --src_test_path /media/htic/Balamurali/Sharath/Gl_challenge/REFUGE-Validation400/TransformedImages --cuda_no 0 --img_ext jpg

    opt = parser.parse_args()
    weight_path = opt.weight_path
    src_test_path   = opt.src_test_path
    img_ext    = opt.img_ext
    src_csv_path   = opt.src_csv_path
    cuda_no    = opt.cuda_no
    device = torch.device("cuda:{}".format(cuda_no))	



    # Normalization 
    data_transforms = transforms.Compose([transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms. transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    models = ['resnet101','resnet152','densenet169','densenet201']
    color_space = ['LAB','Normalized','PseudoDepth']
    # No of classes
    no_classes = 2
    is_pretrained = False
    # Model initiliazation 

    for model_name in models:
        for color in color_space:
            
            csv_path = os.path.join(src_csv_path,'{}_{}'.format(model_name,color),'output.csv')
            model_path = os.path.join(weight_path,'{}_{}'.format(model_name,color),'result.pt')
            model_ft = ModelSelect(model_name,is_pretrained,no_classes).getModel()
            model_ft = nn.Sequential(model_ft,nn.LogSoftmax())
            model_ft = model_ft.to(device) 
            model_ft.load_state_dict(torch.load(model_path))
            model_ft.eval()

            predicted = []
            scores = []
            img_names = []
            
            test_path = os.path.join(src_test_path,color,'*.{}'.format(img_ext))
            print (model_path)

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
            #result = np.hstack([img_names,scores])
            df = pd.DataFrame(result)
            df.to_csv(csv_path,header=['FileName','Glaucoma Risk','Predicted'],index=False)