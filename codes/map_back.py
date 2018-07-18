import cv2
import json
import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    
    json_path = 'predicted.json'
    mask_path  = '/media/balamurali/Balamurali1/Sharath/Gl_challenge/ValidationResults/LAB1_Out_maps'
    out_path   = '/media/balamurali/Balamurali1/Sharath/Gl_challenge/ValidationResults/Lab1_whole_map'

    with open(json_path) as f:
        img_info = json.load(f)

    for key in tqdm(img_info.keys()):
        coord = img_info[key]
        mask  = cv2.imread(os.path.join(mask_path,key),0)
        mask_whole = 255 * np.ones([1634,1634],dtype=np.uint8)
        mask_whole[coord[1]:coord[3],coord[0]:coord[2]] = mask
        cv2.imwrite(os.path.join(out_path,key),mask_whole)