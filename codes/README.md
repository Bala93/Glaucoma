## Algorithms



## Steps:
1. Whole image is resized to the size expected by faster rcnn
2. The faster rcnn gives the coordinates at which the optic disk is present in resized image. 
3. The optic disk coordinates is transformed to the original image dimensions.
4. Store the image name x,y,w,h in the csv file. 
5. A folder containing the cropped images will be created based on this coordinates.
6. The optic disk will be resized according to the segmentation network. 
7. The resized optic disk will be given to the network. 
8. The obtained segmentaion mask will be resized to the actual one. 
9. Using the csv, map the obtained resized segmentation mask to the image.