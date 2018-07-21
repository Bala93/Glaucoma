src_model_path='/media/htic/Balamurali/Glaucoma_models'
src_val_path='/media/htic/NewVolume1/murali/Glaucoma/Refuge'
img_ext='jpg'
src_csv_path='/media/htic/Balamurali/Glaucoma_models'
cuda_no='0'

for model_name in 'densenet169' #'resnet101' #'resnet152' 'densenet201'   #'vgg16_bn' 'vgg19_bn' 'inception'   
  do
  for color in 'LAB' 'Normalized' 'PseudoDepth'
   do 
   val_path=${src_val_path}/${color}
   model_path=${src_model_path}/${model_name}_${color}
   echo ${model_name}_${color}
   csv_path=${src_csv_path}/${model_name}_${color}
   python evaluate_custom.py --model_path ${model_path} --val_path ${val_path} --img_ext ${img_ext} --csv_path ${csv_path} --cuda_no ${cuda_no} --model_name ${model_name}
   done
  done
