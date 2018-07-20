src_train_path='/media/htic/NewVolume1/murali/Glaucoma/PretrainDataSets/Combined_RimOne_Origa'
src_val_path='/media/htic/NewVolume1/murali/Glaucoma/Refuge'
src_save_path='/media/htic/NewVolume3/Balamurali/Glaucoma_models'
cuda_no=0
no_classes=2
is_pretrained=1

for model_name in 'inception' #'resnet101' 'resnet152' 'densenet169' 'densenet201'  'vgg16_bn' 'vgg19_bn' 
  do
  for color in 'LAB' 'Normalized' 'PseudoDepth'
   do 
   train_path=${src_train_path}/${color}
   val_path=${src_val_path}/${color}
   save_path=${src_save_path}/${model_name}_${color}
   python classify_custom.py --train_path ${train_path} --val_path ${val_path} --save_path ${save_path} --cuda_no ${cuda_no} --model_name ${model_name} --no_classes ${no_classes} --is_pretrained ${is_pretrained}
   done
  done
