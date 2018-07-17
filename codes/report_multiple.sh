#######LAB########
mode=LAB
dataset_name=Combined_RimOne_Origa
train_path=/media/htic/NewVolume1/murali/Glaucoma/PretrainDataSets/${dataset_name}/${mode}
val_path=/media/htic/NewVolume1/murali/Glaucoma/Refuge/${mode}
save_path=/media/htic/NewVolume1/murali/Glaucoma/models/${dataset_name}_${mode}
csv_path=${save_path}/result.csv
model_path=${save_path}
cuda_no=0
img_ext=jpg
if [ ! -d "${save_path}" ]; then
mkdir ${save_path}
fi 
train=false
val=true

if [ "$train" = true ]; then 
python classify_multiple_mode.py --train_path ${train_path} --val_path ${val_path} --save_path ${save_path} --cuda_no ${cuda_no}
fi 

if [ "$val" = true ]; then
python eval_multiple.py --model_path ${model_path} --val_path ${val_path} --img_ext ${img_ext} --csv_path ${csv_path} --cuda_no ${cuda_no}
fi 
######################

#########Normalized###############
mode=Normalized
dataset_name=Combined_RimOne_Origa
train_path=/media/htic/NewVolume1/murali/Glaucoma/PretrainDataSets/${dataset_name}/${mode}
val_path=/media/htic/NewVolume1/murali/Glaucoma/Refuge/${mode}
save_path=/media/htic/NewVolume1/murali/Glaucoma/models/${dataset_name}_${mode}
csv_path=${save_path}/result.csv
model_path=${save_path}
cuda_no=0
img_ext=jpg
if [ ! -d "${save_path}" ]; then
mkdir ${save_path}
fi 
train=false
val=true

if [ "$train" = true ]; then 
python classify_multiple_mode.py --train_path ${train_path} --val_path ${val_path} --save_path ${save_path} --cuda_no ${cuda_no}
fi 

if [ "$val" = true ]; then
python eval_multiple.py --model_path ${model_path} --val_path ${val_path} --img_ext ${img_ext} --csv_path ${csv_path} --cuda_no ${cuda_no}
fi 
#############################


###########PseudoDepth############
mode=PseudoDepth
dataset_name=Combined_RimOne_Origa
train_path=/media/htic/NewVolume1/murali/Glaucoma/PretrainDataSets/${dataset_name}/${mode}
val_path=/media/htic/NewVolume1/murali/Glaucoma/Refuge/${mode}
save_path=/media/htic/NewVolume1/murali/Glaucoma/models/${dataset_name}_${mode}
csv_path=${save_path}/result.csv
model_path=${save_path}
cuda_no=0
img_ext=jpg
if [ ! -d "${save_path}" ]; then
mkdir ${save_path}
fi 
train=false
val=true

if [ "$train" = true ]; then 
python classify_multiple_mode.py --train_path ${train_path} --val_path ${val_path} --save_path ${save_path} --cuda_no ${cuda_no}
fi 

if [ "$val" = true ]; then
python eval_multiple.py --model_path ${model_path} --val_path ${val_path} --img_ext ${img_ext} --csv_path ${csv_path} --cuda_no ${cuda_no}
fi 
#########################


###########SegmentMaps###########

mode=SegmentMaps
dataset_name=Combined_RimOne_Origa
train_path=/media/htic/NewVolume1/murali/Glaucoma/PretrainDataSets/${dataset_name}/${mode}
val_path=/media/htic/NewVolume1/murali/Glaucoma/Refuge/${mode}
save_path=/media/htic/NewVolume1/murali/Glaucoma/models/${dataset_name}_${mode}
csv_path=${save_path}/result.csv
model_path=${save_path}
cuda_no=0
img_ext=jpg
if [ ! -d "${save_path}" ]; then
mkdir ${save_path}
fi 
train=false
val=false

if [ "$train" = true ]; then 
python classify_multiple_mode.py --train_path ${train_path} --val_path ${val_path} --save_path ${save_path} --cuda_no ${cuda_no}
fi 

if [ "$val" = true ]; then
python eval_multiple.py --model_path ${model_path} --val_path ${val_path} --img_ext ${img_ext} --csv_path ${csv_path} --cuda_no ${cuda_no}
fi 

#################################
