model_path=/media/htic/NewVolume1/murali/Glaucoma/models/Combined_RimOne_Origa_Normalized/3.pt
val_path=/media/htic/NewVolume1/murali/Glaucoma/Refuge/Normalized/Images/
img_ext=jpg
cuda_no=0
python evaluate.py --model_path ${model_path} --val_path ${val_path} --img_ext ${img_ext} --cuda_no ${cuda_no}
