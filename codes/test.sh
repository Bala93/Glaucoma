model_name=27
color_space='LAB'
csv_name='result'${color_space}'_'${model_name}'.csv'
model_path='/media/htic/NewVolume1/murali/Glaucoma/models/Combined_RimOne_Origa_'${color_space}'/'${model_name}'.pt'
test_path='/media/htic/Balamurali/Sharath/Gl_challenge/REFUGE-Validation400/TransformedImages/'${color_space}
img_ext='jpg'
csv_path='/media/htic/Balamurali/Sharath/Gl_challenge/REFUGE-Validation400/Results/'${csv_name}
echo ${model_path}
echo ${test_path}
echo ${img_ext}
echo ${csv_path}
python test.py --model_path ${model_path} --test_path ${test_path} --img_ext ${img_ext} --csv_path ${csv_path} --cuda_no 1
