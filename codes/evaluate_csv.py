import pandas as pd
import numpy as np
import os 
from scipy import stats
from sklearn.metrics import roc_auc_score

out_csv = '/media/htic/Balamurali/Glaucoma_models/submit.csv'
src_path = '/media/htic/Balamurali/Glaucoma_models/'
csv_name = 'output.csv'
test_data_len = 400
color_space = ['LAB','Normalized','PseudoDepth']
models = ['densenet169','resnet101','densenet201']
score_np = np.empty([test_data_len,0])
pred_np  = np.empty([test_data_len,0])
is_test  = False


for model in models:

	for color in color_space:

		csv_path   = os.path.join(src_path,'{}_{}'.format(model,color),csv_name)
		pd_data    = pd.read_csv(csv_path)
		file_names = pd_data['FileName'].values.reshape(400,1)
		pred_data  = pd_data['Predicted'].values.reshape(400,1)
		score_data = pd_data['Glaucoma Risk'].values.reshape(400,1)
		pred_np    = np.hstack([pred_np,pred_data])
		score_np   = np.hstack([score_np,score_data])

		# break
	# break
best_predict,_ = stats.mode(pred_np,axis=1)
best_predict = best_predict.astype(np.uint8)

score_np_min = np.min(score_np,axis=1).reshape(400,1)
score_np_max = np.max(score_np,axis=1).reshape(400,1)

zero_mask = np.where(best_predict == 0)
one_mask  = np.where(best_predict == 1)

result_score = np.zeros([400,1])
result_score[zero_mask] = score_np_max[zero_mask]
result_score[one_mask]  = score_np_min[one_mask]

# print (result_score)

if is_test:
	gt = np.zeros((400,1))
	gt[:40] = 1
	# print(gt)
	print(roc_auc_score(gt,result_score)) 

result = np.hstack([file_names,result_score])
df = pd.DataFrame(result)
df.to_csv(out_csv,header=['FileName','Glaucoma Risk'],index=False)