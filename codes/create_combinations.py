import itertools as it
# import os
import pandas as pd
import numpy as np
import os 
from scipy import stats
from sklearn.metrics import roc_auc_score,classification_report
# l = [1,2,3,4]
# l2 = ["a","b","c"]
out_csv = '/media/htic/Balamurali/Glaucoma_models/classification_results.csv'
src_path = '/media/htic/Balamurali/Glaucoma_models/'
test_data_len = 400
# Change for test
is_test  = False
csv_name = 'result.csv'
color_space = ['LAB','Normalized','PseudoDepth']
models = ['densenet201','resnet101','densenet169','resnet152']
l3 = models+color_space
# print(l3)
r = [i for x in range(2,8) for i in it.combinations(l3,x)]
s = []
# best_tp = 
min_shape = 999
max_roc_auc = 0
for item in r:
    if(set(item).intersection(set(models)) and set(item).intersection(set(color_space))):
        models_ = list(set(item).intersection(set(models)))
        color_space_ = list(set(item).intersection(set(color_space)))
        score_np = np.empty([test_data_len,0])
        pred_np  = np.empty([test_data_len,0])
        for model in models_:
    
            for color in color_space_:

                csv_path   = os.path.join(src_path,'{}_{}'.format(model,color),csv_name)
                pd_data    = pd.read_csv(csv_path)
                file_names = pd_data['FileName'].values.reshape(400,1)
                pred_data  = pd_data['Predicted'].values.reshape(400,1)
                score_data = pd_data['Glaucoma Risk'].values.reshape(400,1)
                pred_np    = np.hstack([pred_np,pred_data])
                score_np   = np.hstack([score_np,score_data])

                # break
            # break
            #print (file_names)
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

        # if is_test:
        gt = np.zeros((400,1))
        gt[:40] = 1

        if(zero_mask[0].shape[0]<min_shape):
            min_shape = zero_mask[0].shape[0]
            best_tp = models_+color_space_
            tp_auc = roc_auc_score(gt,result_score)

        if(roc_auc_score(gt,result_score)>max_roc_auc):
            max_roc_auc = roc_auc_score(gt,result_score)
            best_auc = models_+color_space_
            auc_tp   = zero_mask[0].shape[0]

        # result = np.hstack([file_names,result_score])
        # df = pd.DataFrame(result)
        # df.to_csv(out_csv,header=['FileName','Glaucoma Risk'],index=False)


print(min_shape,best_tp,tp_auc,max_roc_auc,best_auc,auc_tp)
# print(r)
# print(len(r))
# print(s)
# print(len(s))


# print(len(it.combinations(l)),len(it.combinations(l2)))