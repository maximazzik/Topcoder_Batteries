'''
Created on 31 мар. 2018 г.

@author: Novopoltsev Maxim
'''

# importing all necesary modules
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBRegressor, XGBClassifier

# read data from file
def ReadData(data_dir, fnames):
    df = pd.read_csv(data_dir + fname, parse_dates=['date'], index_col='date')
    return df

def TestPrediction(Xtrain, Xtest, ytrain, ytest, model, comment = None):
    # train model, make prediction and return score and conclusion matrix
    try:
        model.fit_transform(Xtrain, ytrain)
    except Exception:
        # if model haven't fit_transform method
        model.fit(Xtrain, ytrain)
    y_pred = model.predict(Xtest)
    print (comment)
    accuracy = accuracy_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)

    print ('accuracy_score = ', accuracy)
    print ('f1_score       = ', f1)
    
    return y_pred, model


def RegressorPrediction(Xtrain, Xtest, ytrain, ytest, model, comment = None):
    # train model, make prediction and return score and conclusion matrix
    model.fit(Xtrain, ytrain)
    y_pred = model.predict(Xtest)
    print (comment)
    score = explained_variance_score(ytest, y_pred)
    print ('score = ', score)
    
    return y_pred, model

def battery_scorer(truth, pred):
    if len(truth) != len(pred):
        return -1, -1, -1

    mat = [ [0, 0], [0, 0] ]
    MRAE, cnt = 0, 0
    for (t, p) in zip(truth, pred):
        mat[int(t == 0)][int(p == 0)] += 1
        if t > 0:
            cnt += 1
            if p == -1:
                MRAE += 1
            else:
                MRAE += min(1.0, abs(p - t) / t)

    MRAE /= cnt
    if mat[1][1] == 0:
        F1 = 0
    else:
        precision = float(mat[1][1]) / (mat[1][1] + mat[0][1])
        recall = float(mat[1][1]) / (mat[1][1] + mat[1][0])
        F1 = precision * recall * 2 / (precision + recall)

    return F1 + (1 - MRAE), F1, MRAE

print ('program starts, expected time of work ~15 min')

print ('reading and filtering data')
# manualy define wich files to process
data_dir = 'Data/'
fname = 'train.csv'
ftest = 'test.csv'

df = pd.read_csv(data_dir + fname)
test_df = pd.read_csv(data_dir + ftest)

# adding 'OK' feature if risk != 0
df['OK'] = np.where(df['risk']==0, 0, 1)
# replace -1 to max risk value 450
df.loc[df.risk == -1, 'risk'] = 450

cat_features = ['event_country_code', 'batt_manufacturer', 'batt_instance']

# encoding cat features and adding to dataframe
vec = DictVectorizer(sparse=False, dtype=int)
encoded_cat_feat = vec.fit_transform(df[cat_features].T.to_dict().values())
# split our data to train and test for classification task
no_cat_features_df = df.drop(cat_features, axis=1)
cat_df = np.hstack((encoded_cat_feat, no_cat_features_df))

# training XGBClassifier 
print ('training classifier')
xgbCV = XGBClassifier(njobs = -1)
bin_Xtrain, bin_Xtest, bin_ytrain, bin_ytest = train_test_split(cat_df[:, :-2], cat_df[:, -1], random_state=1)
bin_xgb_cat_pred, xgbCV_cat = TestPrediction(bin_Xtrain, bin_Xtest, bin_ytrain, bin_ytest, xgbCV, 'xgbCV with categorial features')

battery_OK_prediction = xgbCV_cat.predict(cat_df[:, :-2])
battery_OK_df = np.hstack((battery_OK_prediction.reshape(-1,1), cat_df))

# training XGBRegressor
print ('training regressor')
Xtrain, Xtest, ytrain, ytest = train_test_split(battery_OK_df[:, :-2], battery_OK_df[:, -2], random_state=1)
xgb = XGBRegressor(learning_rate = 0.1, max_depth = 10, n_estimators = 1500, njobs = -1)
xgb_cat_pred, xgb_cat = RegressorPrediction(Xtrain, Xtest, ytrain, ytest, xgb, 'XGB with categorial features')

# build prediction to the test data
# appling DictVectorizer to test data
print ('applying model to test data')
no_cat_features_df = test_df.drop(cat_features, axis=1)
encoded_cat_feat = vec.transform(test_df[cat_features].T.to_dict().values())
cat_test_df = np.hstack((encoded_cat_feat, no_cat_features_df))

# applying XGBClassifier to test data
battery_OK_test_prediction = xgbCV_cat.predict(cat_test_df)
battery_OK_test_df = np.hstack((battery_OK_test_prediction.reshape(-1,1), cat_test_df))

# applying XGBRegressor to test data
xgb_test_cat_pred = xgb_cat.predict(battery_OK_test_df)

# if predicted value is bigger than 400 - battery will live long and happy
corr_pred = [-1 if val >= 440 else val for val in xgb_test_cat_pred]

# replace negative predictions to near zero value (1)
for i in range(len(battery_OK_test_prediction)):
    if ((corr_pred[i] < -1) & (battery_OK_test_prediction[i] == 1)) :
        corr_pred[i] = 1

# replace prediction for 'bad' batteries to 0
final_pred = []
for (p, batt_OK) in zip (corr_pred, battery_OK_test_prediction):
    if batt_OK == 0:
        final_pred.append(0)
    elif (p > 0) and (p < 1):
        final_pred.append(int(1))
    else:
        final_pred.append(int(round(p)))

# write prediction to file
with open('prediction.csv', 'w') as the_file:
    the_file.write("%s\n" % 'RISK')
    for item in final_pred:
        the_file.write(str(int(item)) + '\n')

print ('program ends, thank you for patience')