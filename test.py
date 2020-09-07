import pandas as pd
import numpy as np
import math
import random
from numpy import arange
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,MultiLabelBinarizer


from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
import deepctr
from keras import backend
from deepctr.models import DeepFM,FNN,AutoInt,DIN,xDeepFM,LR
from deepctr.inputs import SparseFeat,VarLenSparseFeat, DenseFeat, get_feature_names
import kerastuner
from tensorflow.python.keras.models import load_model
from deepctr.layers.utils import *
from deepctr.layers.core import *
from deepctr.layers.interaction import *
from deepctr.layers.sequence import *

from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import os


def BinaryValue(pred_ans, threshold):
    pred_ans = [1 if x > threshold else 0 for x in pred_ans]
    return pred_ans


# Get threshold of a certain postiveRate
def getThreshold(pred_ans, start, positiveRate):
    for threshold in arange(start, 1.0, 0.0005):
        #         start=datetime.now()
        count = pd.DataFrame(BinaryValue(pred_ans, threshold))
        rate = count[0].value_counts()[1] / (count[0].value_counts()[0] + count[0].value_counts()[1])
        #         end=datetime.now()
        #         print('time for computing:',end-start,' Rate: ',rate)
        if rate < positiveRate:
            return threshold


def Matrics(y_true, y_pred, threshold):
    binariedPred = [1 if x > threshold else 0 for x in y_pred]
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, binariedPred, labels=[0, 1])
    print("precision:", precision)
    print("recall:", recall)
    print("fscore:", fscore)
    print("support", support)
    auc = round(roc_auc_score(y_true, y_pred), 4)
    print("test AUC", auc)
    return (precision, recall, fscore, auc)


def split(x, dictName):
    key_ans = x[1:-1].split(",")

    for key in key_ans:
        if key not in dictName:
            dictName[key] = len(dictName) + 1
    return list(map(lambda x: dictName[x], key_ans[:10]))


def loadData(trainFile, testFile, embedding_dim, multivalue_len, multiClass=False):
    train = pd.read_csv(trainFile)
    test = pd.read_csv(testFile)

    ##1. feature type declarion
    sparse_features = ["BaseAdGroupId", "Criteria", 'placementType', 'Week', 'IsRestrict', 'IsNegative',
                       'AccountTimeZone', 'AccountCurrencyCode', 'BiddingStrategyType', 'CampaignId', 'Month']

    dense_features = ['adClicks', 'adConversions', 'adCtr', 'adConversionRate', 'adActiveViewImpressions',
                      'adActiveViewMeasurability',
                      'adActiveViewMeasurableCost', 'adActiveViewViewability', 'adImpressions', 'adActiveViewCpm',
                      'adAverageCpc', 'adAverageCpe',
                      'adCpcBid', 'adActiveViewMeasurableImpressions', 'adActiveViewCtr', 'adAverageCpm',
                      'adAverageCpv', 'adCost',
                      'plaClicks', 'plaConversions', 'plaCtr', 'plaConversionRate', 'plaActiveViewImpressions',
                      'plaActiveViewMeasurability',
                      'plaActiveViewMeasurableCost', 'plaActiveViewViewability', 'plaImpressions', 'plaCpcBid',
                      'plaActiveViewMeasurableImpressions',
                      'plaActiveViewCtr', 'plaActiveViewCpm', 'plaAverageCpc', 'plaAverageCpe', 'plaAverageCpm',
                      'plaAverageCpv', 'plaCost',
                      'histListLen']
    multivalue_features = ['locationName', 'languageCode', 'hist_BaseAdGroupId']
    sparse_features = ["BaseAdGroupId", "Criteria", 'placementType']
    target = ['Ctr']

    # 2. Missing value process.
    train[sparse_features + multivalue_features] = train[sparse_features + multivalue_features].fillna('-1', )
    train[dense_features + target] = train[dense_features + target].fillna(0, )
    test[sparse_features + multivalue_features] = test[sparse_features + multivalue_features].fillna('-1', )
    test[dense_features + target] = test[dense_features + target].fillna(0, )

    train["BaseAdGroupId"] = train["BaseAdGroupId"].apply(lambda x: str((int(x))))

    test["BaseAdGroupId"] = test["BaseAdGroupId"].apply(lambda x: str((int(x))))

    # 3. sparse features transformation
    for feat in sparse_features:
        lbe = LabelEncoder()
        train[feat] = lbe.fit_transform(train[feat])
        test[feat] = lbe.fit_transform(test[feat])

    # 4. dense features transformation
    for numFeature in dense_features:
        train[numFeature] = train[numFeature].apply(lambda x: x if x < 2 else math.sqrt(math.log(x)))
        test[numFeature] = test[numFeature].apply(lambda x: x if x < 2 else math.sqrt(math.log(x)))

    # 5. multivalue features transformation

    for feat in multivalue_features:
        exec('{}_train_list = list([split(x,{}Dict) for x in train[feat].values])'.format(feat, feat, feat))
        exec('{}_test_list = list([split(x,{}Dict) for x in test[feat].values])'.format(feat, feat, feat))

        exec('{}_length = np.array(list(map(len, {}_train_list)))'.format(feat, feat))
        exec('{}_maxlen = max({}_length)'.format(feat, feat))

        exec('{}_train_list = pad_sequences({}_train_list, maxlen=multivalue_len, padding="post",)'.format(feat, feat,
                                                                                                           feat))
        exec('{}_test_list = pad_sequences({}_test_list, maxlen=multivalue_len, padding="post",)'.format(feat, feat,
                                                                                                         feat))

    # 6. feature colums
    fixlen_feature_columns = [SparseFeat(feat,
                                         vocabulary_size=(train[feat].append(test[feat], ignore_index=True)).nunique(),
                                         embedding_dim=embedding_dim)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                            for feat in dense_features]

    varlen_feature_columns = []
    for feat in multivalue_features:
        exec(
            'varlen_feature_columns.append(VarLenSparseFeat("{}", maxlen= multivalue_len,vocabulary_size=len({}Dict) + 1,embedding_dim=embedding_dim, combiner="mean",weight_name=None))'.format(
                str(feat), feat))

    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 7.generate input data for model
    train_model_input = {name: train[name] for name in sparse_features + dense_features}
    test_model_input = {name: test[name] for name in sparse_features + dense_features}

    for feat in multivalue_features:
        name = str(feat)
        exec('train_model_input["{}"] = {}_train_list'.format(name, feat))
        exec('test_model_input["{}"] = {}_test_list'.format(name, feat))

    behavior_feature_list = ["BaseAdGroupId"]
    return train_model_input, train, test_model_input, test, dnn_feature_columns, linear_feature_columns, behavior_feature_list

locationNameDict={}
languageCodeDict={}
hist_BaseAdGroupIdDict={}
target='level'
train_model_input,train,test_model_input,test,dnn_feature_columns,linear_feature_columns,behavior_feature_list \
= loadData('Train20sample.csv','Test20sample.csv',embedding_dim=10,multivalue_len=5)
y_true=keras.utils.to_categorical(train[target].values,4)
print(behavior_feature_list)
# model = DIN( dnn_feature_columns, behavior_feature_list,task='multiclass', dnn_hidden_units=(256, 128),
            # l2_reg_embedding=0.00001, dnn_dropout=0.5, l2_reg_dnn=0.0001,nClass=4)
# model = LR(linear_feature_columns, l2_reg_linear=0.0001, init_std=0.0001, seed=1024, task='multiclass',nClass=4)
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='multiclass', dnn_hidden_units=(256, 128),
               l2_reg_embedding=0.0001, l2_reg_linear=0.0001, dnn_dropout=0.5,
               l2_reg_dnn=0.0001,nClass=4)

model.compile(optimizer=keras.optimizers.Adam(0.0001), loss="categorical_crossentropy", metrics=['acc'])
history=model.fit(train_model_input,y_true,callbacks=[keras.callbacks.EarlyStopping(monitor='loss',patience=5),
                                                                    keras.callbacks.ModelCheckpoint("MCDIN.h5",monitor="loss",verbose=1,save_best_only=True)],
                  batch_size=1024,epochs=200,verbose=2)
pred_ans = model.predict(test_model_input, batch_size=256)
print(pred_ans)