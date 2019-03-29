import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
import utils
from sklearn.utils import shuffle
from metrics import gini_norm
from DeepFM import DeepFM
from DataReader import FeatureDictionary, DataParser
gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)



# params
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": gini_norm
}


def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params,
	numerical_cols = ):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, y_test = data_parser.parse(df=dfTest,has_label=True)

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])
    dfm = DeepFM(**dfm_params)
    
    
    
    dfm.fit(Xi_train, Xv_train, y_train,early_stopping=True)
    pred = dfm.predict(Xi_test,Xv_test)
    print(pred)
    dfm.evaluate(Xi_test, Xv_test, y_test)

    '''
   

    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    _get = lambda x, l: [x[i] for i in l]
    gini_results_cv = np.zeros(len(folds), dtype=float)
    gini_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

        y_train_meta[valid_idx,0] = dfm.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:,0] += dfm.predict(Xi_test, Xv_test)

        gini_results_cv[i] = gini_norm(y_valid_, y_train_meta[valid_idx])
        gini_results_epoch_train[i] = dfm.train_result
        gini_results_epoch_valid[i] = dfm.valid_result

    y_test_meta /= float(len(folds))

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)"%(clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, gini_results_cv.mean(), gini_results_cv.std())

    _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    return y_train_meta, y_test_meta
	'''


if __name__ == '__main__':
    data = utils.generate_training_data()
    data = data.rename(columns={'label':'target','user_id':'id'})
    print('load data success')
    
    data = shuffle(data)
    data = data.iloc[:1000,:20]
    print(data[:2])

    train = data[:int(0.7*data.shape[0])]
    test = data[int(0.7*data.shape[0]):]
    
    y_train = train['target']
    X_train = train.drop(['id','target'],axis=1)
    numerical_cols = train.drop(['target','id','click_article_id']).columnt.tolist()
    folds = list(StratifiedKFold(n_splits=3, shuffle=True,random_state=42).split(X_train, y_train))
    _run_base_model_dfm(train,test,folds,dfm_params)