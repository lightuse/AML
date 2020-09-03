import numpy as np
np.random.seed(seed=32)

evaluation_list = {'AUC':'roc_auc',
                   'F1':'f1',
                   'Recall':'recall',
                   'Precision':'precision',
                   'Accuracy':'accuracy'}
options_algorithm = ['lightgbm', 'knn', 'ols', 'ridge', 'tree', 'rf', 'gbr1', 'gbr2', 'xgboost', 'catboost']

feature_importances_algorithm_list = ['tree', 'rf', 'gbr1', 'gbr2', 'lightgbm', 'xgboost', 'catboost']

#exception_algorithm_list = ['knn', 'rf', 'ols', 'ridge', 'tree', 'lightgbm', 'gbr1', 'gbr2']
#exception_algorithm_list = ['knn', 'rf', 'ols', 'ridge', 'tree', 'gbr1', 'gbr2', 'xgboost']
exception_algorithm_list = ['lightgbm', 'tree', 'gbr1', 'gbr2']
exception_algorithm_list = ['tree', 'gbr1', 'gbr2', 'knn', 'rf', 'ols', 'ridge', 'xgboost']
exception_algorithm_list = ['tree', 'gbr1', 'gbr2', 'ols', 'ridge', 'rf']
exception_algorithm_list = ['tree', 'gbr1', 'gbr2', 'ols', 'ridge', 'rf', 'knn', 'ols', 'xgboost']

# set pipelines for different algorithms
def setup_algorithm(pipelines = {}):
	import lightgbm as lgb
	import xgboost as xgb
	from catboost import CatBoostRegressor
	from sklearn.preprocessing import StandardScaler
	from sklearn.linear_model import LinearRegression,Ridge
	from sklearn.tree import DecisionTreeRegressor
	from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
	from sklearn.neighbors import KNeighborsRegressor
	from sklearn.pipeline import Pipeline
	from sklearn.decomposition import PCA
	from sklearn.feature_selection import RFE
	pipelines = {
	    'catboost':
	        Pipeline([('pca', PCA(random_state=1)),
	                  ('est', CatBoostRegressor(random_state=1))]),
	    'lightgbm':
	        Pipeline([('reduct', PCA(random_state=1)),
	                  ('est', lgb.LGBMRegressor(random_state=0))]),

	    'knn': Pipeline([('reduct', PCA(random_state=1)),
	                      ('est', KNeighborsRegressor())]), 

	    'ols': Pipeline([('scl', StandardScaler()),
	                      ('est', LinearRegression())]),
	    'ridge':Pipeline([('scl', StandardScaler()),
	                      ('est', Ridge(random_state=0))]),
	    'tree': Pipeline([('reduct', PCA(random_state=1)),
	                    ('est', DecisionTreeRegressor(random_state=1))]),
	    'rf': Pipeline([('reduct', PCA(random_state=1)),
	                    ('est', RandomForestRegressor(max_depth=5, n_estimators=10, random_state=0))]),
	    'gbr1': Pipeline([('reduct', PCA(random_state=1)),
	                      ('est', GradientBoostingRegressor(random_state=0))]),

	    'gbr2': Pipeline([('reduct', PCA(random_state=1)),
	                      ('est', GradientBoostingRegressor(n_estimators=250, random_state=0))]),

	    'xgboost':
	        Pipeline([('pca', PCA(random_state=1)),
	                  ('est', xgb.XGBRegressor(objective ='reg:squarederror', random_state=1))]),

	}
	return pipelines

tuning_prarameter_list = []
# パラメータグリッドの設定
tuning_prarameter = {
    'lightgbm':{
        'est__learning_rate': [0.1,0.05,0.01],
        'est__n_estimators':[1000,2000],
        'est__num_leaves':[31,15,7,3],
        'est__max_depth':[4,8,16]
    },
    'tree':{
        "est__min_samples_split": [10, 20, 40],
        "est__max_depth": [2, 6, 8],
        "est__min_samples_leaf": [20, 40, 100],
        "est__max_leaf_nodes": [5, 20, 100],
    },
    'rf':{
        'est__n_estimators':[5,10,20,50,100],
        'est__max_depth':[1,2,3,4,5],
    }
}


# 表示オプションの変更
import pandas as pd
pd.options.display.max_columns = 100
pd.set_option('display.max_rows', 500)

from sklearn.impute import SimpleImputer
# imputation
def imputation(out_put_data_dir, model_columns_file_name, X_ohe):
    imp = SimpleImputer(strategy='mean')
    imp.fit(X_ohe)
    dump(imp, out_put_data_dir + 'imputer.joblib')
    X_ohe_columns = X_ohe.columns.values
    X_ohe = pd.DataFrame(imp.transform(X_ohe), columns=X_ohe_columns)
    pd.DataFrame(X_ohe_columns).to_csv(model_columns_file_name, index=False)
    pd.DataFrame(X_ohe_columns).to_csv(out_put_data_dir + 'model_columns' + "_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.csv', index=False)
    return X_ohe, X_ohe_columns

# one-hot encoding
def one_hot_encoding(X, ohe_columns):
    X_ohe = pd.get_dummies(X, dummy_na=True, columns=ohe_columns)
    return X_ohe

from sklearn.feature_selection import RFE
from joblib import dump
# feature selection
def feature_selection(out_put_data_dir, n_features_to_select, X, y, X_ohe_columns, algorithm_name, estimator):
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=.05)
    selector.fit(X, y)
    dump(selector, out_put_data_dir + algorithm_name + '_selector.joblib')
    X_fin = X.loc[:, X_ohe_columns[selector.support_]]
    return X_fin

from sklearn.metrics import mean_squared_log_error
import numpy as np
def root_mean_squared_log_error(y_true, y_pred):
    #y_true = np.log1p(y_true)
    #y_pred = np.log1p(y_pred)
    #y_true = np.log1p(y_true)
    #y_pred = np.log1p(y_pred)
    y_pred[y_pred<0] = 0
    y_true[y_true<0] = 0
    #return np.sqrt(np.mean((((y_true)-(y_pred))**2)))
    return np.sqrt(mean_squared_log_error(y_true, y_pred))
    #return np.sqrt(np.mean(((np.log1p(y_pred+1) - np.log1p(y_true+1))**2)))
    #return np.sqrt(np.mean(((np.log1p(y_true+1)-np.log1p(y_pred+1))**2)))
    #return np.sqrt(np.mean(((np.log(y_pred+1))**2)-np.log(y_true+1)))
    #return np.sqrt(mean_squared_log_error(np.exp(y_true), np.exp(y_pred)))
    #return np.sqrt(mean_squared_log_error(np.exp(y_pred), np.exp(y_true)))

import numpy as np
from sklearn.metrics import mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    #y_true = np.log1p(y_true)
    #y_pred = np.log1p(y_pred)
    #y_true = np.log1p(y_true)
    #y_pred = np.log1p(y_pred)
    y_pred[y_pred<0] = 0
    y_true[y_true<0] = 0
    y_pred = np.expm1(y_pred)
    y_true = np.expm1(y_true)
    #return np.sqrt(np.mean((((y_true)-(y_pred))**2)))
    return np.sqrt(mean_squared_error(y_true, y_pred))

from joblib import load
def scoring(out_put_data_dir, algorithm_name, X):
    clf = load(out_put_data_dir + algorithm_name + '_regressor.joblib')
    return clf.predict(X)

# holdout
from sklearn.model_selection import train_test_split
def holdout(X_ohe, y):
    X_train, X_test, y_train, y_test = train_test_split(X_ohe,
                                                y,
                                                test_size=0.2,
                                                random_state=1)
    return X_train, X_test, y_train, y_test
    
def evaluation(pipelines, out_put_data_dir, scores, X_train, y_train, phase, function_evaluation):
    for pipe_name, pipeline in pipelines.items():
        scores[(pipe_name, phase)] = function_evaluation(y_train, (scoring(out_put_data_dir, pipe_name, X_train)))

def get_input(x):
    return x

from ipywidgets import interact,interactive,fixed,interact_manual
from IPython.display import display
import ipywidgets as widgets
def choice(options):
    input = get_input(widgets.RadioButtons(options=options))
    display(input)
    return input

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def cross_validatior(scorer, out_put_data_dir, n_features_to_select, pipelines, input_evaluation, X_ohe, y):
    str_all_print = '評価指標:' + input_evaluation.value + '\n'
    str_all_print += 'n_features_to_select:' + str(n_features_to_select) + '\n'
    print('評価指標:' + input_evaluation.value)
    str_print = ''
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for pipe_name, est in pipelines.items():
        cv_results = -cross_val_score(est, X_ohe, y, cv=kf, scoring=scorer)
        str_print = '----------' + '\n' + 'algorithm:' + str(pipe_name) + '\n' + 'cv_results:' + str(cv_results) + '\n' + 'avg +- std_dev ' + str(cv_results.mean()) + '+-' + str(cv_results.std()) + '\n'
        #print('----------')
        #print('algorithm:', pipe_name)
        #print('cv_results:', cv_results)
        #print('avg +- std_dev', cv_results.mean(),'+-', cv_results.std())
        print(str_print)
        str_all_print += str_print
    import datetime
    with open(out_put_data_dir + 'cv_results' + '_' + str(n_features_to_select) + "_" +  datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt', mode='w') as f:
        f.write(str_all_print)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

def decide_evaluation(input_evaluation):
	function_evaluation = mean_squared_error
	if input_evaluation.value == 'RMSE':
	    function_evaluation = root_mean_squared_error
	elif input_evaluation.value == 'MAE':
	    function_evaluation = mean_absolute_error
	elif input_evaluation.value == 'R2':
	    function_evaluation = r2_score
	elif input_evaluation.value == 'RMSLE':
	    function_evaluation = root_mean_squared_log_error
	return function_evaluation

def display_evaluation(is_holdout, out_put_data_dir, pipelines, function_evaluation, input_evaluation, X_train, y_train, X_valid, y_valid):
    scores = {}
    if is_holdout:
        evaluation(pipelines, out_put_data_dir, scores, X_train, y_train, 'train', function_evaluation)
        evaluation(pipelines, out_put_data_dir, scores, X_valid, y_valid, 'valid', function_evaluation)
    else:
        evaluation(pipelines, out_put_data_dir, scores, X_train, y_train, 'train', function_evaluation)
    # sort score
    #sorted_score = sorted(scores.items(), key=lambda x:-x[1])
    ascending = True
    if input_evaluation.value == 'R2':
       ascending = False

    print('評価指標:' + input_evaluation.value)

    if is_holdout:
        display(pd.Series(scores).unstack().sort_values(by=['valid'], ascending=[ascending]))
    else:
        display(pd.Series(scores).unstack().sort_values(by=['train'], ascending=[ascending]))

# preprocessing
def preprocessing(out_put_data_dir, model_columns_file_name, algorithm_name, X_ohe, X_ohe_s):
    model_columns = pd.read_csv(model_columns_file_name)
    X_ohe_columns = model_columns.values.flatten()
    cols_model = set(X_ohe_columns)
    cols_score = set(X_ohe_s.columns.values)
    diff1 = cols_model - cols_score
    print('モデルのみに存在する項目: %s' % diff1)
    diff2 = cols_score - cols_model
    print('スコアのみに存在する項目: %s' % diff2)
    df_cols_m = pd.DataFrame(None, columns=X_ohe_columns, dtype=float)
    X_ohe_s2 = pd.concat([df_cols_m, X_ohe_s])
    set_Xm = set(X_ohe.columns.values)
    set_Xs = set(X_ohe_s.columns.values)
    X_ohe_s3 = X_ohe_s2.drop(list(set_Xs-set_Xm), axis=1)
    X_ohe_s3.loc[:,list(set_Xm-set_Xs)] = X_ohe_s3.loc[:,list(set_Xm-set_Xs)].fillna(0, axis=1)
    X_ohe_s3 = X_ohe_s3.reindex(X_ohe.columns.values, axis=1)
    imp = load(out_put_data_dir + 'imputer.joblib')
    X_ohe_s4 = pd.DataFrame(imp.transform(X_ohe_s3), columns=X_ohe_columns)
    selector = load(out_put_data_dir + algorithm_name + '_selector.joblib')
    X_fin_s = X_ohe_s4.loc[:, X_ohe_columns[selector.support_]]
    return X_fin_s

import datetime
def output_file(out_put_data_dir, n_features_to_select, target_label, df, id_label, y, model_name, extension, header=True):
    y = [i if i > 0.01 else 0 for i in y]
    file_name = out_put_data_dir + "submittion_" + model_name + "_" + str(n_features_to_select) + "_" +  datetime.datetime.now().strftime('%Y%m%d%H%M%S') + "." + extension
    separator = ','
    if extension == 'tsv':
        separator = '\t'
    if id_label != '':
        pd.concat([df[id_label], pd.DataFrame(y, columns=[target_label])], axis=1).to_csv(file_name, index=False, sep=separator, header=header)
        #concated = pd.concat([df, pd.DataFrame(y, columns=[target_label])], axis=1)
        #pd.DataFrame(concated[id_label, target_label]).to_csv(file_name, index=False, sep=separator, header=header)
        #concated = pd.concat([df, pd.DataFrame(y, columns=[target_label])], axis=1)
        #pd.DataFrame(concated[id_label, target_label]).to_csv(file_name, index=False, sep=separator, header=header)
        #concated[id_label, target_label].to_csv(file_name, index=False, sep=separator, header=header)
    else:
        pd.concat([df, pd.DataFrame(y, columns=[target_label])], axis=1).to_csv(file_name, index=False, sep=separator, header=header)

# train
import optuna.integration.lightgbm as lgb
from joblib import dump
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
def train_model(out_put_data_dir, pipelines, feature_selection_rf_list, n_features_to_select, X, y, X_ohe_columns, evaluation, is_holdout, is_optuna):
    #scorer = make_scorer(rmsle, greater_is_better=False)
    # fit
    for pipe_name, pipeline in pipelines.items():
        print(pipe_name)
        if pipe_name in feature_selection_rf_list:
            X_featured = feature_selection(out_put_data_dir, n_features_to_select, X, y, X_ohe_columns, pipe_name, pipelines['rf'].named_steps['est'])
        else:
            X_featured = feature_selection(out_put_data_dir, n_features_to_select, X, y, X_ohe_columns, pipe_name, pipeline.named_steps['est'])
        X_featured.to_csv(out_put_data_dir + "X_featured.csv", index=False, header=True)
        if is_holdout:
            X_train, X_valid, y_train, y_valid = holdout(X_featured, y)
        else:
            X_train, X_valid, y_train, y_valid = X_featured, X_featured, y, y
        if pipe_name in tuning_prarameter_list:
            gs = GridSearchCV(estimator=pipeline,
                        param_grid=tuning_prarameter[pipe_name],
                        #scoring=evaluation_list[evaluation],
                        scoring=scorer,
                        cv=2,
                        return_train_score=False,
                        verbose=1)
            gs.fit(X_train, y_train)
            dump(gs, out_put_data_dir + pipe_name + '_regressor.joblib')
            gs.fit(X_valid, y_valid)
            #print(gs.best_estimator_)
            # 探索した結果のベストスコアとパラメータの取得
            #print(pipe_name + ' Best Score:', gs.best_score_)
            print(pipe_name + ' Best Params', gs.best_params_)
        else:
            if is_optuna and pipe_name in 'lightgbm':
                #lgb_train = lgb.Dataset(X_train, np.log1p(y_train))
                #lgb_eval = lgb.Dataset(X_valid, np.log1p(y_valid), reference=lgb_train)
                lgb_train = lgb.Dataset(X_train, (y_train))
                lgb_eval = lgb.Dataset(X_valid, (y_valid), reference=lgb_train)
                params = {
                    'boosting_type': 'gbdt',
                    'metric': 'rmse',
                    'objective': 'regression',
                    'seed': 20,
                    'learning_rate': 0.01,
                    "n_jobs": -1,
                    "verbose": -1
                }
                best = lgb.train(params,
                            lgb_train,
                            valid_sets=[lgb_train, lgb_eval],
                            verbose_eval=0)
                dump(best, out_put_data_dir + pipe_name + '_regressor.joblib')  
            else:
                clf = pipeline.fit(X_train, y_train)
                dump(clf, out_put_data_dir + pipe_name + '_regressor.joblib')
                if is_holdout:
                    clf = pipeline.fit(X_valid, y_valid)

    return X_train, X_valid, y_train, y_valid


def convert_to_count_encode(X, column:str):
    import collections
    counter = collections.Counter(X[column].values)
    count_dict = dict(counter.most_common())
    encoded = X[column].map(lambda x: count_dict[x]).values
    return encoded

def convert_to_label_encode(X, column:str):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    encoded = le.fit_transform(X[column].values.astype(str))
    #encoded = X[column].apply(lambda i:le.fit_transform(X[column].astype(str)), axis=0, result_type='expand')
    return encoded

def convert_to_label_count_encode(X, column:str):
    import collections
    counter = collections.Counter(X[column].values)
    count_dict = dict(counter.most_common())
    label_count_dict = {key:i for i, key in enumerate(count_dict.keys(), start=1)}
    encoded = X[column].map(lambda x: label_count_dict[x]).values
    return encoded

# making
def convert_to_target_encode(X, column:str, target):
    target_dict = X[[column, target]].groupby([column])[target].mean().to_dict()
    encoded = X[column].map(lambda x: target_dict[x]).values
    return encoded

def convert_to_frequency_encode(X, column:str):
    encoded = X.groupby(column).size() / len(X)
    encoded = X[column].map(encoded)
    return encoded
