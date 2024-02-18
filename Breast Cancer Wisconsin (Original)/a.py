import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

extensions = [
    'with_outliers',
    'with_outliers_na_dropped',
    'no_outliers',
    'no_outliers_na_dropped',
    'nan_outliers',
    'nan_outliers_na_dropped'
]

models = [
    'XGBClassifier',
    'LogisticRegression',
    'RandomForestClassifier',
    'SVC',
    'MLPClassifier',
    'GradientBoostingClassifier'
]

param_grids = {
    'XGBClassifier': {
        'learning_rate': [0.1, 0.01, 0.001],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7, 1.0],
        'colsample_bytree': [0.5, 0.7, 1.0],
        'n_estimators' : [100, 200, 500],
        'objective': ['binary:logistic']
    },
    
}


def param_finder(extension, model):
    param_grid = param_grids[model]
    model = globals()[model](random_state=42)

    grid = GridSearchCV(model, param_grid, n_jobs=-1, cv=5)
    grid.fit(globals()['X_train_' + extension], globals()['y_train_' + extension])

    return grid.best_params_


def impute(extension, model):
    best_score = -np.inf
    best_k_train = 1
    best_k_test = 1

    for k in range(1, 21):
        imputer = KNNImputer(n_neighbors=k)
        X_imputed = imputer.fit_transform(globals()['X_train' + extension])

        model = globals()[model](random_state=42)
        scores = cross_val_score(model, X_imputed, globals()['y_train_' + extension], cv=5)
        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_k_train = k

    best_score = -np.inf

    for k in range(1, 21):
        imputer = KNNImputer(n_neighbors=k)
        X_imputed = imputer.fit_transform(globals()['X_test' + extension])

        model = globals()[model](random_state=42)
        scores = cross_val_score(model, X_imputed, globals()['y_test_' + extension], cv=5)
        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_k_test = k

    return KNNImputer(n_neighbors=best_k_train).fit_transform(globals()['X_train_' + extension]), KNNImputer(n_neighbors=best_k_test).fit_transform(globals()['X_test_' + extension])


def model_eval(extension, model):
    match(extension):
        case 'with_outliers':
            globals()['X_train_' + extension], globals()['X_test_' + extension] = impute(extension, model)
            params = param_finder(extension, model)

            
