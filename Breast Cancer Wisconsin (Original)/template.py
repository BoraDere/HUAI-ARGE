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

    'LogisticRegression': {
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'C': [0.1, 1, 10, 100, 1000],
        'max_iter': [100, 300, 500],
        'multi_class': ['ovr']
    },

    'RandomForestClassifier': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },

    'SVC': {
        'C': [0.1, 1, 10, 100, 1000],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['auto', 'scale']
    },

    'MLPClassifier': {
        'hidden_layer_sizes': [(50,), (50, 50), (100,), (100, 100)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'learning_rate_init': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'max_iter': [100, 200, 300, 400, 500]
    },

    'GradientBoostingClassifier': {
        'loss': ['log_loss', 'exponential'],
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [100, 200, 300, 400, 500],
        'min_samples_split': [2, 4, 6, 8],
        'max_depth': [3, 5, 7]
    }
}


def param_finder(extension, model, X_train):
    param_grid = param_grids[model]
    model = globals()[model](random_state=42)

    grid = GridSearchCV(model, param_grid, n_jobs=-1, cv=5)
    grid.fit(X_train, globals()['y_train_' + extension])

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
    match(extension): # no match, switch to if elif
        case 'with_outliers', 'no_outliers', 'nan_outliers', 'nan_outliers_na_dropped':
            X_train, X_test = impute(extension, model)
            params = param_finder(extension, model)
            
            model = globals()[model](random_state=42, **params)
            model.fit(X_train, globals()['y_train_' + extension])

        case 'with_outliers_na_dropped', 'no_outliers_na_dropped':
            params = param_finder(extension, model)

            model = globals()[model](random_state=42, **params)
            model.fit(globals()['X_train_' + extension], globals()['y_train_' + extension])
    

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(globals()['y_test_' + extension], y_pred)
    precision = precision_score(globals()['y_test_' + extension], y_pred)
    recall = recall_score(globals()['y_test_' + extension], y_pred)
    roc_auc = roc_auc_score(globals()['y_test_' + extension], y_pred)

    print(f"Accuracy: {accuracy * 100.0}%")
    print(f"Precision: {precision * 100.0}%")
    print(f"Recall: {recall * 100.0}%")
    print(f"ROC AUC: {roc_auc * 100.0}%")

    globals()['d_' + extension][model] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'ROC AUC': roc_auc}