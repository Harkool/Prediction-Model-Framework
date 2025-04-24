from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_all_models(train_df, test_df, features, target, random_state):
    X_train, y_train = train_df[features], train_df[target]
    X_test = test_df[features]

    models = {
        "LR": LogisticRegression(),
        "RegLR": LogisticRegression(penalty='l2', C=0.1),
        "MLP": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500),
        "XGB": XGBClassifier(eval_metric='logloss'),
        "CatBoost": CatBoostClassifier(verbose=0)
    }

    preds = {}
    for name, model in models.items():
        pipeline = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        pipeline.fit(X_train, y_train)
        preds[name] = pipeline.predict_proba(X_test)[:, 1]
        models[name] = pipeline
    return models, preds