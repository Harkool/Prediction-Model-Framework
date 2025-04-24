import pandas as pd
from data_loader import load_data
from models import train_all_models
from evaluation import evaluate_models
from statistics import calculate_nri_idi, delong_test
from shap_analysis import shap_summary

if __name__ == "__main__":
    RANDOM_STATE = 42
    train_df, test_df = load_data("traindata.CSV", "testdata.CSV")

    pre_features = ["Age", "Preoperative LVEF", "Preoperative Cr", 
                    "Preoperative leukocyte", "ASA physical status", 
                    "PAP", "Emergency treatment", "COPD"]

    full_features = ["Age", "Preoperative LVEF", "CPB duration",
                     "Intraoperative crystalloid infusion", "Intraoperative colloid infusion",
                     "Intraoperative autologous blood transfusion", "Preoperative Cr", 
                     "Preoperative leukocyte", "ASA physical status", "PAP", 
                     "Emergency treatment", "COPD"]

    target = "RF"
    feature_set = pre_features  # 可改为 full_features
    models, preds = train_all_models(train_df, test_df, feature_set, target, RANDOM_STATE)
    evaluate_models(models, preds, test_df[target].values)
    calculate_nri_idi(models, test_df[feature_set], test_df[target])
    delong_test(models, test_df[feature_set], test_df[target])
    shap_summary(models, test_df[feature_set])