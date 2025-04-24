from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def evaluate_models(models, preds, y_true):
    for name, y_pred_prob in preds.items():
        print(f"--- {name} ---")
        auc = roc_auc_score(y_true, y_pred_prob)
        pr = average_precision_score(y_true, y_pred_prob)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        f1 = f1_score(y_true, y_pred)
        print(f"AUC: {auc:.3f}, AUPRC: {pr:.3f}, Sens: {sens:.3f}, Spec: {spec:.3f}, F1: {f1:.3f}")