# Postoperative Respiratory Failure Prediction Framework

This project provides a complete machine learning pipeline for predicting postoperative respiratory failure (RF), including model comparison, performance evaluation, visualization, and interpretability analysis.

---

## 📂 Project Structure
```
model_pipeline/
├── main.py                  # Main script to run the full workflow
├── data_loader.py           # Data loading and preprocessing module
├── models.py                # Model building and training (LR, MLP, XGB, CatBoost)
├── evaluation.py            # Performance metrics and basic plotting
├── statistics.py            # NRI, IDI, and DeLong test calculations
└── shap_analysis.py         # SHAP interpretability analysis
```

---

## 🚀 How to Use

1. **Prepare Data**  
   Ensure the following files are in the same directory:
   - `traindata.CSV` : Training dataset
   - `testdata.CSV`  : Testing dataset  
   (Ensure column names match the feature definitions in the scripts)

2. **Install Dependencies**
   ```bash
   pip install pandas scikit-learn xgboost catboost matplotlib shap
   ```

3. **Run the Pipeline**
   ```bash
   python main.py
   ```

4. **Select Feature Set**
   In `main.py`, choose between:
   ```python
   feature_set = pre_features    # Preoperative model
   # feature_set = full_features # Preoperative + Intraoperative model
   ```

---

## ⚡ Supported Models
- Logistic Regression (LR)
- Regularized Logistic Regression (L2 penalty)
- Multi-Layer Perceptron (MLP)
- XGBoost
- CatBoost
- Deep Neural Network (DNN) [Extendable]

---

## 🎯 Output Results
- Metrics: AUROC, AUPRC, F1-score, Sensitivity, Specificity
- Visualizations: ROC curve, PR curve, Calibration curve, Decision curve, Nomogram
- Statistical Tests: Net Reclassification Improvement (NRI), Integrated Discrimination Improvement (IDI), DeLong test
- Interpretability: SHAP feature importance analysis

---

## 📊 Future Work
- Enhance DNN architectures
- Implement hyperparameter tuning
- Explore ensemble methods
- Integrate with clinical web-based tools

---

## 📞 Contact
For questions or collaboration opportunities, please contact the project lead.
E-mail: lenhartkoo@foxmail.com
