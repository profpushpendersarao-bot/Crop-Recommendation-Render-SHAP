# Crop Recommendation System — Render Deployment

This Flask application is based on the uploaded `crop-recommendation-system.ipynb` and `Crop_recommendation.csv`.

## Dataset and features
- 2,200 records, 8 columns
- 7 input features: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`
- Target: `label` (22 crop classes)
- Dataset check in the notebook: no missing values and no duplicate rows

## ML workflow
- Stratified 80:20 train/test split, `random_state=42`
- Five-fold `StratifiedKFold` cross-validation on training data
- Models compared: Logistic Regression, KNN, Decision Tree, Gaussian Naive Bayes, Random Forest
- StandardScaler is included in the Logistic Regression and KNN pipelines
- Final deployed algorithm: Random Forest Classifier (`n_estimators=100`, `random_state=42`), matching the notebook's final-model choice
- The final Random Forest is refitted on all 2,200 rows after hold-out evaluation
- Web app returns the predicted crop and the three highest class-probability estimates
- SHAP TreeExplainer gives local, class-specific feature contributions for each prediction
- Global SHAP importance is mean absolute SHAP aggregated over a reproducible held-out sample and all crop classes

## Run locally (Windows PowerShell)
```powershell
py -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:5000`.

## Retrain
Keep `Crop_recommendation.csv` beside `train_model.py`, then run:
```powershell
python train_model.py
```
The script prints 5-fold CV accuracy and hold-out accuracy for each model, calculates global SHAP feature importance, and saves `crop_recommendation_model.joblib`, `validation_results.json` and `confusion_matrix.csv`. The app calculates local SHAP contributions for each prediction.

## Deploy on Render
1. Upload all files and the `templates` folder to a GitHub repository.
2. In Render, select **New + > Web Service** and connect the repository.
3. Use the included `render.yaml`, or set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
4. Deploy. Render supplies a public `onrender.com` URL.
5. Visit `/health` to check that the model loaded.

## SHAP interpretation
- Positive SHAP contribution: supports the predicted crop class.
- Negative SHAP contribution: opposes the predicted crop class.
- Absolute magnitude: relative strength of a feature contribution for the current input.
- Global importance: mean absolute SHAP contribution aggregated across classes and held-out examples.
- SHAP describes model behaviour; it does not prove causal effects or guarantee crop suitability.

## Research caution
The supplied notebook reports very high performance on this clean, balanced dataset. The dataset appears to be an idealised benchmark and does not establish real-world farm performance. Validate with independent, geographically diverse field data and local agronomic expertise before operational use. Model probability scores are not guaranteed crop success or yield estimates.
