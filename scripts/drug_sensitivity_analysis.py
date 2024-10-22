#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:52:03 2024

@author: ferenc.kagan
"""

# Load required libraries
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.pipeline import Pipeline
from scipy.stats import linregress
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

# Setting up working environment
os.chdir("/Users/ferenc.kagan/Documents/Projects/Turbine_hw/")

# Load data
expr = pd.read_csv("input/CCLE_expression.csv")
metadata = pd.read_csv("input/sample_info.csv")
sens = pd.read_excel("input/GDSC2_fitted_dose_response_25Feb20.xlsx")

# Prepare data for downstream analysis
metadata = metadata[metadata['DepMap_ID'].isin(expr['Unnamed: 0'])]
metadata = metadata.set_index('DepMap_ID').reindex(expr['Unnamed: 0']).reset_index()

# Merge datasets and filter for a specific drug
merged_df = expr.merge(metadata, left_on='Unnamed: 0', right_on='Unnamed: 0').merge(
    sens[sens['DRUG_NAME'] == 'Lapatinib'], left_on='Sanger_Model_ID', right_on='SANGER_MODEL_ID')

# Discretize drug sensitivity for classification
threshold = merged_df['LN_IC50'].median()  
merged_df['Drug_Sensitivity'] = ['Resistant' if ln_ic50 > threshold else 'Sensitive' for ln_ic50 in merged_df['LN_IC50']]

# Scaling TPM values (standard practice, especially for regression)
expr_mat = merged_df.iloc[:, 1:expr.shape[1]].values  # Remove first column

# Split data for regression and classification
X = merged_df.iloc[:, 1:expr.shape[1]].values
y_reg = merged_df['LN_IC50'].values  # Regression
y_class = merged_df['Drug_Sensitivity'].values  # Classification

# Train-test split (only once to avoid leakage)
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=7)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_class, test_size=0.2, random_state=7)

#################################
###                           ###
### Exploratory data analysis ###
###                           ###
#################################

# What are the dimensions of the datasets?
print(expr.shape)
print(metadata.shape)
print(sens.shape)

# Have a quick glance at the tables
print(expr.head())
print(metadata.head())
print(sens.head())

# Are there NA terms?
print(expr.isna().sum().sum())

# Are the expressions all numerical?
print(expr.shape[1] - expr.map(np.isreal).all().sum())

# There are missing crossreferences, what are these?
missing_crossrefs = metadata[~metadata['Sanger_Model_ID'].isin(sens['SANGER_MODEL_ID'])]
print(missing_crossrefs)

# Drug sensitivity distribution
plt.figure(figsize=(10, 6))
sns.histplot(merged_df['LN_IC50'], bins=30, kde=True)
plt.xlabel('LN_IC50')
plt.ylabel('Frequency')
plt.title('Distribution of LN_IC50')
plt.axvline(threshold, color='red', linestyle='--', linewidth=1.5, label=f'Median LN_IC50: {threshold:.2f}')
plt.grid(True)
plt.savefig('output/drug_sens.png', dpi=300, bbox_inches='tight')
plt.show()

# Are drug sesnitivity classifications balanced in proportion?
merged_df['Drug_Sensitivity'].value_counts()

######################################
###                                ###
### Feature preprocessing pipeline ###
###                                ###
######################################

# Filter genes with low variance (genes with near-constant expression can be removed)
variance_threshold = VarianceThreshold(threshold=0.1)

# Scale expression data (Z-score normalization)
scaler = RobustScaler()

################################
###                          ###
### Setting up the pipelines ###
###                          ###
################################

## PIPELINE DEFINITIONS
# Pipeline using PCA for dimensionality reduction
rf_pipeline_pca = Pipeline([
    ('pca', PCA()),
    ('regressor', RandomForestRegressor())
])

lasso_pipeline_pca = Pipeline([
    ('pca', PCA()),
    ('regressor', Lasso())
])

# Define pipelines for Gradient Boosting (regression)
gbr_pipeline = Pipeline([
    ('scaler', scaler),
    ('var_filter', variance_threshold),
    ('regressor', GradientBoostingRegressor())
])

# Extract the transformed TPM values from the scaler step
pipeline = Pipeline([
    ('scaler', scaler),
    ('regressor',  RandomForestRegressor())
])


# Define Gradient Boosting Classifier pipeline
gb_clf_pipeline = Pipeline([
    ('var_filter', variance_threshold),
    ('scaler', scaler),
    ('classifier', GradientBoostingClassifier())
])


# Define pipelines for both models (regression)
rf_pipeline = Pipeline([
    ('scaler', scaler),
    ('var_filter', variance_threshold),
    ('regressor', RandomForestRegressor())
])

lasso_pipeline = Pipeline([
    ('scaler', scaler),
    ('var_filter', variance_threshold),
    ('regressor', Lasso())
])

## HYPERPARAMETER SEARCHES
# Hyperparameter grid for models
param_grid_rf = {
    'scaler': [StandardScaler(), MinMaxScaler(), PowerTransformer()],
    'regressor__n_estimators': list(range(10, 200, 20)),
    'regressor__max_depth': [None, 5, 10],
    'regressor__min_samples_split': list(range(2, 20, 2))
}

param_grid_lasso = {
    'regressor__alpha': list(np.arange(0, 1, 0.05)),
    'scaler': [StandardScaler(), MinMaxScaler(), PowerTransformer()] 
}

# Define classifier pipeline
rf_clf_pipeline = Pipeline([
    ('var_filter', variance_threshold),
    ('scaler', scaler),
    ('classifier', RandomForestClassifier())
])

# Hyperparameter grid for classifier
param_grid_rf_clf = {
    'scaler': [StandardScaler(), MinMaxScaler(), PowerTransformer()],
    'classifier__n_estimators': list(range(10, 200, 20)),
    'classifier__max_depth': [None, 5, 10],
    'classifier__min_samples_split': list(range(2, 20, 2))
}

# Hyperparameter grids for PCA pipelines (no feature selection)
param_grid_rf_pca = {
    'regressor__n_estimators': list(range(10, 200, 20)),
    'regressor__max_depth': [None, 5, 10],
    'regressor__min_samples_split': list(range(2, 20, 2)),
    'pca__n_components': list(range(2, 50, 5))
}

param_grid_lasso_pca = {
    'regressor__alpha': list(np.arange(0, 1, 0.05)),
    'pca__n_components': list(range(2, 50, 5))
}


# Hyperparameter grid for Gradient Boosting (regression)
param_grid_gbr = {
    'regressor__n_estimators': list(range(10, 200, 20)),
    'regressor__learning_rate': list(np.arange(0.01, 0.5, 0.05)),
    'regressor__max_depth': [3, 5, 10]
}


# Hyperparameter grid for Gradient Boosting Classifier
param_grid_gb_clf = {
    'classifier__n_estimators': list(range(10, 200, 20)),
    'classifier__learning_rate': list(np.arange(0.01, 0.5, 0.05)),
    'classifier__max_depth': [3, 5, 10]
}

################################
###                          ###
### Dimensionality reduction ###
###                          ###
################################


pipeline.fit(X, y_reg)
# Now apply the scaling to the filtered data
X_scaled = rf_pipeline.named_steps['scaler'].transform(X)

# Dimensionality reduction using UMAP
n_neighbors = 30  
min_dist = 0.1  
n_components = 2  
metric = 'euclidean'

# Initialize UMAP
umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric, random_state=7)

# Apply UMAP on the scaled data
expr_umap = umap_model.fit_transform(X_scaled)

# Plotting UMAP
fig, ax = plt.subplots(figsize=(12, 8))
umap_plot = sns.scatterplot(x=expr_umap[:, 0], y=expr_umap[:, 1], hue=merged_df['primary_disease'], ax=ax, legend='full')
ax.set_title('UMAP of Expression Data')
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.get_legend().remove()

# Add legend separately
handles, labels = umap_plot.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1))
plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.savefig('output/umap.png', dpi=300, bbox_inches='tight')
plt.show()


###################
###             ###
### Regressions ###
###             ###
###################

# Run GridSearchCV for hyperparameter tuning
grid_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=5, scoring='r2', n_jobs=3, verbose = 10)
grid_lasso = GridSearchCV(lasso_pipeline, param_grid_lasso, cv=5, scoring='r2', n_jobs=3, verbose = 10)

# Train and evaluate Random Forest
grid_rf.fit(X_train, y_train_reg)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
rf_r2 = r2_score(y_test_reg, y_pred_rf)
rf_mse = mean_squared_error(y_test_reg, y_pred_rf)
print(f"Random Forest R^2: {rf_r2}")
print(f"Random Forest MSE: {rf_mse}")

# Save the Random Forest model
joblib.dump(best_rf, "output/models/random_forest_model.pkl", compress = 1)

# Train and evaluate Lasso
grid_lasso.fit(X_train, y_train_reg)
best_lasso = grid_lasso.best_estimator_
y_pred_lasso = best_lasso.predict(X_test)
lasso_r2 = r2_score(y_test_reg, y_pred_lasso)
lasso_mse = mean_squared_error(y_test_reg, y_pred_lasso)
print(f"Lasso R^2: {lasso_r2}")
print(f"Lasso MSE: {lasso_mse}")

# Save the Lasso model
joblib.dump(best_lasso, "output/models/lasso_model.pkl", compress = 1)

# Create plots: Random Forest Predicted vs Actual
slope_rf, intercept_rf, r_value_rf, p_value_rf, std_err_rf = linregress(y_test_reg, y_pred_rf)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_rf)
plt.plot(y_test_reg, intercept_rf + slope_rf * y_test_reg, 'r')
plt.title('Random Forest Predictions vs Actual')
plt.xlabel("Actual ln(IC50)")
plt.ylabel("Predicted ln(IC50)")
plt.text(0.05, 0.85, f'R-squared: {rf_r2:.2f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.80, f'MSE: {rf_mse:.2f}', transform=plt.gca().transAxes)
plt.savefig("output/rf_pred-vs-actual.png", dpi=300, bbox_inches='tight')
plt.show()

# Create plots: Lasso Predicted vs Actual
slope_lasso, intercept_lasso, r_value_lasso, p_value_lasso, std_err_lasso = linregress(y_test_reg, y_pred_lasso)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_lasso)
plt.plot(y_test_reg, intercept_lasso + slope_lasso * y_test_reg, 'r')
plt.title('Lasso Predictions vs Actual')
plt.xlabel("Actual ln(IC50)")
plt.ylabel("Predicted ln(IC50)")
plt.text(0.05, 0.85, f'R-squared: {lasso_r2:.2f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.80, f'MSE: {lasso_mse:.2f}', transform=plt.gca().transAxes)
plt.savefig("output/lasso_pred-vs-actual.png", dpi=300, bbox_inches='tight')
plt.show()


#######################
###                 ###
### Classifications ###
###                 ###
#######################

# Run GridSearchCV for hyperparameter tuning
grid_rf_clf = GridSearchCV(rf_clf_pipeline, param_grid_rf_clf, cv=5, scoring='accuracy', n_jobs=3, verbose = 10)
grid_rf_clf.fit(X_train_clf, y_train_clf)

# Evaluate classification model
best_rf_clf = grid_rf_clf.best_estimator_
y_pred_clf = best_rf_clf.predict(X_test_clf)
y_prob_clf = best_rf_clf.predict_proba(X_test_clf)[:, 1]

# Performance metrics for classification
print(f"Accuracy: {accuracy_score(y_test_clf, y_pred_clf)}")
print(f"Precision: {precision_score(y_test_clf, y_pred_clf, pos_label='Sensitive')}")
print(f"Recall: {recall_score(y_test_clf, y_pred_clf, pos_label='Sensitive')}")
print(f"ROC AUC: {roc_auc_score(y_test_clf, y_prob_clf)}")

# Save the Random Forest model
joblib.dump(best_rf_clf, "output/models/random_forest_clf_model.pkl", compress = 1)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test_clf, y_prob_clf, pos_label='Sensitive')
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc_score(y_test_clf, y_prob_clf):.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig("output/roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()


##################################
###                            ###
### Using principal components ###
###                            ###
##################################


# Principal Component Analysis (PCA)
pca = PCA(n_components=10)  # Choose the number of components you want to retain
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# Run GridSearchCV for hyperparameter tuning (PCA pipelines)
grid_rf_pca = GridSearchCV(rf_pipeline_pca, param_grid_rf_pca, cv=5, scoring='r2', n_jobs=3, verbose = 10)
grid_lasso_pca = GridSearchCV(lasso_pipeline_pca, param_grid_lasso_pca, cv=5, scoring='r2', n_jobs=3, verbose = 10)

# Evaluate Random Forest with PCA
grid_rf_pca.fit(X_train_pca, y_train_reg)
best_rf_pca = grid_rf_pca.best_estimator_
y_pred_rf_pca = best_rf_pca.predict(X_test_pca)
rf_r2_pca = r2_score(y_test_reg, y_pred_rf_pca)
print(f"Random Forest with PCA R^2: {rf_r2_pca}")

# Evaluate Lasso with PCA
grid_lasso_pca.fit(X_train_pca, y_train_reg)
best_lasso_pca = grid_lasso_pca.best_estimator_
y_pred_lasso_pca = best_lasso_pca.predict(X_test_pca)
lasso_r2_pca = r2_score(y_test_reg, y_pred_lasso_pca)
print(f"Lasso with PCA R^2: {lasso_r2_pca}")


# Create plots: Random Forest Predicted vs Actual
slope_rf, intercept_rf, r_value_rf, p_value_rf, std_err_rf = linregress(y_test_reg, y_pred_rf_pca)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_rf_pca)
plt.plot(y_test_reg, intercept_rf + slope_rf * y_test_reg, 'r')
plt.title('Random Forest Predictions vs Actual')
plt.xlabel("Actual ln(IC50)")
plt.ylabel("Predicted ln(IC50)")
plt.text(0.05, 0.85, f'R-squared: {rf_r2_pca:.2f}', transform=plt.gca().transAxes)
plt.savefig("output/rf_pred-vs-actual.png", dpi=300, bbox_inches='tight')
plt.show()

# Create plots: Lasso Predicted vs Actual
slope_lasso, intercept_lasso_pca, r_value_lasso, p_value_lasso, std_err_lasso = linregress(y_test_reg, y_pred_lasso_pca)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_lasso_pca)
plt.plot(y_test_reg, intercept_lasso_pca + slope_lasso * y_test_reg, 'r')
plt.title('Lasso Predictions vs Actual')
plt.xlabel("Actual ln(IC50)")
plt.ylabel("Predicted ln(IC50)")
plt.text(0.05, 0.85, f'R-squared: {lasso_r2_pca:.2f}', transform=plt.gca().transAxes)
plt.savefig("output/lasso_pred-vs-actual.png", dpi=300, bbox_inches='tight')
plt.show()


#########################
###                   ###
### Gradient boosting ###
###                   ###
#########################


# Run GridSearchCV for hyperparameter tuning (Gradient Boosting Regressor)
grid_gbr = GridSearchCV(gbr_pipeline, param_grid_gbr, cv=5, scoring='r2', n_jobs=3, verbose = 10)
grid_gbr.fit(X_train, y_train_reg)
best_gbr = grid_gbr.best_estimator_
y_pred_gbr = best_gbr.predict(X_test)

# Evaluate Gradient Boosting Regressor
gbr_r2 = r2_score(y_test_reg, y_pred_gbr)
gbr_mse = mean_squared_error(y_test_reg, y_pred_gbr)
print(f"Gradient Boosting Regressor R^2: {gbr_r2}")
print(f"Gradient Boosting Regressor MSE: {gbr_mse}")

# Save the Gradient Boosting Regressor model
joblib.dump(best_gbr, "output/models/gradient_boosting_regressor_model.pkl", compress=1)

# Create plots: Gradient Boosting Regressor Predicted vs Actual
slope_gbr, intercept_gbr, r_value_gbr, p_value_gbr, std_err_gbr = linregress(y_test_reg, y_pred_gbr)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_gbr)
plt.plot(y_test_reg, intercept_gbr + slope_gbr * y_test_reg, 'r')
plt.title('Gradient Boosting Regressor Predictions vs Actual')
plt.xlabel("Actual ln(IC50)")
plt.ylabel("Predicted ln(IC50)")
plt.text(0.05, 0.85, f'R-squared: {gbr_r2:.2f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.80, f'MSE: {gbr_mse:.2f}', transform=plt.gca().transAxes)
plt.savefig("output/gbr_pred-vs-actual.png", dpi=300, bbox_inches='tight')
plt.show()

####################################
### Gradient Boosting Classifier ###
####################################


# Run GridSearchCV for hyperparameter tuning
grid_gb_clf = GridSearchCV(gb_clf_pipeline, param_grid_gb_clf, cv=5, scoring='accuracy', n_jobs=3, verbose = 10)
grid_gb_clf.fit(X_train_clf, y_train_clf)

# Evaluate Gradient Boosting Classifier
best_gb_clf = grid_gb_clf.best_estimator_
y_pred_gb_clf = best_gb_clf.predict(X_test_clf)
y_prob_gb_clf = best_gb_clf.predict_proba(X_test_clf)[:, 1]

# Performance metrics for classification
print(f"Gradient Boosting Classifier Accuracy: {accuracy_score(y_test_clf, y_pred_gb_clf)}")
print(f"Gradient Boosting Classifier Precision: {precision_score(y_test_clf, y_pred_gb_clf, pos_label='Sensitive')}")
print(f"Gradient Boosting Classifier Recall: {recall_score(y_test_clf, y_pred_gb_clf, pos_label='Sensitive')}")
print(f"Gradient Boosting Classifier ROC AUC: {roc_auc_score(y_test_clf, y_prob_gb_clf)}")

# Save the Gradient Boosting Classifier model
joblib.dump(best_gb_clf, "output/models/gradient_boosting_classifier_model.pkl", compress=1)

# Plot ROC curve for Gradient Boosting Classifier
fpr_gb, tpr_gb, _ = roc_curve(y_test_clf, y_prob_gb_clf, pos_label='Sensitive')
plt.plot(fpr_gb, tpr_gb, color='green', lw=2, label=f'ROC curve (area = {roc_auc_score(y_test_clf, y_prob_gb_clf):.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig("output/gb_roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()


#########################
###                   ###
### Feature selection ###
###                   ###
#########################

## Using the GBM Classifier as it has the best score of the tested approaches
## Will use SelectFromModel to retireve influential genes in classifying drug sensitivity

# Use the trained classifier to select features
selector = SelectFromModel(best_gb_clf.named_steps['classifier'], threshold='mean')

# Fit the selector on the training data
selector.fit(X_train_clf, y_train_clf)

# Get selected features
selected_features_model = pd.Series(merged_df.iloc[:, 1:expr.shape[1]].iloc[:, selector.get_support()].columns)

print(f"Selected {len(selected_features_model)} features using model-based selection:")
print(selected_features_model)

# Extract the numbers in parentheses
pattern = r'\((\d+)\)'
extracted_numbers = selected_features_model.str.extract(pattern, expand = False).dropna().astype(int).tolist()

# Convert to DataFrame
gene_df = pd.DataFrame(selected_features_model, columns = ["GeneID"])
gene_id_df = pd.DataFrame(extracted_numbers)

# Save to a tab-separated file
gene_df.to_csv("output/selected_genes.tsv", sep='\t', index = False, header = False)
gene_id_df.to_csv("output/selected_gene_ids.tsv", sep = "\t", index = False, header = False)
