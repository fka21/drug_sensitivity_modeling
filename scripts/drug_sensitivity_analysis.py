#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:52:03 2024

@author: ferenc.kagan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized script for gene expression analysis and drug sensitivity prediction
"""

# Load required libraries
import os
import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

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



######################################
###                                ###
### Feature preprocessing pipeline ###
###                                ###
######################################

# Filter genes with low variance (genes with near-constant expression can be removed)
variance_threshold = VarianceThreshold(threshold=0.1)

# Scale expression data (Z-score normalization)
scaler = PowerTransformer()

################################
###                          ###
### Setting up the pipelines ###
###                          ###
################################

# Define pipelines for both models (regression)
rf_pipeline = Pipeline([
    ('var_filter', variance_threshold),
    ('scaler', scaler),
    ('regressor', RandomForestRegressor(random_state=7))
])

lasso_pipeline = Pipeline([
    ('var_filter', variance_threshold),
    ('scaler', scaler),
    ('regressor', Lasso())
])

# Hyperparameter grid for both models
param_grid_rf = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 5, 10],
    'regressor__min_samples_split': [2, 5]
}

param_grid_lasso = {
    'regressor__alpha': [0.01, 0.1, 1.0]
}

################################
###                          ###
### Dimensionality reduction ###
###                          ###
################################

# Extract the transformed TPM values from the scaler step
pipeline = Pipeline([
    ('scaler', scaler),
    ('regressor',  RandomForestRegressor(random_state=7))
])


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
grid_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=5, scoring='r2', n_jobs=3)
grid_lasso = GridSearchCV(lasso_pipeline, param_grid_lasso, cv=5, scoring='r2', n_jobs=3)

# Train and evaluate Random Forest
grid_rf.fit(X_train, y_train_reg)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print(f"Random Forest R^2: {r2_score(y_test_reg, y_pred_rf)}")

# Train and evaluate Lasso
grid_lasso.fit(X_train, y_train_reg)
best_lasso = grid_lasso.best_estimator_
y_pred_lasso = best_lasso.predict(X_test)
print(f"Lasso R^2: {r2_score(y_test_reg, y_pred_lasso)}")

#######################
###                 ###
### Classifications ###
###                 ###
#######################

# Define classifier pipeline
rf_clf_pipeline = Pipeline([
    ('var_filter', variance_threshold),
    ('scaler', scaler),
    ('classifier', RandomForestClassifier(random_state=7))
])

# Hyperparameter grid for classifier
param_grid_rf_clf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 5, 10],
    'classifier__min_samples_split': [2, 5]
}

# Run GridSearchCV for hyperparameter tuning
grid_rf_clf = GridSearchCV(rf_clf_pipeline, param_grid_rf_clf, cv=5, scoring='accuracy', n_jobs=3)
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


