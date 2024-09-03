#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drug Sensitivity Analysis using PCA, Feature Selection, and Machine Learning Models

Created on Fri Jun 28 10:28:35 2024

@author: ferenc.kagan
"""

# Import required libraries
import openpyxl
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    explained_variance_score, accuracy_score,
    precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import linregress

# Set the working directory
os.chdir("/Users/ferenc.kagan/Documents/Projects/hw/")

########################
##### READ IN DATA #####
########################

expr = pd.read_csv("input/CCLE_expression.csv")
metadata = pd.read_csv("input/sample_info.csv")
sens = pd.read_excel("input/GDSC2_fitted_dose_response_25Feb20.xlsx")

#############
#### EDA ####
#############

# Display dimensions of the datasets
print("Expression data shape:", expr.shape)
print("Metadata shape:", metadata.shape)
print("Sensitivity data shape:", sens.shape)

# Display the first few rows of each dataset
print("Expression data head:\n", expr.head())
print("Metadata head:\n", metadata.head())
print("Sensitivity data head:\n", sens.head())

# Check for missing values in the expression data
print("Number of NA values in expression data:", expr.isna().sum().sum())

# Ensure all expressions are numerical
print("Number of non-numerical entries in expression data:", expr.shape[1] - expr.applymap(np.isreal).all().sum())

# Identify missing cross-references in metadata
missing_crossrefs = metadata[~metadata['Sanger_Model_ID'].isin(sens['SANGER_MODEL_ID'])]
print("Missing cross-references:\n", missing_crossrefs)

# Filter and merge data for analysis
metadata = metadata[metadata['DepMap_ID'].isin(expr['Unnamed: 0'])]
metadata = metadata.set_index('DepMap_ID').reindex(expr['Unnamed: 0']).reset_index()

merged_df = expr.merge(metadata, left_on='Unnamed: 0', right_on='Unnamed: 0') \
                .merge(sens[sens['DRUG_NAME'] == 'Lapatinib'], left_on='Sanger_Model_ID', right_on='SANGER_MODEL_ID')

# Extract the first column (DepMap_ID)
first_column = expr.iloc[:, 0]

# Discretize drug sensitivity for classification
threshold = merged_df['LN_IC50'].median()
merged_df['Drug_Sensitivity'] = ['Resistant' if ln_ic50 > threshold else 'Sensitive' for ln_ic50 in merged_df['LN_IC50']]

# Plot the distribution of LN_IC50
plt.figure(figsize=(10, 6))
sns.histplot(merged_df['LN_IC50'], bins=30, kde=True)
plt.xlabel('LN_IC50')
plt.ylabel('Frequency')
plt.title('Distribution of LN_IC50')
plt.axvline(threshold, color='red', linestyle='--', linewidth=1.5, label=f'Median LN_IC50: {threshold:.2f}')
plt.grid(True)
plt.legend()
plt.savefig('output/drug_sens.png', dpi=300, bbox_inches='tight')
plt.show()

####################################
##### DIMENSIONALITY REDUCTION #####
####################################

# Extract gene expression features
expr_mat = expr.iloc[:, 1:].values

# Run PCA
n_components = 10
pca = PCA(n_components=n_components)
expr_pca = pca.fit_transform(expr_mat)

# Plot Scree Plot and PCA results
fig, axes = plt.subplots(2, 1, figsize=(12, 12))
components = range(1, n_components + 1)
axes[0].plot(components, pca.explained_variance_ratio_, marker='o')
axes[0].set_title('Scree Plot')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')

pca_plot = sns.scatterplot(x=expr_pca[:, 0], y=expr_pca[:, 1], hue=metadata['primary_disease'], ax=axes[1], legend='full')
axes[1].set_title('PCA of Expression Data')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
handles, labels = pca_plot.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.savefig('output/pca.png', dpi=300, bbox_inches='tight')
plt.show()

#############################
##### FEATURE SELECTION #####
#############################

# Set seed for reproducibility
np.random.seed(7)

# Prepare data for feature selection
X = merged_df.iloc[:, 1:expr.shape[1]].values
y = merged_df['LN_IC50'].values  # Regression target
y_class = merged_df['Drug_Sensitivity'].values  # Classification target

# RandomForest feature selection
rf = RandomForestRegressor(random_state=7)
rf.fit(X, y)
sel_rf = SelectFromModel(rf, prefit=True)
sel_rf_idx = sel_rf.get_support(indices=True)

# Lasso regression feature selection
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
sel_lasso = SelectFromModel(lasso, prefit=True)
sel_lasso_idx = sel_lasso.get_support(indices=True)

# Find overlapping features
rf_features = set(sel_rf_idx)
lasso_features = set(sel_lasso_idx)
common_features = rf_features.intersection(lasso_features)
print("Number of common features selected:", len(common_features))

# Subset expression data for selected features
X = X[:, list(common_features)]

# Save selected gene IDs
selected_gene_ids = [expr.columns[i] for i in common_features]
gene_df = pd.DataFrame(selected_gene_ids, columns=["GeneID"])
gene_df.to_csv("output/selected_genes.tsv", sep='\t', index=False, header=False)

# Extract numerical IDs if applicable
pattern = r'\((\d+)\)'
extracted_numbers = pd.Series(selected_gene_ids).str.extract(pattern, expand=False).dropna().astype(int).tolist()
pd.DataFrame(extracted_numbers).to_csv("output/selected_gene_ids.tsv", sep='\t', index=False, header=False)

#####################################
###### RANDOM FOREST APPROACH #######
#####################################

# Train/test split for regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# RandomForest Regressor with GridSearchCV
rf = RandomForestRegressor(random_state=7)
param_grid = {'n_estimators': [100, 150, 200], 'max_depth': [None, 2, 5, 10], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)

# Evaluate model performance
r2 = r2_score(y_test, y_pred_rf)
mse = mean_squared_error(y_test, y_pred_rf)
mae = mean_absolute_error(y_test, y_pred_rf)
evs = explained_variance_score(y_test, y_pred_rf)
print("Random Forest Regression Performance:", r2, mse, mae, evs)

# Plot predictions vs actual values
slope, intercept, r_value, p_value, std_err = linregress(y_test, y_pred_rf)
plt.scatter(y_test, y_pred_rf)
plt.plot(y_test, intercept + slope * y_test, 'r')
plt.title('Random Forest Predictions vs Actual')
plt.xlabel("Actual ln(IC50)")
plt.ylabel("Predicted ln(IC50)")
plt.text(0.05, 0.85, f'R-squared: {r2:.2f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.80, f'MSE: {mse:.2f}', transform=plt.gca().transAxes)
plt.savefig("output/rf_pred-vs-actual.png", dpi=300, bbox_inches='tight')
plt.show()

#####################################
##### LASSO REGRESSION APPROACH #####
#####################################

# Train/test split for Lasso regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# Evaluate Lasso regression performance
r2 = r2_score(y_test, y_pred_lasso)
mse = mean_squared_error(y_test, y_pred_lasso)
mae = mean_absolute_error(y_test, y_pred_lasso)
evs = explained_variance_score(y_test, y_pred_lasso)
print("Lasso Regression Performance:", r2, mse, mae, evs)

# Plot Lasso predictions vs actual values
slope, intercept, r_value, p_value, std_err = linregress(y_test, y_pred_lasso)
plt.scatter(y_test, y_pred_lasso)
plt.plot(y_test, intercept + slope * y_test, 'r')
plt.title('Lasso Regression Predictions vs Actual')
plt.xlabel("Actual ln(IC50)")
plt.ylabel("Predicted ln(IC50)")
plt.text(0.05, 0.85, f'R-squared: {r2:.2f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.80, f'MSE: {mse:.2f}', transform=plt.gca().transAxes)
plt.savefig("output/lasso_pred-vs-actual.png", dpi=300, bbox_inches='tight')
plt.show()

################################################
##### CLASSIFICATION USING RANDOM FOREST #######
################################################

# Train/test split for classification
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=7)

# RandomForest Classifier with GridSearchCV
rf_clf = RandomForestClassifier(random_state=7)
param_grid_clf = {'n_estimators': [100, 150, 200], 'max_depth': [None, 2, 5, 10], 'min_samples_split': [2, 5, 10]}
grid_search_clf = GridSearchCV(rf_clf, param_grid_clf, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_clf.fit(X_train, y_train)
best_rf_clf = grid_search_clf.best_estimator_
best_rf_clf.fit(X_train, y_train)
y_pred_rf_clf = best_rf_clf.predict(X_test)

# Evaluate classification performance
accuracy = accuracy_score(y_test, y_pred_rf_clf)
precision = precision_score(y_test, y_pred_rf_clf, pos_label='Sensitive')
recall = recall_score(y_test, y_pred_rf_clf, pos_label='Sensitive')
roc_auc = roc_auc_score(y_test, best_rf_clf.predict_proba(X_test)[:, 1])
print("Random Forest Classification Performance:", accuracy, precision, recall, roc_auc)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, best_rf_clf.predict_proba(X_test)[:, 1], pos_label='Sensitive')
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.savefig("output/rf_roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()

