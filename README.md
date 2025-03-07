# Building a predictive model for drug sensitivity

## Introduction

This repo contains code I used to build a predictive model for drug sensitivity. 

The repo structure is:
    - `input/` contains tables used as input. These contain gene expression values from various cancer cell lines, associated meta-data, and drug sensitivity data.
    - `output/` contains some of the figures and tables generated during data exploration and training phases.
    - Jupyter Notebooks contain the code used. I used `data_exploration.ipynb` to get an idea for how the received data looks like. The `model_selection.ipynb` contains different training strategies I tried for selecting the method approach and model for the predictive task. This was followed up by `model_optimization_interpretation.ipynb`, where I tried to optimize the hyperparameters of the model using `optuna` and tried to understand the inner workings of the selected model.

## Data description

The Lapatinib drug sensitivity was the focus of this project. Therefore the target variable is `LN_IC50` (it is on logarithmic scale already) for Lapatinib. This drug is targeting lung and breast cancer.

I had to use gene expression data and the rest was free to use. Based on my knowledge I decided to include `age`, `sex`, and `primary_disease`, next to gene expression. 

> [!NOTE]  
> For a more in depth overview of the data please check the `data_exploration.ipynb`.

## Training approach

In all cases I used a train-test split without taking the `primary_disease` segmentation in consideration as I wanted a predictive model which generalizes well. `RandomizedSearchCV` was used for hyperparameter tuning of all models with 5 fold cross validation. 

For all strategies and models learning curves were plotted (`output/faceted_*`), so were the mean squared errors (MSE) from the training process, and the MSE on the test set. Further metrics were also collected in tables (`output/evaluation_*`)

I decided to try several regressor models from all major categories:
    1. Linear Models:
        LinearRegression – A basic linear regression model without regularization.
        Ridge – A linear model with L2 regularization to prevent overfitting.
        Lasso – A linear model with L1 regularization for feature selection.
        ElasticNet – A combination of L1 and L2 regularization.
    2. Decision Tree-Based Models:
        DecisionTreeRegressor – A single decision tree model.
        RandomForestRegressor – An ensemble of decision trees using bagging.
        ExtraTreesRegressor – Similar to Random Forest but with more randomization.
        GradientBoostingRegressor – A boosting-based tree ensemble.
        HistGradientBoostingRegressor – A more efficient, histogram-based gradient boosting.
    3. Ensemble Models:
        AdaBoostRegressor – Boosting model that adjusts weights iteratively.
        BaggingRegressor – Uses bootstrap aggregation for variance reduction.
    4. Support Vector and Nearest Neighbor Models:
        SVR – A Support Vector Machine for regression with a kernel trick.
        KNeighborsRegressor – A non-parametric model based on the k-nearest neighbors.
    5. CatBoostRegressor – A gradient boosting method optimized for categorical data.

Different pre-processing steps were also considered:
    1. Using the gene expression values only, after `VarianceThreshold()` filtering, and `StandardScaler()` transformed.
    2. Feature selection to reduce the feature space using a genetic algorithm, and adding meta-data (`age`, `sex`, `primary_disease`)
    3. Dimensionality reduction (`PCA`) to reduce the feature space, and adding meta-data (`age`, `sex`, `primary_disease`)
    4. Quantile binning of the gene expresison data, and adding meta-data (`age`, `sex`, `primary_disease`)
    5. Dimensionality reduction (`PCA`) to reduce the feature space, and follow it up with a neural network

In the end I picked `ElasticNet` as the model of my choice and used only the gene expression dataset (through regularization it dropped the added meta-data). Using `optuna` I tried to find the best hyperparameters, which resulted in a model with moderate regularization and a mix of L1 and L2 regularization. The test set MSE turned out to be 2.14, which is high. Despite the model reducing inherently the feature space the data seems to be difficult to train on. 