import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os


# Function to tune models
def tune_models(
    model_params,
    preprocessor,
    X_train,
    y_train,
    n_iter=50,
    cv=5,
    n_jobs=15,
    random_state=17,
):
    """
    Perform hyperparameter tuning using RandomizedSearchCV for multiple models.

    Parameters:
    - model_params (dict): Dictionary containing model names as keys and a dictionary with
      'model' (estimator) and 'params' (hyperparameter grid) as values.
    - preprocessor: Preprocessing pipeline to apply before model training.
    - X_train: Training features.
    - y_train: Training target.
    - n_iter (int): Number of iterations for RandomizedSearchCV.
    - cv (int): Number of cross-validation folds.
    - n_jobs (int): Number of parallel jobs.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - best_models (dict): Dictionary of best estimators for each model.
    - mse_results (dict): Dictionary of best MSE scores for each model.
    """
    best_models = {}
    mse_results = {}

    for name, config in model_params.items():
        model = config["model"]
        param_dist = prefix_params(name, config["params"])  # Apply prefix to parameters

        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="neg_mean_squared_error",
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        search.fit(X_train, y_train)

        best_models[name] = search.best_estimator_
        best_mse = -search.best_score_  # Convert to positive MSE
        mse_results[name] = best_mse  # Store results for plotting

        print(
            f"{name}: Best MSE = {best_mse:.4f}, Best Params: {search.best_params_}",
            flush=True,
        )

    return best_models, mse_results


# Function to evaluate tuned models
def evaluate_tuned_models(
    best_models,
    mse_results,
    X_train,
    y_train,
    X_test,
    y_test,
    strategy,
    save_path,
):
    """
    Evaluates tuned models using various regression metrics and saves the results to a CSV file.

    Parameters:
    - best_models (dict): Dictionary of best estimators for each model (output of tune_models).
    - mse_results (dict): Dictionary of best MSE scores for each model (output of tune_models).
    - X_train: Training features.
    - y_train: Training target.
    - X_test: Test features.
    - y_test: Test target.
    - strategy (str): A string indicating the strategy used (e.g., "Strategy 1").
    - save_path (str): Desired path for saving the results (CSV file).
    """

    results = []

    for model_name, model in best_models.items():
        # Make predictions on train and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate evaluation metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # Append results to the list
        results.append(
            [
                strategy,
                model_name,
                mse_results[model_name],  # Training MSE (from tuning)
                train_mse,
                test_mse,
                train_r2,
                test_r2,
                train_mae,
                test_mae,
            ]
        )

    # Create a Pandas DataFrame from the results
    df = pd.DataFrame(
        results,
        columns=[
            "Strategy",
            "Model",
            "Tuning MSE",
            "Train MSE",
            "Test MSE",
            "Train R2",
            "Test R2",
            "Train MAE",
            "Test MAE",
        ],
    )

    # Save the DataFrame to a CSV file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    df.to_csv(save_path, index=False)

    print(f"Evaluation results saved to: {save_path}")


# Custom functions
def plot_all_learning_curves(
    models,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_squared_error",
    output_filename="output/faceted_learning_curves.png",
):
    plt.figure(figsize=(15, 15))
    num_models = len(models)
    rows = (num_models + 2) // 3  # 3 plots per row

    for i, (model_name, pipeline) in enumerate(models.items(), 1):
        plt.subplot(rows, 3, i)
        train_sizes, train_scores, test_scores = learning_curve(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=15,
            train_sizes=np.linspace(0.1, 1.0, 10),
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="b",
        )
        plt.plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        plt.plot(
            train_sizes,
            test_scores_mean,
            "o-",
            color="b",
            label="Cross-validation score",
        )
        plt.title(model_name)
        plt.xlabel("Training Examples")
        plt.ylabel("Negative MSE")
        plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Learning curves saved to {output_filename}")


# Function to print out MSE on test data
def evaluate_models(
    models,
    X_train,
    y_train,
    X_test,
    y_test,
    cv=5,
    scoring="neg_mean_squared_error",
    save_plot_path=None,
):
    """
    Evaluate models and generate a horizontal sorted bar plot for Test MSE.

    Parameters:
        models (dict): A dictionary of model names as keys and pipelines as values.
        X_train (array-like): Training features.
        y_train (array-like): Training target.
        X_test (array-like): Test features.
        y_test (array-like): Test target.
        cv (int): Number of cross-validation folds. (Not used in this implementation)
        scoring (str): Scoring metric for evaluation. (Not used in this implementation)
        save_plot_path (str or None): Path to save the bar plot. If None, the plot won't be saved.

    Returns:
        dict: A dictionary of model names and their corresponding test MSE.
    """
    mse_results = {}

    # Evaluate each model
    for model_name, pipeline in models.items():
        # Train the model
        pipeline.fit(X_train, y_train)

        # Compute the test MSE
        test_mse = mean_squared_error(y_test, pipeline.predict(X_test))

        mse_results[model_name] = test_mse
        print(f"{model_name}: Test MSE = {test_mse:.4f}")

    # Plotting the horizontal sorted bar plot
    sorted_mse_results = dict(sorted(mse_results.items(), key=lambda item: item[1]))
    plt.figure(figsize=(10, 6))
    plt.barh(
        list(sorted_mse_results.keys()),
        list(sorted_mse_results.values()),
        color="skyblue",
    )
    plt.xlabel("Test MSE")
    plt.ylabel("Models")
    plt.title("Model Performance Comparison (Test MSE)")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Save the plot if a path is provided
    if save_plot_path:
        plt.savefig(save_plot_path, bbox_inches="tight")
        print(f"Plot saved to {save_plot_path}")

    # Show the plot
    plt.show()

    return mse_results


# Prefix model parameters with "model__" for use in Pipeline
def prefix_params(model_name, param_grid):
    return {f"model__{key}": value for key, value in param_grid.items()}
