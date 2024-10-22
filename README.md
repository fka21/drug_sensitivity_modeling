# Drug sensitivity modeling
This repo contains scripts which I wrote to build predictive models on cancer cell line drug sensitivities. This was part of a homework I had. The tasks with the required data were provided _a priori_.

## Aim
The tasks are described in the `input/comp-bio-homework-description.pdf` file.
  1. Given the data perform a dimensionality reduction with a preferred method and viusalize it.
  2. Using gene expression data identify biomarkers for Lapatinib sensitivity.
  3. Build a predictive model for Lapatinib sensitivity and evaluate it's performance.
> [!NOTE]  
> The created predictive model is by no means exhaustive.

## Summary

  **Task 1**
  * I decided to use both principal component analysis (PCA) and Uniform Manifold Approximation and Projection (UMAP). 
  * I retrieved only the first 10 principal components as from PC5 the explained variability does not change significantly (see: `output/pca.png`)
  * The samples are colored according to the cancer type, some degree of separation of the samples is noticeable on the PC1.
  * In the UMAP the blood cancer cell lines show a distinct separation from the rest, indicating some signature expression profile for these cancer types.

  **Task 2**
  * As the gene expression provided was TPM (transcripts-per-million) transformed and only cancer cell line samples were provided, there was no possibility of differential gene expression analysis. Instead, I used a feature selection approach.
  * I used three regression approaches: *random forest*, *lasso regression*, and *gradient boosting*. Next to this I also binarized the drug sensitivity into *sensitive* and *resistant* using the median of drug sensitivity as a cut-off. This was translated then into a classifying problem for which I used 3 approaches: *random forest*, *lasso regression*, and *gradient boosting*. Models were wrapped into Pipeline and a hyperparameter search was performed using an exhaustive GridSearchCV.  
  * A total of **318 genes** were selected as influential for Lapatinib sensitivity of cancer cells (see: `output/selected_genes.tsv` and `output/selected_gene_ids.tsv`).
  * A systematic approach was also performed to get an idea of what functions are affected by the drug treatment. The analysis resulted in no enriched gene ontological terms.
  * 
  **Task 3**
  * To evaluate the model I used a simple train/test split approach. I retained 20% of the samples as testing data.
  * Firstly I used a *random forest regression* approach. I also implemented a short optimization step for the model fitting to try to improve the evaluation metrics. The model performance was lackluster (see: `output/rf_pred-vs-actual.png`).
  * Secondly, I used the *lasso regression* in order to get better predictions. I used similar approaches as above. The overall model predictions showed slight improvement (see: `output/lasso_pred-vs-actual.png`).
  * Thirdly, I used the *gradient boosting regression* with a further increase in performance (see: `output/gbr_pred-vs-actual.png`).
  * Lastly I tried to convert the problem into a classification. As a classifier I chose the *random forest approach* and *gradient boosting classifier*. Similar steps were used as above to optimize the model fit and evaluate its performance. The classified performed somewhat poorly (see: `output/roc_curve.png` and `output/gb_roc_curve.png`)
