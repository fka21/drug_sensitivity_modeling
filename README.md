# Drug sensitivity modeling
This repo contains scripts which I wrote to build predictive models on cancer cell line drug sensitivities. This was part of a homework I had. The tasks with the required data were provided _a priori_.

## Aim
The tasks are described in the `input/comp-bio-homework-description.pdf` file.
  1. Given the data perform a dimensionality reduction with a preferred method and viusalize it.
  2. Using gene expression data identify biomarkers for Lapatinib sensitivity.
  3. Build a predictive model for Lapatinib sensitivity and evaluate it's performance.
> [!NOTE]  
> The created predictive model is by no means exhaustive and was not optimized.

## Summary

  **Task 1**
  * I decided to use principal component analysis (PCA). 
  * I retrieved only the first 10 principal components as from PC5 the explained variability does not change significantly (see: `output/pca.png`)
  * The samples are colored according to the cancer type, some degree of separation of the samples is noticeable on the PC1.

  **Task 2**
  * As the gene expression provided was TPM transformed and only cancer cell line samples were provided, there was no possibility of differential gene expression analysis. Instead, I used a feature selection approach.
  * I used two regression approaches: *random forest* and *lasso regression*. The output of these approaches was overlapped and only commonly selected features were kept.
  * A total of **60 genes** were selected as influential for Lapatinib sensitivity of cancer cells (see: `output/selected_genes.tsv` and `output/selected_gene_ids.tsv`).
  * A systematic approach was also performed to get an idea of what functions are affected by the drug treatment. The analysis resulted in only a single enriched function, i.e. **regulation of blood pressure** (see: `output/singificant_terms.xlsx`). This could make sense as the RTK is involved in metabolism, growth and differentiation in cancers and blood supply of tumors could be essential.

  **Task 3**
  * To evaluate the model I used a simple train/test split approach. I retained 20% of the samples as testing data.
  * Firstly I used a *random forest regression* approach. I also implemented a short optimization step for the model fitting to try to improve the evaluation metrics. The model performance was lackluster with a **R2 = 0.36** and **MSE = 1.64** (see: `output/rf_pred-vs-actual.png`).
  * Secondly, I used the *lasso regression* in order to get better predictions. I used similar approaches as above, without the optimization step. The overall model predictions showed improvement (see: `output/lasso_pred-vs-actual.png`) with an **R2 = 0.56** and a **MSE = 1.13**.
  * Lastly I tried to use a classifier. As a classifier I chose the *random forest approach*. Similar steps were used as above to optimize the model fit and evaluate its performance. The classified performed somewhat poorly (see: `output/roc_curve.png`) with a ROC **AUC = 0.77**. Accuracy of model is 0.68 and precision is 0.68 with a recall of 0.59.
