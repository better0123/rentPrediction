## Applied Machine Learning -- Rent Prediction
### Emily Hua, Ming Zhou
We modelled the dataset using three linear models (ridge, linear SVM and Lasso). R Square<sup>[1](#fn_r2)</sup> is used as the metric to evaluate our models (since is a regression task, meaning the dependent variable is continuous, R Square is a reasonable metric). During the model training stage, we used grid search and cross validation to select the best hyper-parameter for each model. For the final model selection, we choose the model with the highest R Square. The best model is ridge regression with alpha-110 and it reaches 0.592 R Square on the test dataset. 

### The details of the data processing, feature selection and modelling are described as following:
#### Process data:
1. Manually identify binary, categorical(binary excluded) and numerical features from [Census's documentation](https://www.census.gov/housing/nychvs/data/2014/occ_14_long.pdf); record the range of each variables' meaningful value for further missing data (NaN) interpretation; identify and remove features that directly associated with rent and can't be used in modelling  
2. Replace native-encoded missing value with `np.nan`;
3. Split original dataset into `train` and `test` datasets;
4. Impute training dataset's missing value (use "most_frequent" strategy for categorical data; "mean"/"median" strategy for numerical data);
5. Impute test dataset's missing value using information from training data;
6. Perform `oneHot` on training and testing datasets' categorical data (binary features are left out of this process to avoid multi-colinearity).

### Feature engineering:
Generating polynomial features with degree 2 on 537 features raised `MemoryError` on a machine with 8G RAM, hence it was discarded.

### Feature selection:
1. Apply standard scaler to the training dataset;
2. Select features using LASSO;
3. Remove variables with zero or below median LASSO coefficients.

### Modelling:
1. Model with Ridge Regression 
2. Model with LASSO;  
(using grid search after scaling (pipeline fashion))

### Result:
* Ridge regression (with `alpha = 110`) results in an R2 of 0.611 (averaged CV R square)
* LASSO (with `alpha = 0.01425`) results in an R2 of 0.608 (averaged CV R square)   

The final model is chosen as ridge with alpha=110 (built using median to impute numerical data, most_frequent to impute categorical data, and features only have non-zero Lasso coefficients during feature selections, which results in the final 238 features)
#### The test result reaches 0.592 of R Square!

* Feature engineering comparison: using "median" to impute numerical data yields averaged 0.1% lower R Square than using "mean";
* Feature selection comparison: remove features that have zero LASSO coefficients yields an average 0.15% higher R Square than using "remove when below median".

detailed model-train is in modelling.ipynb (for in-line output see modelling_unformatted.ipynb)  

<a name="fn_r2">(1)</a>: A number that indicates the proportion of the variance in the dependent variable that is predictable from the independent variable
