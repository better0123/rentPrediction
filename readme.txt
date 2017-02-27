We modelled the dataset using two linear models (ridge and Lasso). R square (1) is used as the metric to evaluate our models (since is a regression task, meaning the dependent variable is continuous, R square is a reasonable metric). During the model training stage, we use grid search and corss validation to select the best hyperparameter for each model. For the final model selection, we choose the model with the highest R square.     

The details of the data processing, feature selection and modelling are described below:   

process data:    
1. manually identify binary, categorical(binary excluded) and numerical features from Census's https://www.census.gov/housing/nychvs/data/2014/occ_14_long.pdf documentation; record the range of each variables' meaningful value for further missing data (NaN) interpretation  
2. replace native-encoded missing value with np.nan  
3. split original dataset into train and test dataset   
4. impute training dataset's missing value (use "most_frequent" strategy for categorical data; "mean"/"median" strategy for numerical data)  
5. impute test dataset's missing value using information from train data  
6. perform oneHot on train and test dataset's categorical data (binary features are left out of this processto avoid multicolinearity)   

feature engineering:
* tried polynomial features with degree 2 on 537 features, raise memory error on the machine with 8G RAM, hence being discarded. 

feature selection:  
7. apply standard scaler to the train dataset 
8. LASSO based feature selection 
9. remove variables with zero/below median LASSO coeficients 

modelling:  
10. model with Ridge Regression using grid search after scaling (pipeline fashion)
11. model with Lasso 
  
result:  
12. Ridge regression (alpha=110): 0.599 
    Lasso (alpha=0.01425): 0.600
The final model is chosen as Lasso with alpha=0.01425 (use median to impute numerical data, most_frequent to impute categorical data, and features only have non-zero Lasso coefficients during feature selections) 

13. feature engineering comparison: using "median" to impute numerical data yields averaged 0.1% lower R square than using "mean"
14. feature selection comparison: remove features that have zero LASSO coeficients yeilds averaged 0.15% higher R square than using "remove when below median".    

(1) a number that indicates the proportion of the variance in the dependent variable that is predictable from the independent variable
