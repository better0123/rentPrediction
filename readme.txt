process data:  
1. manually identify binary, categorical(binary excluded) and numerical features from Census's https://www.census.gov/housing/nychvs/data/2014/occ_14_long.pdf documentation  
2. replace native-encoded missing value with np.nan  
3. split original dataset into train and test dataset   
4. impute training dataset's missing value (use "most_frequent" strategy for categorical data; "mean" strategy for numerical data)  
5. impute test dataset's missing value using information from test data  
6. perform oneHot on train and test dataset's categorical data (binary features are left out of this processto avoid multicolinearity)   

feature engineering:  
7. apply standard scaler to the train dataset 
8. model with LASSO  
9. remove variables with zero LASSO coeficients 

modelling:  
10. model with Ridge Regression using grid search after scaling (pipeline fashion)   
  
result:  
11. reaches 0.597 R square
