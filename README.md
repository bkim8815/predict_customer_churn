# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

The goal of this project is to predict customer churn based on their information to proactively prevent the customer leaving credit card service.

Given customer dataset(`./data/bank_data.csv`) contains 18 features of 10000 customers(both churned and existing) including age, gender, marital status, income level, credit card limit, number of dependents and etc.

Using this project you can predict which

This Project contains two main files.
- churn_library.py
- churn_script_logging_and_tests.py

churn_library.py runs EDA on dataframe and finds association between features and customer churn which then produces trained models.

you can utilized this output models to make prediction on customer churn.
Output files can be found in images and models folder.
Output of this prediction includes:
  - feature_importance.png
  - logistic_model.pkl
  - rfc_model.pkl



## Running Files
### To run prediction model.
Follow this steps to run the prediciton model.

1. Make sure python3/pip is installed and run pip install all dependencies
    ```
    pip install scikit-learn==0.22 shap pylint au pylint autopep8
    ```
2. To run test
    ```
    python churn_script_logging_and_tests.py
    ```
3. output files can be found in images and logs can be found in logs folder

4. To lint, Run
    ```
    pylint churn_library.py
    pylint churn_script_logging_and_tests.py
    ```
5. To run autopep:
    ```
    autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
    autopep8 --in-place --aggressive --aggressive churn_library.py
    ```
