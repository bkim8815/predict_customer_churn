# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Objective of this project is to predict customer churn.



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
