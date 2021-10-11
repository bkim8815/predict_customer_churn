'''
This is a test file for churn_libarary.py

Author: Brian Kim
'''
import os
import logging
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_frame = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    return data_frame


def test_eda(df, perform_eda):
    '''
    test perform eda function
    '''
    try:
        perform_eda(df)
        assert os.path.isfile('./images/churn_hist.png')
        assert os.path.isfile('./images/customer_age_hist.png')
        assert os.path.isfile('./images/heatmap.png')
        assert os.path.isfile('./images/marital_status_hist.png')
        assert os.path.isfile('./images/total_trans_ct.png')
        logging.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda: The png file wasn't found")
        raise err


def test_encoder_helper(df, encoder_helper):
    '''
    test encoder helper
    '''
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    try:
        encoded_df = encoder_helper(df, category_lst)
        for cat in category_lst:
            assert f"{cat}_Churn" in encoded_df.columns
        logging.info("Testing encoder_helper: SUCCESS")
        return encoded_df
    except FileNotFoundError as err:
        logging.error(
            "Testing encoder_helper: The encoded column was not found")
        raise err


def test_perform_feature_engineering(df, perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''

    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)
        assert len(X_train) > 0
        assert len(X_train) == len(y_train)
        assert len(y_train) > 0
        assert len(X_test) == len(y_test)
        logging.info("Testing perform_feature_engineering: SUCCESS")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as err:
        logging.error(
            "Testing perform_feature_engineering: train, test dataset invalid")
        raise err


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    try:
        train_models(X_train, X_test, y_train, y_test)
        assert os.path.isfile('./images/RFTrain_RFTest.png')
        assert os.path.isfile('./images/LRTrain_LRRgression.png')
        assert os.path.isfile('./models/rfc_model.pkl')
        assert os.path.isfile('./models/logistic_model.pkl')
        assert os.path.isfile('./images/shap_summary.png')
        assert os.path.isfile('./images/lrc_plot.png')
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: output files not found")
        raise err


if __name__ == "__main__":
    df = test_import(cls.import_data)
    test_eda(df, cls.perform_eda)
    encoded_df = test_encoder_helper(df, cls.encoder_helper)
    X_train, X_test, y_train, y_test = test_perform_feature_engineering(
        encoded_df, cls.perform_feature_engineering)
    test_train_models(cls.train_models, X_train, X_test, y_train, y_test)
