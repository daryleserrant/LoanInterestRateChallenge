
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from datetime import datetime
import autosklearn.regression

def main():
    df = pd.read_csv('Data for Cleaning & Modeling.csv')
    
    column_names = ['interest_rate','loan_id','borrower_id','amount_requested','amount_funded',
                    'investor_funded_portion','number_of_payments','loan_grade','loan_subgrade',
                    'employer_or_jobtitle','years_employed','home_ownership_status','annual_income',
                    'income_verification_status','issue_date','loan_reason','loan_category','loan_title',
                    'zip_code','state','debt_to_income_ratio','number_of_deliquencies','earliest_credit_line_date',
                    'number_of_inquiries','months_since_last_delinquency','months_since_last_public_record',
                    'number_of_open_credit_lines','number_of_derogatory_public_records','total_revolving_credit',
                    'line_utilization_rate','number_of_credit_lines','listing_status']
    df.columns = column_names

    # Drop rows with missing interest rates, line utilization rate, and loan_id

    loan_df = df[pd.notnull(df.loan_id) & pd.notnull(df.interest_rate) & pd.notnull(df.line_utilization_rate)].copy()

    # Convert amount requested, line utilization rate, and interest rate fields into numeric types
    loan_df.amount_requested = loan_df.amount_requested.apply(lambda x: float(x.replace('$','').replace(',','')))
    loan_df.line_utilization_rate = loan_df.line_utilization_rate.apply(lambda x: float(x.replace('%','')) if pd.notnull(x) else x)
    loan_df.interest_rate = loan_df.interest_rate.apply(lambda x: float(x.replace('%','')))
    
    # Extract date fields
    loan_df['issue_year'] = loan_df.issue_date.apply(lambda x: int(x.split('-')[1]))
    
    month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    loan_df['issue_month'] = loan_df.issue_date.apply(lambda x: month_map[x.split('-')[0]])
    
    loan_df.listing_status = loan_df.listing_status.apply(lambda x: 1 if x == 'f' else 0)
    loan_df.number_of_payments = loan_df.number_of_payments.apply(lambda x: 1 if x == " 36 months" else 0)

    subgrade_map = {g:n+1 for n,g in enumerate(sorted(loan_df[loan_df.loan_subgrade.notnull()].loan_subgrade.unique()))}
    loan_df.loan_subgrade = loan_df.loan_subgrade.apply(lambda x: subgrade_map[x] if x in subgrade_map else x)

    # Clean deliquency values
    loan_df['had_deliquency'] = loan_df.months_since_last_delinquency.apply(lambda x: 1 if pd.notnull(x) else 0)
    loan_df.months_since_last_delinquency = loan_df.months_since_last_delinquency.fillna(0)
    
    loan_df['credit_age'] = loan_df.earliest_credit_line_date.apply(lambda x: 2016 - datetime.strptime(x,'%b-%y').year)

    # To property validate performance of machine learning model we will split the dataset into training
    # and test set before running the automl process. Test set will only have rows that don't have missing data
    # data. Autosklearn will impute any data that is missing

    nonnull_data = loan_df[loan_df.annual_income.notnull() & loan_df.loan_subgrade.notnull()]

    testset_idx = np.random.choice(nonnull_data.index, int(0.30*nonnull_data.shape[0]))
    testset_df = nonnull_data.loc[testset_idx].copy()
    loan_df = loan_df[~loan_df.index.isin(testset_idx)]
    
    features = ['amount_requested','number_of_payments','annual_income','loan_subgrade',
            'issue_year','issue_month','credit_age', 'debt_to_income_ratio','line_utilization_rate','had_deliquency',
            'months_since_last_delinquency','number_of_open_credit_lines','number_of_inquiries', 'total_revolving_credit',
            'number_of_derogatory_public_records','listing_status']
    
    feature_types = ['numerical','categorical','numerical','numerical',
                     'numerical','numerical','numerical','numerical', 'numerical','numerical',
                     'numerical','numerical','numerical','numerical','numerical',
                     'categorical']
    
    X_train = loan_df[features]
    y_train = loan_df['interest_rate']

    X_test = testset_df[features]
    y_test = testset_df['interest_rate']
    
    automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=3600,
    per_run_time_limit=360,
    tmp_folder='/tmp/autosklearn_regression_example_tmp',
    output_folder='/tmp/autosklearn_regression_example_out',
    )
    automl.fit(X_train, y_train, dataset_name='loan',
               feat_type=feature_types)

    print(automl.show_models())
    predictions = automl.predict(X_test)
    print("RMSE:", mean_squared_error(y_test,predictions)**0.5)

    '''
    OUTPUT:
    
    [(0.720000,
      SimpleRegressionPipeline({'categorical_encoding:__choice':'one_hot_encoding',
                                'imputation:strategy':'mean',
                                'preprocessor:__choice__':'no_preprocessing',
                                'regressor:__choice__':'random_forest',
                                'rescaling:__choice__':'standardize',
                                'categorical_encoding:one_hot_encoding:use_minimum_fraction': 'True',
                                'regressor:random_forest:bootstrap':'True',
                                'regressor:random_forest:criterion':'mse',
                                'regressor:random_forest:max_depth':'None',
                                'regressor:random_forest:max_features':1.0,
                                'regressor:random_forest:max_leaf_nodes':'None',
                                'regressor:random_forest:min_impurity_decrease':0.0,
                                'regressor:random_forest:min_samples_leaf':1,
                                'regressor:random_forest:min_samples_split':2,
                                'regressor:random_forest:min_weight_fraction_leaf':0.0,
                                'regressor:random_forest:n_estimators':100,
                                'categorical_encoding:one_hot_encoding:minimum_fraction':0.01}
      dataset_properties={
        'task':4,
        'sparse':False,
        'multilabel':False,
        'multiclass':False,
        'target_type':'regression',
        'signed':False})),
     (0.280000,
      SimpleRegressionPipeline({'categorical_encoding:__choice':'one_hot_encoding',
                                'imputation:strategy':'median',
                                'preprocessor:__choice__':'polynomial',
                                'regressor:__choice__':'gradient_boosting',
                                'rescaling:__choice__':'robust_scaler',
                                'preprocessor:polynomial:degree':3,
                                'preprocessor:polynomial:include_bias':'False',
                                'preprocessor:polynomial:interaction_only':'True',
                                'regressor:gradient_boosting:early_stop': 'valid',
                                'regressor:gradient_boosting:l2_regularization':0.0017344914851347946,
                                'regressor:gradient_boosting:learning_rate':0.3380352235102556,
                                'regressor:gradient_boosting:loss': 'least_squares',
                                'regressor:gradient_boosting:max_bins': 256,
                                'regressor:gradient_boosting:max_depth': 'None',
                                'regressor:gradient_boosting:max_leaf_nodes': 64,
                                'regressor:gradient_boosting:min_samples_leaf':3,
                                'regressor:gradient_boosting:scoring':'loss',
                                'regressor:gradient_boosting:tol':  1e-07,
                                'rescaling:robust_scaler:q_max': 0.849311043670883,
                                'rescaling:robust_scaler:q_min': 0.065001913429137,
                                'regressor:gradient_boosting:n_iter_no_change':9,
                                'regressor:gradient_boosting:validation_fraction':0.15774326288476428},
      dataset_properties={
        'task':4,
        'sparse':False,
        'multilabel':False,
        'multiclass':False,
        'target_type':'regression',
        'signed':False}))]
     
    RMSE: 0.23420240438526985'''


if __name__ == '__main__':
    main()
