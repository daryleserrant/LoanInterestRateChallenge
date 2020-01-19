
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

    '''annual_income_lkup1 = loan_df.groupby(['state','zip_code','employer_or_jobtitle'])['annual_income'].median().dropna().to_dict()
    annual_income_lkup2 = loan_df.groupby(['state','employer_or_jobtitle'])['annual_income'].median().dropna().to_dict()
    annual_income_lkup3 = loan_df.groupby(['state','zip_code'])['annual_income'].median().dropna().to_dict()
    annual_income_lkup4 = loan_df.groupby(['state'])['annual_income'].median().to_dict()
    
    def fill_annual_income(state, zip_code, employment):
        if (state, zip_code, employment) in annual_income_lkup1:
            return annual_income_lkup1[(state, zip_code, employment)]
        elif (state, employment) in annual_income_lkup2:
            return annual_income_lkup2[(state, employment)]
        elif (state, zip_code) in annual_income_lkup3:
            return annual_income_lkup3[(state, zip_code)]
        else:
            return annual_income_lkup4[state]
    
    loan_df.annual_income = loan_df.apply(lambda row: fill_annual_income(row['state'],row['zip_code'],row['employer_or_jobtitle']) if pd.isnull(row['annual_income']) else row['annual_income'], axis=1)

    knn_clf = KNeighborsClassifier(n_neighbors=5)
    
    knn_features = ['interest_rate']
    
    knn_df = loan_df[loan_df.loan_subgrade.notnull()][knn_features]
    
    Y_knn = loan_df[loan_df.loan_subgrade.notnull()]['loan_subgrade']
    
    knn_clf.fit(knn_df,Y_knn)
    
    null_grades = loan_df[loan_df.loan_subgrade.isnull()][['interest_rate','loan_subgrade']].copy()
    null_grades['loan_subgrade'] = knn_clf.predict(null_grades.interest_rate.values.reshape(-1,1))
    loan_df.update(null_grades)'''
    
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

    #features_df = loan_df[features]
    #Y = loan_df['interest_rate']

    #X_train, X_test, y_train, y_test = train_test_split(features_df, Y, test_size=0.3, random_state=50)
    
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

if __name__ == '__main__':
    main()
