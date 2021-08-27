import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,classification_report,confusion_matrix
from sklearn.model_selection import cross_validate
from pandas.api.types import is_numeric_dtype
import statsmodels.api as sm
import warnings
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def log_reg(df,response,include_diagnostics='Yes'):
    if not logistic_regression_utility_check_response(df[response]):
        return None

    df1 = df[~df.isna().any(axis=1)].copy()

    if len(df1) < len(df):
        warnings.warn(
            'There are NaNs in the dataset. After removing NaNs, the rows reduce from {} to {}'.format(len(df),
                                                                                                       len(df1)))
        print('There are NaNs in the dataset. After removing NaNs, the rows reduce from {} to {}'.format(len(df),
                                                                                                       len(df1)))

    cat_cols = list(filter(lambda x: not is_numeric_dtype(df1[x]),df1.columns))
    cat_cols = list(set(cat_cols) - {response})
    df1 = pd.get_dummies(df1,columns=cat_cols,drop_first=True)

    result,model = log_reg_basic(df1,response)

    if include_diagnostics == 'Yes':
        log_reg_basic(df1, response,with_diagnostics=True)

    return result

def log_reg_basic(df,response,with_diagnostics = False):
    y = df[response]
    X = df[list(filter(lambda x: x != response,df.columns))]

    X = sm.add_constant(X)

    model = sm.Logit(y,X)

    result = model.fit(disp=0)

    if with_diagnostics:
        log_reg_diagnostic_performance(df, response)
        log_reg_diagnostic_correlations(df)
        logistic_regression_get_report(df, response)

    return result,model


def log_reg_with_feature_selection(df, response, run_for=0, include_diagnostics='Yes'):
    start_time = time.time()

    df1 = df[~df.isna().any(axis=1)].copy()

    if len(df1) < len(df):
        warnings.warn(
            'There are NaNs in the dataset. After removing NaNs, the rows reduce from {} to {}'.format(len(df),
                                                                                                       len(df1)))
        print(
            'There are NaNs in the dataset. After removing NaNs, the rows reduce from {} to {}'.format(len(df),
                                                                                                       len(df1)))
    if not logistic_regression_utility_check_response(df1[response]):
        return None


    cat_cols = list(filter(lambda x: not is_numeric_dtype(df1[x]), df.columns))
    cat_cols = list(set(cat_cols) - {response})
    df1 = pd.get_dummies(df1, columns=cat_cols, drop_first=False)

    if len(df1.columns) > len(df1):
        warnings.warn(
            'Note: The number of columns after getting dummies is larger than the number of rows. n_cols = {}, nrows = {}'.format(
                len(df1.columns), len(df1)))

        print(
            'Note: The number of columns after getting dummies is larger than the number of rows. n_cols = {}, nrows = {}'.format(
                len(df1.columns), len(df1)))

    remaining = list(set(df1.columns) - {response})
    full_feature_set = []

    first_result, first_model = log_reg_basic(df1[[response]], response)
    prsquared = first_result.prsquared

    final_model = first_model
    final_result = first_result

    while len(remaining) > 0:
        last_prsquared = prsquared
        next_col = None
        next_model = None
        next_result = None
        remove_cols = None

        for col in sorted(remaining):
            this_feature_set = full_feature_set + [col]
            try:
                result, model = log_reg_basic(df1[this_feature_set + [response]], response)
            except:
                remaining.remove(col)
                continue
            this_prsquared = result.prsquared
            #result.pvalues['Age']
            if this_prsquared is np.nan:
                print('Note: Feature {} is resulting with a nan adjusted r2. Skipping feature'.format(col))
                continue

            if (this_prsquared > last_prsquared) and (result.pvalues.loc[col] <= 0.05):
                last_prsquared = this_prsquared
                next_col = col
                next_model = model
                next_result = result

        if next_col is None:
            break

        full_feature_set.append(next_col)
        print('********Adding {} with prsquared = {}********'.format(next_col,last_prsquared))
        final_model = next_model
        final_result = next_result
        remaining.remove(next_col)

        if remove_cols is not None:
            for rem_col in remove_cols:
                remaining.remove(rem_col)

        if (time.time() - start_time > run_for) and (run_for > 0):
            print(
                'Aborting: Has been running for {}s > {}s. {} out of {} columns left. There are probably too many categories in one of the columns'.format(
                    round(time.time() - start_time, 2), run_for, len(remaining), len(df1.columns) - 1))
            return

    if include_diagnostics == 'Yes':
        log_reg_basic(df1[full_feature_set + [response]], response, with_diagnostics=True)

    return final_result

def log_reg_diagnostic_performance(df,response):
    print("Performance (0 is negative 1 is positive)")
    y = df[response]
    X = df[list(filter(lambda x: x != response, df.columns))]

    cvs = cross_validate(LogisticRegression(), X, y, cv=5, scoring=['accuracy', 'f1','precision','recall','roc_auc'])
    s = """5-Fold Cross Validation Results:\nTest Set accuracy = {}\nf1 = {}\nprecision = {}
    \nrecall = {}\nauc = {}""".format(
        round(cvs['test_accuracy'].mean(), 2), round(cvs['test_f1'].mean(), 2), round(cvs['test_precision'].mean(), 2),
                round(cvs['test_recall'].mean(), 2), round(cvs['test_roc_auc'].mean(), 2))

    print("{}".format(s))

def log_reg_diagnostic_correlations(df):
    print("Correlations")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    upp_mat = np.triu(df.corr())
    sns.heatmap(df.corr(), vmin=-1, vmax=+1, annot=True, cmap='coolwarm', mask=upp_mat,ax=ax)

    fig

def logistic_regression_utility_check_response(series):
    if (len(series.unique()) > 2):
        print('The response variable has more than 2 categories and is not suitable for logistic regression')
        return False

    if (not is_numeric_dtype(series)):
        print('The response variable should be binary 0 and 1 and numeric type (i.e. int)')
        return False

    return True

def logistic_regression_get_report(df,response):
    print("Classification Report")

    y = df[response]
    X = df[list(filter(lambda x: x != response, df.columns))]

    model = LogisticRegression()
    model.fit(X,y)
    preds = model.predict(X)

    display(pd.DataFrame(classification_report(y,preds,output_dict=True)))

    print("Confusion Matrix")
    df_confusion = pd.DataFrame(confusion_matrix(y, preds))
    df_confusion.index = list(map(lambda x: 'True_' + str(x), df_confusion.index))
    df_confusion.columns = list(map(lambda x: 'Predicted_' + str(x), df_confusion.columns))
    display(df_confusion)

def unit_test_1():
    import sys
    import os
    import warnings

    np.random.seed(101)
    warnings.filterwarnings("ignore")

    current_dir = sys.path[0]
    data_dir = os.path.join(current_dir,'Data')
    titanic_csv = os.path.join(data_dir,'Titanic.csv')
    df = pd.read_csv(titanic_csv)
    log_reg_result = log_reg(df[['Age','Sex','Pclass','Embarked','survived','Fare']],'Sex', include_diagnostics='Yes')

    this_result = list(map(lambda x: round(x,2),log_reg_result.params))
    required_result = list(map(lambda x: round(x,2),[-1.26, -0.01, 0.14, -0.12, 1.84, 0.01]))

    assert (sorted(this_result) == sorted(required_result))

def unit_test_2():
    import sys
    import os
    import warnings

    np.random.seed(101)
    warnings.filterwarnings("ignore")

    current_dir = sys.path[0]
    data_dir = os.path.join(current_dir, 'Data')
    titanic_csv = os.path.join(data_dir, 'Titanic.csv')
    df = pd.read_csv(titanic_csv)
    log_reg_result = log_reg_with_feature_selection(df[['Age', 'Sex', 'Pclass', 'Embarked', 'survived', 'Fare']], 'Sex',
                             include_diagnostics='Yes')

    this_result = list(map(lambda x: round(x, 2), log_reg_result.params))
    required_result = list(map(lambda x: round(x, 2), [-1.26, -0.01, 0.14, -0.12, 1.84, 0.01]))

    assert (sorted(this_result) == sorted(required_result))

def unit_test_3():
    import sys
    import os
    import warnings

    np.random.seed(101)
    warnings.filterwarnings("ignore")

    current_dir = sys.path[0]
    data_dir = os.path.join(current_dir, 'Data')
    titanic_csv = os.path.join(data_dir, 'Titanic.csv')
    df = pd.read_csv(titanic_csv)
    df['Sex'] = df['Sex'].map({0:'Woman',1:'Man'})
    log_reg_result = log_reg_with_feature_selection(df[['Age', 'Sex', 'Pclass', 'Embarked', 'survived', 'Fare']], 'Sex',
                             include_diagnostics='Yes')

    this_result = list(map(lambda x: round(x, 2), log_reg_result.params))
    required_result = list(map(lambda x: round(x, 2), [-1.26, -0.01, 0.14, -0.12, 1.84, 0.01]))

    assert (sorted(this_result) == sorted(required_result))

if __name__ == '__main__':
    # unit_test_1()
    # unit_test_2()
    unit_test_3()
