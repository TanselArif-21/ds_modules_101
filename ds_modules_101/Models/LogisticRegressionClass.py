import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_validate
from pandas.api.types import is_numeric_dtype
import statsmodels.api as sm
import warnings
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

class LogisticRegressionClass:
    def __init__(self,df,response,sig_level=0.05):
        '''
        :param df: a dataframe
        :param response: a string. This must be an existing column in df
        '''

        # attach attributes to the object
        self.df = df.copy()
        self.response = response
        self.sig_level = sig_level

    def prepare_data(self,df,response):
        y = df[response]
        X = df[list(filter(lambda x: x != response, df.columns))]
        X = sm.add_constant(X, has_constant='add')

        return X, y

    def logistic_regression_utility_check_response(self,series):
        if (len(series.unique()) > 2):
            print('The response variable has more than 2 categories and is not suitable for logistic regression')
            return False

        if (not is_numeric_dtype(series)):
            print('The response variable should be binary 0 and 1 and numeric type (i.e. int)')
            return False

        return True

    def log_reg_diagnostic_performance(self,X,y):
        cvs = cross_validate(LogisticRegression(), X, y, cv=5,
                             scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'])
        s = """Performance (0 is negative 1 is positive)\n5-Fold Cross Validation Results:\nTest Set accuracy = {}\nf1 = {}\nprecision = {}\nrecall = {}\nauc = {}""".format(
            round(cvs['test_accuracy'].mean(), 2), round(cvs['test_f1'].mean(), 2),
            round(cvs['test_precision'].mean(), 2),
            round(cvs['test_recall'].mean(), 2), round(cvs['test_roc_auc'].mean(), 2))

        self.performance = s
        self.performance_df = pd.DataFrame(data=[round(cvs['test_accuracy'].mean(), 2), round(cvs['test_f1'].mean(), 2),
            round(cvs['test_precision'].mean(), 2),
            round(cvs['test_recall'].mean(), 2), round(cvs['test_roc_auc'].mean(), 2)],
                              index=['test_accuracy','test_f1','test_precision','test_recall','test_roc_auc'],
                                           columns=['Score'])

        return s

    def log_reg_diagnostic_correlations(self,X):
        print("Correlations")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)

        upp_mat = np.triu(X.corr())
        sns.heatmap(X.corr(), vmin=-1, vmax=+1, annot=True, cmap='coolwarm', mask=upp_mat, ax=ax)

        self.fig_correlations = fig
        self.ax_correlations = ax

        return fig,ax

    def logistic_regression_get_report(self,model,X,y,verbose=True):

        preds = model.predict(X)
        df_classification_report = pd.DataFrame(classification_report(y, preds>0.5, output_dict=True))

        df_confusion = pd.DataFrame(confusion_matrix(y, preds>0.5))
        df_confusion.index = list(map(lambda x: 'True_' + str(x), df_confusion.index))
        df_confusion.columns = list(map(lambda x: 'Predicted_' + str(x), df_confusion.columns))

        if verbose:
            print("Classification Report")
            print(df_classification_report)
            print("Confusion Matrix")
            print(df_confusion)

        self.df_confusion = df_confusion
        self.df_classification_report = df_classification_report

        return df_confusion

    def prepare_categories(self,df, response, drop=False):
        cat_cols = list(filter(lambda x: not is_numeric_dtype(df[x]), df.columns))
        cat_cols = list(set(cat_cols) - {response})
        df = pd.get_dummies(df, columns=cat_cols, drop_first=drop)

        return df

    def get_interpretation(self,result,feature_list):
        '''
        Given a trained model, calculate the average probabilities due to feature changes
        '''

        # take a copy of the original df and prepare the dataset
        df_temp = self.df.copy()
        df_temp = self.prepare_categories(df_temp, self.response, drop=False)
        X, y = self.prepare_data(df_temp,self.response)

        full_feature_list = list(feature_list)
        if 'const' not in full_feature_list:
            full_feature_list = ['const'] + full_feature_list

        # get a base probability (this is just the average probability)
        base_probability = result.predict(X[full_feature_list]).mean()
        probability_dict = dict()
        probability_dict['base'] = base_probability

        # for each column in the original df
        for col in self.df.columns:
            # for each column in the result's feature list
            for col2 in feature_list:
                # check if this feature was dummified from this column
                if col + '_' in col2:
                    # if this feature was dummified from this column then update this column to be this feature value
                    df_temp = self.df.copy()
                    df_temp[col] = col2.replace(col + '_', '')
                    df_temp = self.prepare_categories(df_temp, self.response, drop=False)
                    X, y = self.prepare_data(df_temp, self.response)

                    # check that all features the model is expecting exist in X
                    for col3 in feature_list:
                        if col3 not in X.columns:
                            X[col3] = 0

                    # calculate the probability
                    probability = result.predict(X[full_feature_list]).mean()
                    probability_dict[col2] = probability
                elif col == col2:
                    # if this column was not dummified then it is numeric so add 1 to it
                    df_temp = self.df.copy()
                    df_temp[col] = df_temp[col] + 1
                    df_temp = self.prepare_categories(df_temp, self.response, drop=False)
                    X, y = self.prepare_data(df_temp, self.response)
                    probability = result.predict(X[full_feature_list]).mean()
                    probability_dict[col2] = probability

        # save the probability dictionary
        self.feature_interpretability_dict = probability_dict
        self.feature_interpretability_df = pd.DataFrame(data=probability_dict.values(), index=probability_dict.keys(), columns=['Probability'])

        return self.feature_interpretability_df

    def log_reg_basic(self,df=None):
        '''
        Run a basic logistic regression model
        '''

        if df is None:
            df = self.df

        X, y = self.prepare_data(df, self.response)

        model = sm.Logit(y, X)

        result = model.fit(disp=0)

        self.basic_result = result
        self.basic_model = model
        self.X = X
        self.y = y

        return result


    def log_reg(self):
        if not self.logistic_regression_utility_check_response(self.df[self.response]):
            return None

        df1 = self.df[~self.df.isna().any(axis=1)].copy()

        if len(df1) < len(self.df):
            warnings.warn(
                'There are NaNs in the dataset. After removing NaNs, the rows reduce from {} to {}'.format(len(self.df),
                                                                                                           len(df1)))
            print('There are NaNs in the dataset. After removing NaNs, the rows reduce from {} to {}'.format(len(self.df),
                                                                                                           len(df1)))

        df1 = self.prepare_categories(df1,self.response,drop=True)

        result = self.log_reg_basic(df1)

        self.result = result
        self.model = self.basic_model

        return result

    def log_reg_with_feature_selection(self,run_for=0,verbose=True):
        # start the timer in case the is a time limit specified
        start_time = time.time()

        # get rid of nans. There should be no nans. Imputation should be performed prior to this point
        df1 = self.df[~self.df.isna().any(axis=1)].copy()

        # show a warning to let the user know of the droppped nans
        if len(df1) < len(self.df):
            warnings.warn(
                'There are NaNs in the dataset. After removing NaNs, the rows reduce from {} to {}'.format(len(self.df),
                                                                                                           len(df1)))
            print(
                'There are NaNs in the dataset. After removing NaNs, the rows reduce from {} to {}'.format(len(self.df),
                                                                                                           len(df1)))

        # check that the response is in the correct format to perform logistic regression
        if not self.logistic_regression_utility_check_response(df1[self.response]):
            return None

        # automatically identify categorical variables and dummify them
        df1 = self.prepare_categories(df1, self.response, drop=False)

        # raise a warning if the number of columns surpasses the number of rows
        if len(df1.columns) > len(df1):
            warnings.warn(
                'Note: The number of columns after getting dummies is larger than the number of rows. n_cols = {}, nrows = {}'.format(
                    len(df1.columns), len(df1)))

            print(
                'Note: The number of columns after getting dummies is larger than the number of rows. n_cols = {}, nrows = {}'.format(
                    len(df1.columns), len(df1)))

        # the initial list of features
        remaining = list(set(df1.columns) - {self.response})

        # this holds the tried and successful feature set
        full_feature_set = []

        # get the first logistic regression output for only the constant/base model
        first_result = self.log_reg_basic(df1[[self.response]])

        # save the model and the X and y used to train it
        self.X_with_feature_selection = self.X.copy()
        self.y_with_feature_selection = self.y.copy()
        self.model_with_feature_selection = self.basic_model


        # get the pseudo r2 of the base model
        prsquared = first_result.prsquared

        # store the result of the first model
        final_result = first_result

        # while there are still remaining features to try keep looping
        while len(remaining) > 0:
            # store the last pseudo r2 value
            last_prsquared = prsquared

            # the next feature to add to the full feature set
            next_col = None

            # the result corresponding to the addition of the next col
            next_result = None

            # try adding each column from the remaining columns
            for col in sorted(remaining):
                # add the next column to the feature set and try it out. Try except is added because sometimes
                # when categorical variables are dummified and you add both variables you get a singular matrix
                this_feature_set = full_feature_set + [col]
                try:
                    result = self.log_reg_basic(df1[this_feature_set + [self.response]])
                except:
                    remaining.remove(col)
                    continue

                # the resulting pseudo r2 from this fit
                this_prsquared = result.prsquared

                # if a feature results in nan for pseudo r2 skip it
                if this_prsquared is np.nan:
                    print('Note: Feature {} is resulting with a nan adjusted r2. Skipping feature'.format(col))
                    continue

                # this feature is recorded as a candidate if the conditions are met
                if (this_prsquared > last_prsquared) and (result.pvalues.loc[col] <= self.sig_level):
                    last_prsquared = this_prsquared
                    next_col = col
                    next_result = result

                    # save the model and the X and y used to train it
                    self.X_with_feature_selection = self.X.copy()
                    self.y_with_feature_selection = self.y.copy()
                    self.model_with_feature_selection = self.basic_model

            # if after the loop no new candidates were found then we stop looking
            if next_col is None:
                break

            # add the candidate to the permanent list
            full_feature_set.append(next_col)

            # show progress
            if verbose:
                print('********Adding {} with prsquared = {}********'.format(next_col, last_prsquared))

            # store the result
            final_result = next_result

            # remove the chosen candidate from the remaining features
            remaining.remove(next_col)

            # check if it's not taking too long
            if (time.time() - start_time > run_for) and (run_for > 0):
                print(
                    'Aborting: Has been running for {}s > {}s. {} out of {} columns left. There are probably too many categories in one of the columns'.format(
                        round(time.time() - start_time, 2), run_for, len(remaining), len(df1.columns) - 1))
                return

        self.final_feature_set = full_feature_set
        self.result_with_feature_selection = final_result

        return final_result


def unit_test_1():
    print('Unit test 1...')
    import sys
    import os
    import warnings

    np.random.seed(101)
    warnings.filterwarnings("ignore")

    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare']]
    df = df.dropna()
    my_logistic_regresion_class = LogisticRegressionClass(df,'Survived',sig_level=0.05)
    my_logistic_regresion_class.log_reg_basic()

    result_required = [2.75415827153399, -1.2422486253277711, 2.6348448348873723, -0.043952595897772694, -0.375754870508454, -0.06193736644803373, 0.002160033540727779]
    result_actual = list(my_logistic_regresion_class.basic_result.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    result_required = 0.4061624649859944
    result_actual = my_logistic_regresion_class.basic_result.predict(my_logistic_regresion_class.X).mean()

    result_required = round(result_required, 2)
    result_actual = round(result_actual, 2)

    assert (result_required == result_actual)

    result_required = '''Performance (0 is negative 1 is positive)
5-Fold Cross Validation Results:
Test Set accuracy = 0.79
f1 = 0.73
precision = 0.76
recall = 0.7
auc = 0.85'''
    result_actual = my_logistic_regresion_class.log_reg_diagnostic_performance(my_logistic_regresion_class.X,
                                                               my_logistic_regresion_class.y)

    assert (result_required == result_actual)

    result_required = [0.4061624649859944, 0.24581360407372246, 0.795820946281563, 0.3999162261394402, 0.3539768140703711, 0.39737068898845873, 0.4064703482674913]
    result_actual = list(my_logistic_regresion_class.get_interpretation(my_logistic_regresion_class.basic_result,my_logistic_regresion_class.X.columns)['Probability'])

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (result_required == result_actual)

    print('Success!')


def unit_test_2():
    print('Unit test 2...')
    import sys
    import os
    import warnings

    np.random.seed(101)
    warnings.filterwarnings("ignore")

    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare']]
    df = df.dropna()
    my_logistic_regresion_class = LogisticRegressionClass(df,'Survived',sig_level=0.05)
    my_logistic_regresion_class.log_reg()

    result_required = [2.75415827153399, -1.2422486253277711, 2.6348448348873723, -0.043952595897772694, -0.375754870508454, -0.06193736644803373, 0.002160033540727779]
    result_actual = list(my_logistic_regresion_class.basic_result.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    result_required = 0.4061624649859944
    result_actual = my_logistic_regresion_class.basic_result.predict(my_logistic_regresion_class.X).mean()

    result_required = round(result_required, 2)
    result_actual = round(result_actual, 2)

    assert (result_required == result_actual)

    result_required = '''Performance (0 is negative 1 is positive)
5-Fold Cross Validation Results:
Test Set accuracy = 0.79
f1 = 0.73
precision = 0.76
recall = 0.7
auc = 0.85'''

    result_actual = my_logistic_regresion_class.log_reg_diagnostic_performance(my_logistic_regresion_class.X,
                                                                               my_logistic_regresion_class.y)

    assert (result_required == result_actual)

    result_required = [0.4061624649859944, 0.24581360407372246, 0.795820946281563, 0.3999162261394402,
                       0.3539768140703711, 0.39737068898845873, 0.4064703482674913]
    result_actual = list(my_logistic_regresion_class.get_interpretation(my_logistic_regresion_class.basic_result,
                                                                        my_logistic_regresion_class.X.columns)[
                             'Probability'])

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (result_required == result_actual)

    print('Success!')

def unit_test_3():
    print('Unit test 3...')
    import sys
    import os
    import warnings

    np.random.seed(101)
    warnings.filterwarnings("ignore")

    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare']]
    df = df.dropna()
    my_logistic_regresion_class = LogisticRegressionClass(df,'Survived',sig_level=0.05)
    my_logistic_regresion_class.log_reg()

    result_required = [5.389003106421364, -1.2422486253277716, -0.043952595897772714, -0.3757548705084541, -0.0619373664480337, 0.002160033540727774, -2.6348448348873723]
    result_actual = list(my_logistic_regresion_class.result.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    result_required = 0.4061624649859944
    result_actual = my_logistic_regresion_class.result.predict(my_logistic_regresion_class.X).mean()

    result_required = round(result_required, 2)
    result_actual = round(result_actual, 2)

    assert (result_required == result_actual)

    result_required = '''Performance (0 is negative 1 is positive)
5-Fold Cross Validation Results:
Test Set accuracy = 0.79
f1 = 0.73
precision = 0.76
recall = 0.7
auc = 0.85'''

    result_actual = my_logistic_regresion_class.log_reg_diagnostic_performance(my_logistic_regresion_class.X,
                                                                               my_logistic_regresion_class.y)

    assert (result_required == result_actual)

    result_required = [0.40616246498599445, 0.24581360407372244, 0.23089275089703268, 0.3999162261394402, 0.3539768140703711, 0.39737068898845873, 0.4064703482674913]
    result_actual = list(my_logistic_regresion_class.get_interpretation(my_logistic_regresion_class.result,
                                                                        my_logistic_regresion_class.X.columns)[
                             'Probability'])

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (result_required == result_actual)

    print('Success!')


def unit_test_4():
    print('Unit test 4...')
    import sys
    import os
    import warnings

    np.random.seed(101)
    warnings.filterwarnings("ignore")

    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare']]
    df = df.dropna()
    my_logistic_regresion_class = LogisticRegressionClass(df,'Survived',sig_level=0.05)
    my_logistic_regresion_class.log_reg_with_feature_selection(verbose=False)

    result_required = [2.98, 2.62, -1.32, -0.04, -0.38]
    result_actual = list(my_logistic_regresion_class.result_with_feature_selection.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    result_required = 0.4061624649859944
    result_actual = my_logistic_regresion_class.result_with_feature_selection.predict(my_logistic_regresion_class.X_with_feature_selection).mean()

    result_required = round(result_required, 2)
    result_actual = round(result_actual, 2)

    assert (result_required == result_actual)

    result_required = '''Performance (0 is negative 1 is positive)
5-Fold Cross Validation Results:
Test Set accuracy = 0.8
f1 = 0.74
precision = 0.78
recall = 0.72
auc = 0.85'''

    result_actual = my_logistic_regresion_class.log_reg_diagnostic_performance(my_logistic_regresion_class.X_with_feature_selection,
                                                                               my_logistic_regresion_class.y_with_feature_selection)

    assert (result_required == result_actual)

    result_required = [0.4061624649859944, 0.236473141666754, 0.7141621577909735, 0.3998399415589254, 0.3537524130019424]
    result_actual = list(my_logistic_regresion_class.get_interpretation(my_logistic_regresion_class.result_with_feature_selection,
                                                                        my_logistic_regresion_class.X_with_feature_selection.columns)[
                             'Probability'])

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (result_required == result_actual)

    result_required = [365, 59, 78, 212]
    result_actual = my_logistic_regresion_class.logistic_regression_get_report(
        my_logistic_regresion_class.result_with_feature_selection,
        my_logistic_regresion_class.X_with_feature_selection,
        my_logistic_regresion_class.y_with_feature_selection,verbose=False)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), list(np.array(result_actual).flatten())))

    assert (result_required == result_actual)


    print('Success!')

if __name__ == '__main__':
    unit_test_1()
    unit_test_2()
    unit_test_3()
    unit_test_4()
