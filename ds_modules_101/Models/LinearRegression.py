import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from pandas.api.types import is_numeric_dtype
import statsmodels.api as sm
import warnings
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

class LinearRegressionClass:
    def __init__(self,df,response,sig_level=0.05,max_iter=500,cols_to_keep_static=[],cols_to_try_individually=[]):
        '''
        :param df: a dataframe
        :param response: a string. This must be an existing column in df
        :param sig_level: a float. The significance level the forward selection will use
        :param max_iter: an integer. The maximum iterations the solvers will use to try to converge
        :param cols_to_keep_static: a list. Used in forward selection to not omit these columns
        :param cols_to_try_individually: a list. The columns to test in a regression one at a time to identify which
            one has the greatest relationship with the response controlled for the cols_to_keep_static
        '''

        # attach attributes to the object
        self.df = df.copy()
        self.response = response
        self.sig_level = sig_level
        self.max_iter=max_iter
        self.warnings = ''
        self.error_message = ''
        self.cols_to_keep_static = cols_to_keep_static
        self.cols_to_try_individually = cols_to_try_individually

        if self.response in self.cols_to_keep_static:
            print('The response - {} is in the static columns. Removed it.'.format(response))
            self.cols_to_keep_static = list(filter(lambda x: x != self.response,self.cols_to_keep_static))

        if self.response in self.cols_to_try_individually:
            print('The response - {} is in the cols to try individually columns. Removed it.'.format(response))
            self.cols_to_try_individually = list(filter(lambda x: x != self.response,self.cols_to_try_individually))

    def prepare_data(self,df,response):
        y = df[response]
        X = df[list(filter(lambda x: x != response, df.columns))]
        X = sm.add_constant(X, has_constant='add')

        return X, y

    def linear_regression_utility_check_response(self,series):

        if (not is_numeric_dtype(series)):
            self.error_message = self.error_message + '\n' + 'The response variable should be numeric type'
            print('The response variable should be numeric type')
            return False

        return True

    def lin_reg_diagnostic_performance(self,X,y):
        cvs = cross_validate(LinearRegression(), X, y, cv=5,
                             scoring=['r2', 'neg_mean_squared_error', 'neg_root_mean_squared_error'])
        s = """Performance\n5-Fold Cross Validation Results:\nTest Set r2 = {}\nneg_mean_squared_error = {}\nneg_root_mean_squared_error = {}""".format(
            round(cvs['test_r2'].mean(), 2), round(cvs['test_neg_mean_squared_error'].mean(), 2),
            round(cvs['test_neg_root_mean_squared_error'].mean(), 2))

        self.performance = s
        self.performance_df = pd.DataFrame(data=[round(cvs['test_r2'].mean(), 2), round(cvs['test_neg_mean_squared_error'].mean(), 2),
            round(cvs['test_neg_root_mean_squared_error'].mean(), 2)],
                              index=['test_r2','test_neg_mean_squared_error','test_neg_root_mean_squared_error'],
                                           columns=['Score'])

        return s

    def lin_reg_diagnostic_correlations(self,X):
        print("Correlations")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)

        upp_mat = np.triu(X.corr())
        sns.heatmap(X.corr(), vmin=-1, vmax=+1, annot=True, cmap='coolwarm', mask=upp_mat, ax=ax)

        self.fig_correlations = fig
        self.ax_correlations = ax

        return fig,ax

    def linear_regression_get_report(self,model,X,y,verbose=True):

        pass

    def prepare_categories(self,df, response, drop=False):
        cat_cols = list(filter(lambda x: not is_numeric_dtype(df[x]), df.columns))
        cat_cols = list(set(cat_cols) - {response} - set(self.cols_to_keep_static))
        df = pd.get_dummies(df, columns=cat_cols, drop_first=drop)
        df = pd.get_dummies(df, columns=self.cols_to_keep_static, drop_first=True)

        self.cols_to_keep_static_dummified = []
        for col in self.cols_to_keep_static:
            for col_dummy in df.columns:
                if col in col_dummy:
                    self.cols_to_keep_static_dummified.append(col_dummy)

        return df

    def get_interpretation(self,result=None,feature_list=None,df=None):
        '''
        Given a trained model, calculate the average probabilities due to feature changes
        '''

        if (result is None) or (feature_list is None):
            try:
                feature_list = self.X_with_feature_selection.columns
                result = self.result_with_feature_selection
            except:
                feature_list = self.X.columns
                try:
                    result = self.result
                except:
                    result = self.basic_result


        # take a copy of the original df and prepare the dataset
        if df is None:
            df = self.df.copy()

        df_temp = df.copy()
        df_temp = self.prepare_categories(df_temp, self.response, drop=False)
        X, y = self.prepare_data(df_temp,self.response)

        full_feature_list = list(feature_list)
        if 'const' not in full_feature_list:
            full_feature_list = ['const'] + full_feature_list

        # comparative uplift section
        comparative_dict = dict()
        for col1 in df.columns:
            for col2 in full_feature_list:
                # if this feature was dummified
                if col1 + '_' in col2:
                    t = X[full_feature_list].copy()
                    # First get prediction with 0
                    t[col2] = 0
                    comparative_dict[col2] = [result.predict(t).mean()]
                    # Then get prediction with 1
                    t[col2] = 1
                    comparative_dict[col2].append(result.predict(t).mean())
                elif col1 == col2:
                    t = X[full_feature_list].copy()
                    # first get prediction with average
                    t[col2] = t[col2].mean()
                    comparative_dict[col2] = [result.predict(t).mean()]
                    # then get prediction with +1
                    t[col2] = t[col2] + 1
                    comparative_dict[col2].append(result.predict(t).mean())

        feature_interpretability_comparative_df = pd.DataFrame(comparative_dict).T
        feature_interpretability_comparative_df.columns = ['Prediction_average_or_without','Prediction_add1_or_with']
        feature_interpretability_comparative_df['diff'] = feature_interpretability_comparative_df['Prediction_add1_or_with'] - feature_interpretability_comparative_df['Prediction_average_or_without']
        self.feature_interpretability_comparative_df = feature_interpretability_comparative_df

        # get a base probability (this is just the average probability)
        base_probability = result.predict(X[full_feature_list]).mean()
        probability_dict = dict()
        probability_dict['base'] = base_probability

        # for each column in the original df
        for col in df.columns:
            # for each column in the result's feature list
            for col2 in feature_list:
                # check if this feature was dummified from this column
                if col + '_' in col2:
                    # if this feature was dummified from this column then update this column to be this feature value
                    df_temp = df.copy()
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
                    df_temp = df.copy()
                    df_temp[col] = df_temp[col] + 1
                    df_temp = self.prepare_categories(df_temp, self.response, drop=False)
                    X, y = self.prepare_data(df_temp, self.response)
                    probability = result.predict(X[full_feature_list]).mean()
                    probability_dict[col2] = probability

        # save the probability dictionary
        self.feature_interpretability_dict = probability_dict
        self.feature_interpretability_df = pd.DataFrame(data=probability_dict.values(), index=probability_dict.keys(), columns=['Probability'])

        return self.feature_interpretability_df

    def lin_reg_basic(self,df=None):
        '''
        Run a basic logistic regression model
        '''

        if df is None:
            df = self.df

        X, y = self.prepare_data(df, self.response)

        model = sm.OLS(y, X)

        result = model.fit(maxiter=self.max_iter)

        self.basic_result = result
        self.basic_model = model
        self.X = X
        self.y = y

        return result

    def predict_from_original(self,df):
        df = self.prepare_categories(df, self.response, drop=False)
        all_cols = []
        try:
            all_cols = list(self.X_with_feature_selection.columns)
        except:
            all_cols = list(self.X.columns)

        for col in all_cols:
            if col not in df.columns:
                df[col] = 0

        res = None
        try:
            res = self.result_with_feature_selection
        except:
            res = self.result

        return res.predict(df[all_cols])

    def lin_reg(self,df=None):
        if df is None:
            df1 = self.df[~self.df.isna().any(axis=1)].copy()

            if len(df1) < len(self.df):
                warning_message = 'There are NaNs in the dataset. After removing NaNs, the rows reduce from {} to {}'.format(len(self.df),
                                                                                                               len(df1))
                warnings.warn(warning_message)
                print(warning_message)

                self.warnings = self.warnings + '\n' + warning_message
        else:
            df1 = df[~df.isna().any(axis=1)].copy()

            if len(df1) < len(df):
                warning_message = 'There are NaNs in the dataset. After removing NaNs, the rows reduce from {} to {}'.format(
                    len(df),
                    len(df1))
                warnings.warn(warning_message)
                print(warning_message)

                self.warnings = self.warnings + '\n' + warning_message

        if not self.linear_regression_utility_check_response(df1[self.response]):
            return None

        df1 = self.prepare_categories(df1,self.response,drop=True)

        result = self.lin_reg_basic(df1)

        self.result = result
        self.model = self.basic_model

        return result

    def lin_reg_with_feature_selection(self,df=None,run_for=0,verbose=True):
        # start the timer in case the is a time limit specified
        start_time = time.time()

        if df is None:
            # get rid of nans. There should be no nans. Imputation should be performed prior to this point
            df1 = self.df[~self.df.isna().any(axis=1)].copy()

            # show a warning to let the user know of the droppped nans
            if len(df1) < len(self.df):
                warning_message = 'There are NaNs in the dataset. After removing NaNs, the rows reduce from {} to {}'.format(
                    len(self.df),
                    len(df1))
                warnings.warn(warning_message)
                print(warning_message)

                self.warnings = self.warnings + '\n' + warning_message
        else:
            # get rid of nans. There should be no nans. Imputation should be performed prior to this point
            df1 = df[~df.isna().any(axis=1)].copy()

            # show a warning to let the user know of the droppped nans
            if len(df1) < len(df):
                warning_message = 'There are NaNs in the dataset. After removing NaNs, the rows reduce from {} to {}'.format(
                    len(df),
                    len(df1))
                warnings.warn(warning_message)
                print(warning_message)

                self.warnings = self.warnings + '\n' + warning_message

        # check that the response is in the correct format to perform linear regression
        if not self.linear_regression_utility_check_response(df1[self.response]):
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
        remaining = list(set(df1.columns) - {self.response} - set(self.cols_to_keep_static_dummified))

        # this holds the tried and successful feature set
        full_feature_set = self.cols_to_keep_static_dummified

        # get the first linear regression output for only the constant/base model
        first_result = self.lin_reg_basic(df1[[self.response]])

        # save the model and the X and y used to train it
        self.X_with_feature_selection = self.X.copy()
        self.y_with_feature_selection = self.y.copy()
        self.model_with_feature_selection = self.basic_model


        # get the r2 of the base model
        rsquared = first_result.rsquared

        # store the result of the first model
        final_result = first_result

        # while there are still remaining features to try keep looping
        while len(remaining) > 0:
            # store the last pseudo r2 value
            last_rsquared = rsquared

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
                    result = self.lin_reg_basic(df1[this_feature_set + [self.response]])
                except Exception as e:
                    remaining.remove(col)
                    continue

                # the resulting r2 from this fit
                this_rsquared = result.rsquared

                # if a feature results in nan for r2 skip it
                if this_rsquared is np.nan:
                    print('Note: Feature {} is resulting with a nan r2. Skipping feature'.format(col))
                    continue

                # this feature is recorded as a candidate if the conditions are met
                if (this_rsquared > last_rsquared) and (result.pvalues.loc[col] <= self.sig_level):
                    last_rsquared = this_rsquared
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
                print('********Adding {} with prsquared = {}********'.format(next_col, last_rsquared))

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

    def lin_reg_one_at_a_time(self,with_feature_selection=False,get_interpretability=False):

        dic = dict()
        df1 = self.df.copy()
        df1 = df1[[self.response]+self.cols_to_keep_static + self.cols_to_try_individually].copy()

        for this_col_to_try in self.cols_to_try_individually:
            if with_feature_selection:
                result = self.lin_reg_with_feature_selection(df=df1[self.cols_to_keep_static + [self.response, this_col_to_try]])
                if get_interpretability:
                    self.get_interpretation(self.result_with_feature_selection,self.final_feature_set
                                            ,df=df1[self.cols_to_keep_static + [self.response, this_col_to_try]])
            else:
                result = self.lin_reg(df=df1[self.cols_to_keep_static + [self.response,this_col_to_try]])
                if get_interpretability:
                    self.get_interpretation(self.result, self.X.columns
                                            , df=df1[self.cols_to_keep_static + [self.response, this_col_to_try]])
            for col in list(filter(lambda x: this_col_to_try in x,result.params.index)):
                if get_interpretability:
                    dic[col] = [result.params[col],result.pvalues[col],self.feature_interpretability_df['Probability'][col],
                                self.feature_interpretability_df['Probability']['base']]
                else:
                    dic[col] = [result.params[col], result.pvalues[col]]

        df_one_at_a_time = pd.DataFrame(dic).T
        if get_interpretability:
            df_one_at_a_time.columns = ['Coefficient','Pvalue','Controlled Probability','Base Probability']
        else:
            df_one_at_a_time.columns = ['Coefficient','Pvalue']
        self.df_one_at_a_time = df_one_at_a_time
        return df_one_at_a_time


def unit_test_1():
    print('Unit test 1...')
    import sys
    import os
    import warnings

    np.random.seed(101)
    #warnings.filterwarnings("ignore")

    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare']]
    df = df.dropna()
    my_linear_regresion_class = LinearRegressionClass(df,'Fare',sig_level=0.05)
    my_linear_regresion_class.lin_reg_basic()

    result_required =  [110.08, 3.74, -35.75, 2.54, -0.17, 5.51, 10.21]
    result_actual = list(my_linear_regresion_class.basic_result.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    result_required = 34.69
    result_actual = my_linear_regresion_class.basic_result.predict(my_linear_regresion_class.X).mean()

    result_required = round(result_required, 2)
    result_actual = round(result_actual, 2)

    assert (result_required == result_actual)

    result_required = '''Performance
5-Fold Cross Validation Results:
Test Set r2 = 0.36
neg_mean_squared_error = -1812.52
neg_root_mean_squared_error = -41.66'''
    result_actual = my_linear_regresion_class.lin_reg_diagnostic_performance(my_linear_regresion_class.X,
                                                               my_linear_regresion_class.y)

    assert (result_required == result_actual)

    result_required = [34.69, 38.44, -1.05, 37.23, 34.52, 40.21, 44.9]
    result_actual = list(my_linear_regresion_class.get_interpretation()['Probability'])

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (result_required == result_actual)

    result_required = [-1.05, 34.52, 37.23, 38.44, 40.21, 44.9]
    result_actual = sorted(list(my_linear_regresion_class.feature_interpretability_comparative_df['Prediction_add1_or_with']))

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
    my_linear_regresion_class = LinearRegressionClass(df,'Survived',sig_level=0.05)
    my_linear_regresion_class.lin_reg()

    result_required =  [0.88, -0.19, 0.49, -0.01, -0.05, -0.01, 0.0]
    result_actual = list(my_linear_regresion_class.basic_result.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    result_required = 0.4061624649859944
    result_actual = my_linear_regresion_class.basic_result.predict(my_linear_regresion_class.X).mean()

    result_required = round(result_required, 2)
    result_actual = round(result_actual, 2)

    assert (result_required == result_actual)

    result_required = '''Performance
5-Fold Cross Validation Results:
Test Set r2 = 0.36
neg_mean_squared_error = -0.15
neg_root_mean_squared_error = -0.39'''

    result_actual = my_linear_regresion_class.lin_reg_diagnostic_performance(my_linear_regresion_class.X,
                                                                               my_linear_regresion_class.y)

    assert (result_required == result_actual)

    result_required = [0.41, 0.21, 0.89, 0.4, 0.35, 0.39, 0.41]
    result_actual = list(my_linear_regresion_class.get_interpretation()['Probability'])

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
    my_linear_regresion_class = LinearRegressionClass(df,'Fare',sig_level=0.05)
    my_linear_regresion_class.lin_reg()

    result_required = [112.61, 3.74, -35.75, -0.17, 5.51, 10.21, -2.54]
    result_actual = list(my_linear_regresion_class.result.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    result_required = 34.69
    result_actual = my_linear_regresion_class.result.predict(my_linear_regresion_class.X).mean()

    result_required = round(result_required, 2)
    result_actual = round(result_actual, 2)

    assert (result_required == result_actual)

    result_required = '''Performance
5-Fold Cross Validation Results:
Test Set r2 = 0.36
neg_mean_squared_error = -1812.52
neg_root_mean_squared_error = -41.66'''

    result_actual = my_linear_regresion_class.lin_reg_diagnostic_performance(my_linear_regresion_class.X,
                                                                               my_linear_regresion_class.y)

    assert (result_required == result_actual)

    result_required = [34.69, 38.44, -1.05, 33.77, 34.52, 40.21, 44.9]
    result_actual = list(my_linear_regresion_class.get_interpretation()['Probability'])

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (result_required == result_actual)

    result_required = [-1.05, 33.77, 34.52, 38.44, 40.21, 44.9]
    result_actual = sorted(list(my_linear_regresion_class.feature_interpretability_comparative_df['Prediction_add1_or_with']))

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (result_required == result_actual)

    result_required = [1.0, 0.83, -2.94, -3.65, -4.17, -4.59, -8.48, -8.77, -10.16]
    result_actual = sorted(list(my_linear_regresion_class.predict_from_original(df)))[:-10:-1]

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
    #warnings.filterwarnings("ignore")

    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare']]
    df = df.dropna()
    my_linear_regresion_class = LinearRegressionClass(df,'Fare',sig_level=0.05)
    my_linear_regresion_class.lin_reg_with_feature_selection(verbose=False)

    result_required = [106.7, -35.73, 11.05, 6.15]
    result_actual = list(my_linear_regresion_class.result_with_feature_selection.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    result_required = 34.69
    result_actual = my_linear_regresion_class.result_with_feature_selection.predict(my_linear_regresion_class.X_with_feature_selection).mean()

    result_required = round(result_required, 2)
    result_actual = round(result_actual, 2)

    assert (result_required == result_actual)

    result_required = '''Performance
5-Fold Cross Validation Results:
Test Set r2 = 0.37
neg_mean_squared_error = -1798.9
neg_root_mean_squared_error = -41.44'''

    result_actual = my_linear_regresion_class.lin_reg_diagnostic_performance(my_linear_regresion_class.X_with_feature_selection,
                                                                               my_linear_regresion_class.y_with_feature_selection)

    assert (result_required == result_actual)

    result_required = [34.69, -1.04, 40.84, 45.75]
    result_actual = list(my_linear_regresion_class.get_interpretation()['Probability'])

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (result_required == result_actual)

    result_required = [14.62, 4.81, 4.81, 4.81, -1.34, -1.34, -7.48, -7.48, -7.48]
    result_actual = sorted(list(my_linear_regresion_class.predict_from_original(df)))[:-10:-1]

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (result_required == result_actual)

    print('Success!')

def unit_test_5():
    print('Unit test 5...')
    import sys
    import os
    import warnings

    np.random.seed(101)
    #warnings.filterwarnings("ignore")

    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare']]
    df.loc[1,'Survived'] = np.nan
    my_linear_regresion_class = LinearRegressionClass(df,'Survived',sig_level=0.05)
    my_linear_regresion_class.lin_reg()

    result_required = [1.36, -0.19, -0.01, -0.05, -0.01, 0.0, -0.49]
    result_actual = list(my_linear_regresion_class.result.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))


    print('Success!')

def unit_test_6():
    print('Unit test 6...')
    import sys
    import os
    import warnings

    np.random.seed(101)
    #warnings.filterwarnings("ignore")

    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare']]
    df.loc[1,'Survived'] = np.nan
    for col in df.columns:
        if col in ['Pclass','Parch']:
            df[col] = df[col].astype('str')
    my_linear_regresion_class = LinearRegressionClass(df,'Fare',sig_level=0.05,cols_to_keep_static=['Pclass'])
    my_linear_regresion_class.lin_reg_with_feature_selection(verbose=False)

    result_required = [110.47, -66.41, -75.03, -30.54, -14.45, 4.2]
    result_actual = list(my_linear_regresion_class.result_with_feature_selection.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    print('Success!')


def unit_test_7():
    print('Unit test 7...')
    import sys
    import os
    import warnings

    np.random.seed(101)
    #warnings.filterwarnings("ignore")

    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare']]
    for col in df.columns:
        if col in ['Pclass','Parch']:
            df[col] = df[col].astype('str')
    my_linear_regresion_class = LinearRegressionClass(df,'Fare',sig_level=0.05,cols_to_keep_static=['Pclass'],
                                                          cols_to_try_individually=['Parch','Sex','Age','Fare'],
                                                          max_iter=1000)
    my_linear_regresion_class.lin_reg_one_at_a_time(with_feature_selection=True)

    result_required = [-36.41, -17.25, 22.02, 34.63, -0.51]
    result_actual = list(my_linear_regresion_class.df_one_at_a_time['Coefficient'])

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    print('Success!')

def unit_test_8():
    print('Unit test 8...')
    import sys
    import os
    import warnings

    np.random.seed(101)
    #warnings.filterwarnings("ignore")

    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare']]
    for col in df.columns:
        if col in ['Pclass','Parch']:
            df[col] = df[col].astype('str')
    my_linear_regresion_class = LinearRegressionClass(df,'Survived',sig_level=0.05,cols_to_keep_static=['Pclass'],
                                                          cols_to_try_individually=['Age','Fare'],
                                                          max_iter=1000)
    my_linear_regresion_class.lin_reg_one_at_a_time(with_feature_selection=False)

    result_required = [-0.01, 0.0]
    result_actual = list(my_linear_regresion_class.df_one_at_a_time['Coefficient'])

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    print('Success!')

def unit_test_9():
    print('Unit test 9...')
    import sys
    import os
    import warnings

    np.random.seed(101)
    #warnings.filterwarnings("ignore")

    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare']]
    for col in df.columns:
        if col in ['Pclass','Parch']:
            df[col] = df[col].astype('str')
    my_linear_regresion_class = LinearRegressionClass(df,'Survived',sig_level=0.05,cols_to_keep_static=['Pclass'],
                                                          cols_to_try_individually=['Age','Fare'],
                                                          max_iter=1000)
    my_linear_regresion_class.lin_reg_one_at_a_time(with_feature_selection=False,get_interpretability=True)

    result_required = [0.4, 0.39]
    result_actual = list(my_linear_regresion_class.df_one_at_a_time['Controlled Probability'])

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    print('Success!')

def unit_test_10():
    print('Unit test 10...')
    import sys
    import os
    import warnings

    np.random.seed(101)
    #warnings.filterwarnings("ignore")

    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare']]
    for col in df.columns:
        if col in ['Pclass','Parch']:
            df[col] = df[col].astype('str')
    my_linear_regresion_class = LinearRegressionClass(df,'Survived',sig_level=0.05,cols_to_keep_static=['Pclass'],
                                                          cols_to_try_individually=['Age','Fare','Parch','Sex'],
                                                          max_iter=1000)
    my_linear_regresion_class.lin_reg_one_at_a_time(with_feature_selection=True,get_interpretability=True)

    result_required = [0.4, 0.39, 0.35, 0.04, 0.72, 0.2]
    result_actual = list(my_linear_regresion_class.df_one_at_a_time['Controlled Probability'])

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    print('Success!')

if __name__ == '__main__':
    unit_test_1()
    unit_test_2()
    unit_test_3()
    unit_test_4()
    unit_test_5()
    unit_test_6()
    unit_test_7()
    unit_test_8()
    unit_test_9()
    unit_test_10()
