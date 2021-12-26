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

class LogisticRegressionClass:
    def __init__(self,df,response,sig_level=0.05,max_iter=500,cols_to_keep_static=[],cols_to_try_individually=[],
                 regularization_C=None):
        '''
        :param df: a dataframe
        :param response: a string. This must be an existing column in df
        :param sig_level: a float. The significance level the forward selection will use
        :param max_iter: an integer. The maximum iterations the solvers will use to try to converge
        :param cols_to_keep_static: a list. Used in forward selection to not omit these columns
        :param cols_to_try_individually: a list. The columns to test in a regression one at a time to identify which
            one has the greatest relationship with the response controlled for the cols_to_keep_static
        :param C: Regularisation contant for the l1 regulatisation. The weight multiplying the penalty term
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
        self.regularization_C = regularization_C
        self.exception_message = None

        if regularization_C is None:
            self.sklearn_model = LogisticRegression(max_iter=self.max_iter)
        else:
            self.sklearn_model = LogisticRegression(max_iter=self.max_iter, penalty='l1',
                                                    C=1 / (self.regularization_C + 0.000000001), solver='liblinear')

    def prepare_data(self,df,response):
        y = df[response]
        X = df[list(filter(lambda x: x != response, df.columns))]
        X = sm.add_constant(X, has_constant='add')

        return X, y

    def logistic_regression_utility_check_response(self,series):
        if (len(series.unique()) > 2):
            self.error_message = 'The response variable has more than 2 categories and is not suitable for logistic regression'
            print('The response variable has more than 2 categories and is not suitable for logistic regression')
            return False

        if (not is_numeric_dtype(series)):
            self.error_message = self.error_message + '\n' + 'The response variable should be binary 0 and 1 and numeric type (i.e. int)'
            print('The response variable should be binary 0 and 1 and numeric type (i.e. int)')
            return False

        return True

    def log_reg_diagnostic_performance(self,X=None,y=None):
        if X is None:
            try:
                X = self.X_with_feature_selection
                y = self.y_with_feature_selection
            except:
                X = self.X
                y = self.y

        # cvs = cross_validate(LogisticRegression(max_iter=self.max_iter), X, y, cv=5,
        #                      scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'])

        cvs = cross_validate(self.sklearn_model, X, y, cv=5,
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

    def log_reg_diagnostic_correlations(self,X=None):
        print("Correlations")

        if X is None:
            try:
                X = self.X_with_feature_selection
            except:
                X = self.X

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)

        upp_mat = np.triu(X.corr())
        sns.heatmap(X.corr(), vmin=-1, vmax=+1, annot=True, cmap='coolwarm', mask=upp_mat, ax=ax)

        self.fig_correlations = fig
        self.ax_correlations = ax

        return fig,ax

    def logistic_regression_get_report(self,model=None,X=None,y=None,verbose=True):
        if model is None:
            try:
                model = self.result_with_feature_selection
            except:
                model = self.result

        if X is None:
            try:
                X = self.X_with_feature_selection
                y = self.y_with_feature_selection
            except:
                X = self.X
                y = self.y

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

    def log_reg_basic(self,df=None):
        '''
        Run a basic logistic regression model
        '''

        if df is None:
            df = self.df

        X, y = self.prepare_data(df, self.response)

        model = sm.Logit(y, X)

        if self.regularization_C is None:
            result = model.fit(disp=0,maxiter=self.max_iter)
        else:
            result = model.fit_regularized(disp=0, maxiter=self.max_iter,alpha=self.regularization_C)

        self.basic_result = result
        self.basic_model = model
        self.X = X
        self.y = y

        return result

    def predict_from_original(self,df):
        df = self.prepare_categories(df, self.response, drop=False)

        try:
            all_cols = list(self.X_with_feature_selection.columns)
        except:
            all_cols = list(self.X.columns)

        if 'const' not in df.columns:
            df['const'] = 1

        for col in all_cols:
            if col not in df.columns:
                df[col] = 0

        try:
            res = self.result_with_feature_selection
        except:
            res = self.result

        return res.predict(df[all_cols])

    def log_reg(self,df=None):
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

        if not self.logistic_regression_utility_check_response(df1[self.response]):
            return None

        df1 = self.prepare_categories(df1,self.response,drop=True)

        result = self.log_reg_basic(df1)

        self.result = result
        self.model = self.basic_model

        return result

    def log_reg_with_feature_selection(self,df=None,run_for=0,verbose=True,max_pr2=None,max_features=None):
        import warnings
        self.params_with_convergence_errors = []
        # start the timer in case the is a time limit specified
        start_time = time.time()
        n_features = 0

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
        remaining = list(set(df1.columns) - {self.response} - set(self.cols_to_keep_static_dummified))

        # this holds the tried and successful feature set
        full_feature_set = self.cols_to_keep_static_dummified

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
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        result = self.log_reg_basic(df1[this_feature_set + [self.response]])
                except Exception as e:
                    self.exception_message = e
                    self.params_with_convergence_errors.append(col)
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

            n_features = n_features+1

            # show progress
            if verbose:
                print('********Adding {} with prsquared = {}********'.format(next_col, last_prsquared))

            # store the result
            final_result = next_result

            # remove the chosen candidate from the remaining features
            remaining.remove(next_col)

            # if the user has specified a max r2 then stop if it has been reached
            if (max_pr2 is not None) and (max_pr2 <= last_prsquared):
                break

            # if the user has specified a max number of features then stop if it has been reached
            if (max_features is not None) and (max_features <= n_features):
                break

            # check if it's not taking too long
            if (time.time() - start_time > run_for) and (run_for > 0):
                print(
                    'Aborting: Has been running for {}s > {}s. {} out of {} columns left. There are probably too many categories in one of the columns'.format(
                        round(time.time() - start_time, 2), run_for, len(remaining), len(df1.columns) - 1))
                return

        self.final_feature_set = full_feature_set
        self.result_with_feature_selection = final_result
        if (len(self.params_with_convergence_errors) > 0) & verbose:
            print('There were converge errors. See params_with_convergence_errors for the list of columns')

        return final_result

    def log_reg_one_at_a_time(self,with_feature_selection=False,get_interpretability=False):

        dic = dict()
        df1 = self.df.copy()
        df1 = df1[[self.response]+self.cols_to_keep_static + self.cols_to_try_individually].copy()

        for this_col_to_try in self.cols_to_try_individually:
            if with_feature_selection:
                result = self.log_reg_with_feature_selection(df=df1[self.cols_to_keep_static + [self.response, this_col_to_try]])
                if get_interpretability:
                    self.get_interpretation(self.result_with_feature_selection,self.final_feature_set
                                            ,df=df1[self.cols_to_keep_static + [self.response, this_col_to_try]])
            else:
                result = self.log_reg(df=df1[self.cols_to_keep_static + [self.response,this_col_to_try]])
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
    result_actual = my_logistic_regresion_class.log_reg_diagnostic_performance()

    assert (result_required == result_actual)

    result_required = [0.4061624649859944, 0.24581360407372246, 0.795820946281563, 0.3999162261394402, 0.3539768140703711, 0.39737068898845873, 0.4064703482674913]
    result_actual = list(my_logistic_regresion_class.get_interpretation()['Probability'])

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (result_required == result_actual)

    result_required = [0.22395117322839098, 0.3558311002657131, 0.3977261128754439, 0.4027792667251947,
                       0.4068054774641371, 0.8543626808843827]
    result_actual = sorted(list(my_logistic_regresion_class.feature_interpretability_comparative_df['Prediction_add1_or_with']))

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
    result_actual = list(my_logistic_regresion_class.get_interpretation()['Probability'])

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
    result_actual = list(my_logistic_regresion_class.get_interpretation()['Probability'])

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (result_required == result_actual)

    result_required = [0.22395117322839098, 0.23089275089703268, 0.35583110026571313,
                       0.3977261128754439, 0.4027792667251947, 0.4068054774641371]
    result_actual = sorted(list(my_logistic_regresion_class.feature_interpretability_comparative_df['Prediction_add1_or_with']))

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (result_required == result_actual)

    result_required = [0.98, 0.98, 0.98, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97]
    result_actual = sorted(list(my_logistic_regresion_class.predict_from_original(df)))[:-10:-1]

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
    result_actual = list(my_logistic_regresion_class.get_interpretation()['Probability'])

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (result_required == result_actual)

    result_required = [365, 59, 78, 212]
    result_actual = my_logistic_regresion_class.logistic_regression_get_report(verbose=False)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), list(np.array(result_actual).flatten())))

    assert (result_required == result_actual)

    result_required = [0.98, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97]
    result_actual = sorted(list(my_logistic_regresion_class.predict_from_original(df)))[:-10:-1]

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
    my_logistic_regresion_class = LogisticRegressionClass(df,'Survived',sig_level=0.05)
    my_logistic_regresion_class.log_reg()

    result_required = [5.38, -1.24, -0.04, -0.38, -0.06, 0.0, -2.63]
    result_actual = list(my_logistic_regresion_class.result.params)

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
    my_logistic_regresion_class = LogisticRegressionClass(df,'Survived',sig_level=0.05,cols_to_keep_static=['Pclass'])
    my_logistic_regresion_class.log_reg_with_feature_selection(verbose=False)

    result_required = [1.7, -1.41, -2.65, 2.62, -0.04, -0.38]
    result_actual = list(my_logistic_regresion_class.result_with_feature_selection.params)

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
    my_logistic_regresion_class = LogisticRegressionClass(df,'Survived',sig_level=0.05,cols_to_keep_static=['Pclass'],
                                                          cols_to_try_individually=['Parch','Sex','Age','Fare'],
                                                          max_iter=1000)
    my_logistic_regresion_class.log_reg_one_at_a_time(with_feature_selection=True)

    result_required = [-0.73, 2.64, -0.04, 0.01]
    result_actual = list(my_logistic_regresion_class.df_one_at_a_time['Coefficient'])

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
    my_logistic_regresion_class = LogisticRegressionClass(df,'Survived',sig_level=0.05,cols_to_keep_static=['Pclass'],
                                                          cols_to_try_individually=['Age','Fare'],
                                                          max_iter=1000)
    my_logistic_regresion_class.log_reg_one_at_a_time(with_feature_selection=False)

    result_required = [-0.04, 0.01]
    result_actual = list(my_logistic_regresion_class.df_one_at_a_time['Coefficient'])

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
    my_logistic_regresion_class = LogisticRegressionClass(df,'Survived',sig_level=0.05,cols_to_keep_static=['Pclass'],
                                                          cols_to_try_individually=['Age','Fare'],
                                                          max_iter=1000)
    my_logistic_regresion_class.log_reg_one_at_a_time(with_feature_selection=False,get_interpretability=True)

    result_required = [0.4, 0.39]
    result_actual = list(my_logistic_regresion_class.df_one_at_a_time['Controlled Probability'])

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
    my_logistic_regresion_class = LogisticRegressionClass(df,'Survived',sig_level=0.05,cols_to_keep_static=['Pclass'],
                                                          cols_to_try_individually=['Age','Fare','Parch','Sex'],
                                                          max_iter=1000)
    my_logistic_regresion_class.log_reg_one_at_a_time(with_feature_selection=True,get_interpretability=True)

    result_required = [0.4, 0.39, 0.35, 0.72]
    result_actual = list(my_logistic_regresion_class.df_one_at_a_time['Controlled Probability'])

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    print('Success!')


def unit_test_11():
    print('Unit test 11...')
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
    my_logistic_regresion_class = LogisticRegressionClass(df,'Survived',sig_level=0.05,cols_to_keep_static=['Pclass'])
    my_logistic_regresion_class.log_reg_with_feature_selection(verbose=True,max_pr2=0.32)

    result_required = [1.25, -1.31, -2.58, 2.52, -0.04]
    result_actual = list(my_logistic_regresion_class.result_with_feature_selection.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    print('Success!')

def unit_test_12():
    print('Unit test 12...')
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
    my_logistic_regresion_class = LogisticRegressionClass(df,'Survived',sig_level=0.05,cols_to_keep_static=['Pclass'])
    my_logistic_regresion_class.log_reg_with_feature_selection(verbose=True,max_features=2)

    result_required = [1.25, -1.31, -2.58, 2.52, -0.04]
    result_actual = list(my_logistic_regresion_class.result_with_feature_selection.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    print('Success!')

def unit_test_13():
    print('Unit test 13...')
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
    my_logistic_regresion_class = LogisticRegressionClass(df,'Survived',sig_level=0.05,cols_to_keep_static=['Pclass'])
    my_logistic_regresion_class.log_reg_with_feature_selection(verbose=True,max_features=2,max_pr2=0.3)

    result_required = [-0.24, -0.92, -1.97, 2.57]
    result_actual = list(my_logistic_regresion_class.result_with_feature_selection.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    print('Success!')


def unit_test_14():
    print('Unit test 14...')
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
    df['Survived'] = df['Survived'].astype('str')
    my_logistic_regresion_class = LogisticRegressionClass(df,'Survived',sig_level=0.05)
    my_logistic_regresion_class.log_reg()

    result_required = '\nThe response variable should be binary 0 and 1 and numeric type (i.e. int)'
    result_actual = my_logistic_regresion_class.error_message
    assert (result_required == result_actual)


    print('Success!')

def unit_test_15():
    print('Unit test 15...')
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
    df['AllZeros'] = 0
    df['Survived'] = df['Survived'].astype('int')
    my_logistic_regresion_class = LogisticRegressionClass(df,'Survived',sig_level=0.05,max_iter=5)
    my_logistic_regresion_class.log_reg_with_feature_selection()

    result_required = 'Singular matrix'
    result_actual = str(my_logistic_regresion_class.exception_message)
    assert (result_required == result_actual)


    print('Success!')

def unit_test_16():
    print('Unit test 16...')
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
    df['Survived'] = df['Survived'].astype('int')
    my_logistic_regresion_class = LogisticRegressionClass(df,'Survived',sig_level=0.05,max_iter=500,regularization_C=0)
    my_logistic_regresion_class.log_reg()

    result_required = [5.389003106421364, -1.2422486253277716, -0.043952595897772714, -0.3757548705084541, -0.0619373664480337, 0.002160033540727774, -2.6348448348873723]
    result_actual = list(my_logistic_regresion_class.result.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    my_logistic_regresion_class = LogisticRegressionClass(df, 'Survived', sig_level=0.05, max_iter=500,
                                                          regularization_C=2)
    my_logistic_regresion_class.log_reg()

    result_required = [4.405751074517862, -1.019718809617492, -0.035243106198760185, -0.3133464821793585, -0.03593459183804765, 0.0036461299221873813, -2.4156954170686182]
    result_actual = list(my_logistic_regresion_class.result.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    result_required = '''Performance (0 is negative 1 is positive)
5-Fold Cross Validation Results:
Test Set accuracy = 0.78
f1 = 0.72
precision = 0.76
recall = 0.69
auc = 0.85'''

    result_actual = my_logistic_regresion_class.log_reg_diagnostic_performance()

    assert (result_required == result_actual)

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
    unit_test_11()
    unit_test_12()
    unit_test_13()
    unit_test_14()
    unit_test_15()
    unit_test_16()