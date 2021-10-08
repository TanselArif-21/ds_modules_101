import pandas as pd
import numpy as np
import seaborn as sns
import LogisticRegression
import MultinomialLogisticRegression
import os
import sys

class MicrosimulationClass:
    '''
    A class to help in running simulations based on given predictive algorithms to project populations forward in time.

    Example usage:
    # imports
    from ds_modules_101 import Models as dsm
    from ds_modules_101 import Data as dsd

    # get all the data
    df = dsd.hr_df

    # get the data for the last time period the we want to start projecting
    df_last_time = df[(df['Year'] == 2019) & (df['left'] != 1)].copy()

    # train a logistic regression algorithm
    predictors = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours',
                  'time_spend_company', 'Work_accident', 'promotion_last_5years', 'sales', 'salary']
    model_left = dsm.LogisticRegressionClass(df[predictors+['left']],'left')
    model_left.log_reg()
    #### Training complete

    # define a 'predict_next' method and attach it to the trained model (this must always be done
    # the simulation will ask the model to predict on a df so this method should be able to take in a non dummified
    # dataframe - i.e. the original dataframe format - and provide predictions)
    def predict(model,df):
        preds = model.predict_from_original(df)
        a = []
        for p in preds:
            a.append(np.random.choice([0,1],1,p=[1-p,p]))
        return np.squeeze(a)
    model_left.predict_next = predict

    # create another function which tells the simulation how to update the time step
    def data_time_update_function(df):
        df['Year'] = df['Year'] + 1
        df = df[df['left'] == 0]
        return df

    # create the microsimulation object by specifying the original df, the df for the last time step and the
    # function to update the dataframe to the next time step
    my_microsimulation = MicrosimulationClass(df,df_last_time,data_time_update_function=data_time_update_function)

    # attach a model to the simulation along with what it is trying to predict - in this case the 'left' flag
    my_microsimulation.add_model_beginning_period(model_left,'left')

    # advance the population by 4 years/time steps
    my_microsimulation.advance()
    my_microsimulation.advance()
    my_microsimulation.advance()
    my_microsimulation.advance()
    '''
    def __init__(self,df,df_last_time,models_beginning_period=None,models_beginning_period_responses=None,
                 models_beginning_period_predictors=None,models_beginning_period_transforms=None,
                 models_end_period=None, models_end_period_responses=None,
                 models_end_period_predictors=None, models_end_period_transforms=None,
                 data_time_update_function = None):
        '''
        :param df: a dataframe
        '''

        # attach attributes to the object
        self.df = df.copy()
        self.df_last_time = df_last_time.copy()

        self.dfs_projected = [self.df_last_time.copy()]

        self.models_beginning_period = []
        self.models_beginning_period_responses = []
        self.models_beginning_period_predictors = []
        self.models_beginning_period_transforms = []

        if models_beginning_period is not None:
            self.models_beginning_period = models_beginning_period
            if models_beginning_period_responses is not None:
                self.models_beginning_period_responses = models_beginning_period_responses
            else:
                raise Exception('Model is given but model responses is not')

            if models_beginning_period_predictors is not None:
                self.models_beginning_period_predictors = models_beginning_period
            else:
                raise Exception('Model is given but model predictors is not')

            if models_beginning_period_transforms is not None:
                self.models_beginning_period_transforms = models_beginning_period_transforms
            else:
                raise Exception('Model is given but model transforms is not')

        self.models_end_period = []
        self.models_end_period_responses = []
        self.models_end_period_predictors = []
        self.models_end_period_transforms = []

        if models_end_period is not None:
            self.models_end_period = models_end_period
            if models_end_period_responses is not None:
                self.models_end_period_responses = models_end_period_responses
            else:
                raise Exception('Model is given but model responses is not')

            if models_end_period_predictors is not None:
                self.models_end_period_predictors = models_beginning_period
            else:
                raise Exception('Model is given but model predictors is not')

            if models_end_period_transforms is not None:
                self.models_end_period_transforms = models_end_period_transforms
            else:
                raise Exception('Model is given but model transforms is not')

        if data_time_update_function is not None:
            MicrosimulationClass.data_time_update_function = data_time_update_function

    def add_model_beginning_period(self,model,response,predictors=None,transform=None):
        self.models_beginning_period.append(model)
        self.models_beginning_period_responses.append(response)
        self.models_beginning_period_predictors.append(predictors)
        self.models_beginning_period_transforms.append(transform)

    def add_model_end_period(self,model,response,predictors=None,transform=None):
        self.models_end_period.append(model)
        self.models_end_period_responses.append(response)
        self.models_end_period_predictors.append(predictors)
        self.models_end_period_transforms.append(transform)

    def data_time_update_function(df):
        return df

    def advance(self):
        next_df = self.dfs_projected[-1].copy()
        for model,response,predictors,transform in zip(self.models_beginning_period,
                                                       self.models_beginning_period_responses,
                                                       self.models_beginning_period_predictors,
                                                       self.models_beginning_period_transforms):
            this_next_df = next_df.copy()
            if transform is not None:
                this_next_df = transform(next_df,predictors)
            if predictors is not None:
                next_df[response] = model.predict_next(model,this_next_df[predictors])
            else:
                next_df[response] = model.predict_next(model, this_next_df)

        next_df = MicrosimulationClass.data_time_update_function(next_df)

        for model,response,predictors,transform in zip(self.models_end_period,
                                                       self.models_end_period_responses,
                                                       self.models_end_period_predictors,
                                                       self.models_end_period_transforms):
            this_next_df = next_df.copy()
            if transform is not None:
                this_next_df = transform(next_df,predictors)
            if predictors is not None:
                next_df[response] = model.predict_next(model,this_next_df[predictors])
            else:
                next_df[response] = np.array(model.predict_next(model, this_next_df))

        self.dfs_projected.append(next_df.copy())



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'HR')
    hr_csv = os.path.join(data_dir, 'HR.csv')
    df = pd.read_csv(hr_csv)
    df_last_time = df[(df['Year'] == 2019) & (df['left'] != 1)].copy()

    ######### Logistic regression to predict leavers
    predictors = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours',
                  'time_spend_company', 'Work_accident', 'promotion_last_5years', 'sales', 'salary']
    model_left = LogisticRegression.LogisticRegressionClass(df[predictors+['left']],'left')
    model_left.log_reg()

    def predict(model,df):
        preds = model.predict_from_original(df)
        a = []
        for p in preds:
            a.append(np.random.choice([0,1],1,p=[1-p,p]))
        return np.squeeze(a)

    model_left.predict_next = predict
    ################################################





    def data_time_update_function(df):
        df['time_spend_company'] = df['time_spend_company'] + 1
        df['Year'] = df['Year'] + 1

        probability_level_of_external_entry = 0.75
        this_probability = np.random.rand(len(df[df['left'] == 1]))
        idx_copy_list = []
        for idx,prob in zip(list(df[df['left'] == 1].index),this_probability):
            sales = df.loc[idx, 'sales']
            if prob <= probability_level_of_external_entry:
                df.loc[idx, :] = None
                df.loc[idx, ['sales','left','time_spend_company','promotion_last_5years']] = [sales,0,0,0]
            else:
                idx_copy = np.random.choice(list(set(df[df['left'] == 0].index) - set([idx])))
                df.loc[idx, :] = df.loc[idx_copy, :].copy()
                idx_copy_list.append(idx_copy)

        df = df[df['left'] == 0].copy()

        df.loc[idx_copy_list, 'left'] = 1

        return df

    ######### Estimate other values
    def predict_value(value,category=False,agg_type=0):
        def predict_this_value(model,df):
            df_prev = df.copy()
            df_prev['time_spend_company'] = df_prev['time_spend_company'] - 1
            t = df_prev[['sales','time_spend_company',value]].copy()
            if category:
                t = t[~pd.isna(t[value])].groupby(by=['sales', 'time_spend_company']).agg(pd.Series.mode).reset_index()
            else:
                if agg_type == 0:
                    t = t[~pd.isna(t[value])].groupby(by=['sales','time_spend_company']).mean().reset_index()
                elif agg_type == 1:
                    t = t[~pd.isna(t[value])].groupby(by=['sales', 'time_spend_company']).min().reset_index()
                else:
                    t = t[~pd.isna(t[value])].groupby(by=['sales', 'time_spend_company']).max().reset_index()

            if value == 'salary':
                t[value] = 'low'

            df = pd.merge(left=df,right=t,on=['sales','time_spend_company'],suffixes=['','_y'],how='left')

            t = df_prev[['sales', value]].copy()
            if category:
                t = t[~pd.isna(t[value])].groupby(by=['sales']).agg(pd.Series.mode).reset_index()
            else:
                if agg_type == 0:
                    t = t[~pd.isna(t[value])].groupby(by=['sales']).mean().reset_index()
                elif agg_type == 1:
                    t = t[~pd.isna(t[value])].groupby(by=['sales']).min().reset_index()
                else:
                    t = t[~pd.isna(t[value])].groupby(by=['sales']).max().reset_index()

            if value == 'salary':
                t[value] = 'low'

            df = pd.merge(left=df, right=t, on=['sales'], suffixes=['', '_y2'],how='left')

            df[value] = df[[value, value + '_y', value + '_y2']].apply(
                lambda x: x[2] if (pd.isna(x[0]) and pd.isna(x[1])) else x[1] if pd.isna(x[0]) else x[0], axis=1)

            return df[value]
        return predict_this_value

    value_models = []
    values = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours',
                  'Work_accident', 'promotion_last_5years', 'salary','Year']
    class dummy_class:
        def __init__(self):
            pass

    for value in values:
        category = False
        this_model = dummy_class()
        if value in ['salary','number_project','Work_accident']:
            category = True
        this_model.predict_next = predict_value(value, category=category,agg_type=1)
        value_models.append(this_model)
    ################################################


    my_microsimulation = MicrosimulationClass(df,df_last_time,data_time_update_function=data_time_update_function)
    my_microsimulation.add_model_beginning_period(model_left,'left')
    for model,value in zip(value_models,values):
        my_microsimulation.add_model_end_period(model,value)
    my_microsimulation.advance()
    my_microsimulation.advance()
    my_microsimulation.advance()
    my_microsimulation.advance()
    my_microsimulation.advance()
    my_microsimulation.advance()
    final_df = pd.concat(my_microsimulation.dfs_projected, axis=0,ignore_index=True).reset_index(drop=True)
    final_df_grouped = final_df[['Year', 'satisfaction_level']].groupby(by='Year').mean().reset_index()
    sns.lineplot(x=final_df_grouped['Year'], y=final_df_grouped['satisfaction_level'])
    plt.show()
    final_df_grouped = final_df[['Year', 'number_project']].groupby(by='Year').mean().reset_index()
    sns.lineplot(x=final_df_grouped['Year'], y=final_df_grouped['number_project'])
    plt.show()
    final_df['high_salary'] = (final_df['salary'] == 'high').astype('int')
    final_df_grouped = final_df[['Year', 'high_salary']].groupby(by='Year').mean().reset_index()
    sns.lineplot(x=final_df_grouped['Year'], y=final_df_grouped['high_salary'])
    plt.show()
    final_df_grouped = final_df[['Year', 'time_spend_company']].groupby(by='Year').mean().reset_index()
    sns.lineplot(x=final_df_grouped['Year'], y=final_df_grouped['time_spend_company'])
    plt.show()
    a=1