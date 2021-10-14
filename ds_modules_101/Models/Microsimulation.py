import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys

class MicrosimulationClass:
    '''
    A class to help in running simulations based on given predictive algorithms to project populations forward in time.

    General Usage:
    model1 # A model you have developed to predict a feature (must have a predict_next method which predicts the next
           # dataframe given a dataframe)
    model2 # A model you have developed to predict another feature
    model3 # A model to update the dateframe to the next time step
    model4 # A model to predict another feature
    my_microsimulation = MicrosimulationClass(df) # df is the dataframe to project into the future
    my_microsimulation.add_model(model1, type='predict')
    my_microsimulation.add_model(model2, type='predict')
    my_microsimulation.add_model(model3,type='update')
    my_microsimulation.add_model(model4, type='predict')
    # my_microsimulation.dfs_projected is now a list of projected dataframes for each timestep

    Example usage:
    import matplotlib.pyplot as plt
    # imports
    from ds_modules_101 import Models as dsm
    from ds_modules_101 import Data as dsd

    # get all the data
    df = dsd.hr_df
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

        df[model.response] = np.array(np.squeeze(a))
        return df

    model_left.predict_next = predict
    ################################################

    def data_time_update_function(df):
        df['time_spend_company'] = df['time_spend_company'] + 1
        df['Year'] = df['Year'] + 1

        probability_level_of_external_entry = 0.5
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
            cols = list(df.columns)
            t = df_prev[['sales','time_spend_company',value]].copy()
            if category:
                t = t[~pd.isna(t[value])].groupby(by=['sales', 'time_spend_company']).agg(lambda x: pd.Series.mode(x)[0]).reset_index()
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
                t = t[~pd.isna(t[value])].groupby(by=['sales']).agg(lambda x: pd.Series.mode(x)[0]).reset_index()
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

            return df[cols]
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

    date_time_update_model = dummy_class()
    date_time_update_model.predict_next = data_time_update_function

    my_microsimulation = MicrosimulationClass(df_last_time)
    my_microsimulation.add_model(model_left, type='predict')
    my_microsimulation.add_model(date_time_update_model,type='update')
    for model,value in zip(value_models,values):
        my_microsimulation.add_model(model,type='predict')
    my_microsimulation.advance()
    print('1')
    my_microsimulation.advance()
    print('2')
    my_microsimulation.advance()
    print('3')
    my_microsimulation.advance()
    print('4')
    my_microsimulation.advance()
    print('5')
    my_microsimulation.advance()
    print('6')
    my_microsimulation.advance()
    print('7')
    my_microsimulation.advance()
    print('8')
    my_microsimulation.advance()
    print('9')
    my_microsimulation.advance()
    print('10')
    my_microsimulation.advance()
    print('11')
    my_microsimulation.advance()
    print('12')
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
    '''
    def __init__(self,df_last_time,models=None,models_types=None):
        '''
        :param df: a dataframe
        '''

        # attach attributes to the object
        self.df_last_time = df_last_time.copy()

        self.dfs_projected = [self.df_last_time.copy()]

        self.models = []
        self.models_types = []

        if models is not None:
            self.models = models

            if models_types is not None:
                self.models_types = models_types
            else:
                raise Exception('Model is given but model types is not')

    def add_model(self,model,type,predictors=None):
        self.models.append(model)
        self.models_types.append(type)

    def advance(self):
        next_df = self.dfs_projected[-1].copy()
        for model,type in zip(self.models,self.models_types):
            next_df = self.advance_type(next_df.copy(),model,type)



        self.dfs_projected.append(next_df.copy())

    def advance_type(self,next_df,model,type):
        this_next_df = next_df.copy()
        if type == 'predict':
            next_df = self.advance_predict(this_next_df,model)
        elif type == 'update':
            next_df = self.advance_update(this_next_df,model)

        return next_df

    def advance_predict(self,next_df,model):
        this_next_df = next_df.copy()
        next_df = model.predict_next(model, this_next_df)

        return next_df

    def advance_update(self,next_df,model):
        this_next_df = next_df.copy()
        next_df = model.predict_next(this_next_df)

        return next_df

    def finalise(self):
        self.evolved_df = pd.concat(self.dfs_projected)
        self.summary_df = self.evolved_df.groupby(by='Year').describe()
        self.summary_df.columns = [' '.join(col).strip() for col in self.summary_df.columns.values]
        self.summary_df = self.summary_df.T.reset_index()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import LogisticRegression
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

        df[model.response] = np.array(np.squeeze(a))
        return df

    model_left.predict_next = predict
    ################################################





    def data_time_update_function(df):
        df['time_spend_company'] = df['time_spend_company'] + 1
        df['Year'] = df['Year'] + 1

        probability_level_of_external_entry = 0.5
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
            cols = list(df.columns)
            t = df_prev[['sales','time_spend_company',value]].copy()
            if category:
                t = t[~pd.isna(t[value])].groupby(by=['sales', 'time_spend_company']).agg(lambda x: pd.Series.mode(x)[0]).reset_index()
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
                t = t[~pd.isna(t[value])].groupby(by=['sales']).agg(lambda x: pd.Series.mode(x)[0]).reset_index()
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

            return df[cols]
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

    date_time_update_model = dummy_class()
    date_time_update_model.predict_next = data_time_update_function

    my_microsimulation = MicrosimulationClass(df_last_time)
    my_microsimulation.add_model(model_left, type='predict')
    my_microsimulation.add_model(date_time_update_model,type='update')
    for model,value in zip(value_models,values):
        my_microsimulation.add_model(model,type='predict')
    my_microsimulation.advance()
    print('1')
    my_microsimulation.advance()
    print('2')
    # my_microsimulation.advance()
    # print('3')
    # my_microsimulation.advance()
    # print('4')
    # my_microsimulation.advance()
    # print('5')
    # my_microsimulation.advance()
    # print('6')
    # my_microsimulation.advance()
    # print('7')
    # my_microsimulation.advance()
    # print('8')
    # my_microsimulation.advance()
    # print('9')
    # my_microsimulation.advance()
    # print('10')
    # my_microsimulation.advance()
    # print('11')
    # my_microsimulation.advance()
    # print('12')

    my_microsimulation.finalise()

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