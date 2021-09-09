import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import plotly.graph_objects as go
import os
from pandas.api.types import is_numeric_dtype
import warnings

class Oaxaca:
    '''
    Example usage:
    import ds_modules_101.Models as dsm
    import ds_modules_101.Data as dsd
    df=dsd.titanic_df
    my_oaxaca = dsm.Oaxaca(df, grp_col='Sex', response='Fare', grp_groups=['male', 'female'],
           dummy_cols=['Survived', 'Pclass', 'SibSp','Parch'],
           all_factors=['Survived', 'Pclass', 'Age', 'SibSp','Parch'],
                       response_transform_func=np.log,
                       response_inverse_transform_func=np.exp)
    my_oaxaca.run_oaxaca_analysis()
    my_oaxaca.figs[2].show()
    '''

    def __init__(self,df,response,grp_col,grp_groups=None,
                 dummy_cols = [],
                higher_order_dict=dict(),
                all_factors = [], response_transform_func = None,response_inverse_transform_func = None,
                output_location=None,verbose=0):
        '''
        :param df: the dataframe
        :param response: the response variable that has a gap between the groups we want to investigate. Should be numerical
        :param grp_col: the column containing the two groups to compare the differences between
        :param grp_groups: the two groups we want to compare. If None it will select a random 2 to compare
        :param dummy_cols: the list of columns to dummify. These are the categorical columns. Give 'infer' for the algorithm to infer categorical columns
        :param higher_order_dict: a dictionary containing higher order versions of numerical variables to apply. e.g.
            if we want Age^2, Age^3 and LENGTH_OF_SERVICE_IN_YRS^2 we provide dict((('Age',[2,3]),('LENGTH_OF_SERVICE_IN_YRS',[2])))
        :param all_factors: the full list of factors to try. If None it will include all columns from the dataframe
        :param response_transform_func,response_inverse_transform_func: the function to fit the model on a transformed
            version of the response variable and the inverse of that function. i.e. np.log and np.exp as its inverse
            or lambda x: x as an identity function (i.e. no transform). None also applied the identity function.
        :param output_location: the output folder path to store the results if required
        '''

        # saving variables to the object
        self.df = df.copy()
        self.response = response
        self.grp_col = grp_col
        self.grp_groups = grp_groups
        self.dummy_cols = dummy_cols
        self.higher_order_dict = higher_order_dict
        self.all_factors = all_factors
        self.response_transform_func = response_transform_func
        self.response_inverse_transform_func = response_inverse_transform_func
        self.output_location = output_location
        self.verbose = verbose

        # check that the dataframe has at least 1 row
        if len(self.df) == 0:
            raise Exception('The dataframe has no rows!')

        # check that the response variable is a numeric type
        if not is_numeric_dtype(self.df[self.response]):
            try:
                self.df[self.response] = self.df[self.response].astype(float)
            except:
                raise Exception('The response variable is not of numeric type and could not be converted to float!')

        # if dummy cols are to be infered
        if self.dummy_cols == 'infer':
            self.dummy_cols = []
            for col in all_factors:
                if not is_numeric_dtype(self.df[col]):
                    self.dummy_cols.append(col)

        # check that the group col has at least 2 groups
        if len(list(map(lambda x: not pd.isna(x),self.df[grp_col].unique()))) < 2:
            raise Exception('There should be more than one non nan group in the column {}'.format(grp_col))

        if self.grp_groups is None:
            self.grp_groups = np.random.choice(list(self.df[grp_col].unique()),2,replace=False)

        # if no dummy columns have been specified set it to an empty list
        if self.dummy_cols is None:
            self.dummy_cols = []

        if (self.all_factors is None) or (self.all_factors == []):
            self.all_factors = list(set(self.df.columns) - set(response) - set(self.dummy_cols) - set(self.grp_col))

        # the instructions to create higher power factors
        if self.higher_order_dict is None:
            self.higher_order_dict = dict()

        # the function to use to transform the response variable
        if (self.response_transform_func is None) or (self.response_inverse_transform_func is None):
            self.response_transform_func = (lambda x: x)
            self.response_inverse_transform_func = (lambda x: x)
        
        # we are only interested in the groups specified
        self.df = self.df[self.df[self.grp_col].isin(self.grp_groups)]

        # ensure there aren't any awkward values
        self.df = self.df.replace(-np.inf, np.nan)
        self.df = self.df.replace(np.inf, np.nan)

        self.df[response] = self.df[response].astype('float')
        self.df = self.df[[self.response, self.grp_col] + self.dummy_cols + self.all_factors].copy()

        # to store the model summaries for each group
        self.model_grp_summary = dict()

        # the full model
        self.lin_reg = None

        # a list to hold the figures
        self.figs = []
        
        # the output location for the results
        self.output_location = output_location
        

    def get_dummies_for_oaxaca(self,t):
        '''
        Gets a list of columns to create dummies for while keeping track of the list of created columns
        :param ls: a list of columns to create dummies for
        :returns: a tuple (df,dummy_factors). A transformed df as well as a list of create columns
        '''

        temp = t.copy()

        # we do not drop first because the forward selection will decide which category to add and which not to add
        temp = pd.get_dummies(temp,columns=self.dummy_cols,drop_first=False)

        dummy_factors = []
        for col in self.dummy_cols:
            dummy_factors.extend(list(map(lambda x: x,filter(lambda x: col in x,temp.columns))))

        return temp,dummy_factors


    def create_higher_power_factors(self,t):
        '''
        Creates higher power factors in the dataframe given a dictionary specifying the list of powers
        :param df: a dataframe
        :param d: a dictionary containing column name:power pairs. e.g. {'Age': [2, 3], 'LENGTH_OF_SERVICE_IN_YRS': [2]}
        :returns: (df,power_factors). A transformed df and a list of the transformed factors
        '''

        temp = t.copy()
        power_factors = []

        # for each key value pair in the higher order dictionary, create a variable corresponding to its power
        for factor,ls_power in self.higher_order_dict.items():
            for power in ls_power:
                this_factor = factor+str(power)
                temp[this_factor] = np.power(temp[factor],power)
                power_factors.append(this_factor)

        return temp,power_factors

    #
    #
    # def impute_data(self,t):
    #     '''
    #     Fills in missing data in the columns specified. Using simple means
    #     :param df: a dataframe
    #     :param ls: a list of columns to impute
    #     :returns: temp. A dataframe
    #     '''
    #     temp = t.copy()
    #     for col in self.impute_cols:
    #         temp[col] = temp[col].apply(lambda x: temp[col].mean() if pd.isna(x) else x)
    #
    #     return temp


    def set_denom(self,t):
        '''
        Gets the grp with the largest response value
        :param df: a dataframe
        :param grp_col: the column with the groups. i.e. Gender
        :param ctry_col: the column with the country
        :param response: the column with the response. i.e. YearlyPay
        :returns: country_denoms. A dictionary with the country:group specifying the group with the greatest response value
        '''

        grp = t[[self.grp_col,self.response]].groupby(by=[self.grp_col]).mean().reset_index()
        grp = grp.sort_values(self.response, ascending=False)
        denom = list(grp[self.grp_col])[0]

        self.denom = denom


    def gap(self,t):
        '''
        Gets the raw gap between the groups for a specific country
        :param df: a dataframe
        :param grp_col: the column with the groups. i.e. Gender
        :param grp_col1: one of the groups. i.e. Male
        :param grp_col2: one of the groups. i.e. Female
        :param ctry_col: the column with the country
        :param denom: the group to be in the denominator of the pay gap calculation
        :param response: the column with the response. i.e. YearlyPay
        :returns: pg. The pay gap
        '''
        t_temp = t[[self.grp_col,self.response]].groupby(by=self.grp_col).mean().reset_index()

        g1 = t_temp[t_temp[self.grp_col]==self.grp_groups[0]][self.response].iloc[0]
        g2 = t_temp[t_temp[self.grp_col]==self.grp_groups[1]][self.response].iloc[0]
        
        if self.denom == self.grp_groups[0]:
            s = 'Raw Gap ({denom}-{num})/{denom} = {res:.2f}% or {denom} - {num} = {res2:.2f}'.format(denom=self.grp_groups[0],num=self.grp_groups[1],res=(g1-g2)*100/g1,res2=(g1-g2))
            if self.verbose != 0:
                print(s)
        else:
            s = 'Raw Gap ({denom}-{num})/{denom} = {res:.2f}% or {denom} - {num} = {res2:.2f}'.format(denom=self.grp_groups[1],num=self.grp_groups[0],res=(g2-g1)*100/g2,res2=(g2-g1))
            if self.verbose != 0:
                print(s)
            
        if self.output_location is not None:
            f = open(os.path.join(self.output_location,'raw_pay_gap.txt'), "w")
            f.write(s)
            f.close()

        self.raw_gap_statement = s


    def get_full_model_and_important_factors(self,t):
        '''
        Returns a combined model along with the important/significant factors as well as the metrics of the model
        :param df: a dataframe
        :param all_independent_features: all the features in the data without the response variable
        :param ctry: the country to focus on
        :param country_col: the column with the country
        :param response: the column with the response. i.e. YearlyPay
        :param denom: the group to be in the denominator of the pay gap calculation
        :returns: (countries_params,chosen_variables,r2_values,chosen_model). 
                    A dictionary with the country parameters, a list of the chosen variables for this country, a list of
                    the adjusted r2 values at each addition of a variable and the final chosen model.
        '''
        t = t.copy()

        first_pass = True
        ls_remaining = list(t.drop([self.response,self.grp_col],axis=1).columns)

        # dummify the group column
        t[self.grp_groups[0]]=(t[self.grp_col]==self.grp_groups[0]).astype('int')

        # this will be the running list of features that are significant
        ls = [self.grp_groups[0]]

        # this is to store the next selected variable values
        r2_values = []
        last_r2 = 0
        last_model = None
        chosen_model = []
        chosen_variables = []
        last_anova_pvalue = None
        
        if self.output_location is not None:
            f = open(os.path.join(self.output_location,'model_diagnostics.txt'), "w")
            f.write('Model progression')
            f.close()

        while len(ls_remaining) > 0:
            last_variable = None
            for new_var in ls_remaining:

                # append the new variable to ls
                ls_new = ls + [new_var]

                # we create X and y
                X = t[ls_new].copy()
                X = sm.add_constant(X)
                y = t[self.response]

                # create the model and fit it
                lin_model = sm.OLS(y,X,missing = 'drop')
                self.lin_model_fit_3 = lin_model.fit()
                if not first_pass:
                    # statsmodels can throw a lot of warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        anova_ftest = sm.stats.anova_lm(chosen_model[-1],self.lin_model_fit_3)

                    # if the ftest with respect to the previous model is significant
                    if (anova_ftest['Pr(>F)'].max() < 0.05) or first_pass:
                        if (self.lin_model_fit_3.rsquared > last_r2) or first_pass:
                            last_r2 = self.lin_model_fit_3.rsquared
                            last_variable = new_var
                            last_model = self.lin_model_fit_3
                            last_anova_pvalue = anova_ftest['Pr(>F)'].max()
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        anova_ftest = self.lin_model_fit_3.f_pvalue

                    if (anova_ftest < 0.05) or first_pass:
                        if (self.lin_model_fit_3.rsquared > last_r2):
                            last_r2 = self.lin_model_fit_3.rsquared
                            last_variable = new_var
                            last_model = self.lin_model_fit_3
                            last_anova_pvalue = anova_ftest

            if last_variable is None:
                break

            if first_pass:
                ls_remaining.remove(last_variable)
                ls.append(last_variable)
                first_pass = False
                r2_values.append(last_r2)
                chosen_variables.append(last_variable)
                chosen_model.append(last_model)
                s = '\nChose variable {} with r2 {} and pvalue {}'.format(last_variable,last_r2,last_anova_pvalue)
                if self.verbose != 0:
                    print(s)
            else:
                ls_remaining.remove(last_variable)
                ls.append(last_variable)
                r2_values.append(last_r2)
                chosen_variables.append(last_variable)
                chosen_model.append(last_model)

                s = '\nChose variable {} with r2 {} and pvalue {}'.format(last_variable,last_r2,last_anova_pvalue)
                if self.verbose != 0:
                    print(s)
                
            if self.output_location: 
                f = open(os.path.join(self.output_location,'model_diagnostics.txt'), "a")
                f.write(s)
                f.close()

        if len(chosen_model) > 0:
            self.chosen_variables = chosen_variables
            self.chosen_model = chosen_model
            self.lin_model_fit_3 = chosen_model[-1]
            if self.output_location:
                f = open(os.path.join(self.output_location,'model_diagnostics_full_model.txt'), "w")
                f.write(self.lin_model_fit_3.summary().as_text())
                f.close()

            if self.verbose != 0:
                print(self.lin_model_fit_3.summary())


    def single_model(self,t,this_grp):
        '''
        Returns a combined model along with the important/significant factors as well as the metrics of the model
        :param df: a dataframe
        :param ctry: the country to focus on
        :param ctry_col: the column with the country
        :param grp_col: the column with the groups
        :param grp: the group to run this model against
        :param countries_params: a dictionary having a dictionary for each country. The dictionary for each country 
                    must contain a 'Factors' key being a list of factors that are important for that country
        :param org_cols_cat: all the columns that are categorical
        :returns: (lin_model_fit_grp,X_grp_average,grp_avg). 
                    A linear regression model, the average of the features used to train it for this group and
                    the prediction on this group (the average of the response variable)
        '''
        t = t.copy()
        t_grp = t.copy()
        t_grp = t_grp[(t_grp[self.grp_col] == this_grp)].copy()

        # we create X and y
        X = t_grp[self.chosen_variables].copy()
        X = sm.add_constant(X)
        y = t_grp[self.response]

        # create the model and fit it
        lin_model = sm.OLS(y,X,missing = 'drop')
        lin_model_fit_grp = lin_model.fit()

        if self.verbose != 0:
            print(lin_model_fit_grp.summary())

        self.model_grp_summary[this_grp] = lin_model_fit_grp.summary()
            
        if self.output_location: 
            f = open(os.path.join(self.output_location,'model_diagnostics_{}.txt'.format(this_grp)), "w")
            f.write(lin_model_fit_grp.summary().as_text())
            f.close()

        X_grp = X.copy()
        X_grp_average = pd.DataFrame(X.mean(axis=0)).T
        grp_avg = lin_model_fit_grp.predict(X_grp_average)[0]

        return lin_model_fit_grp,X_grp_average,grp_avg
    
    
    def run_oaxaca_analysis(self,verbose=0):
        dic_output = dict()
        dic_output['s'] = []
        dic_output['figs'] = []

        # create the dummy variables
        t,extra_factors = self.get_dummies_for_oaxaca(self.df)

        # remove any columns that were duplicated
        t=t.loc[:,~t.columns.duplicated()].copy()

        # create higher power features
        t,power_factors = self.create_higher_power_factors(t)
        extra_factors.extend(power_factors)

        # get the group for this country that has a higher response
        self.set_denom(t)

        # report the raw gap
        dic_output['s'].append(self.gap(t))

        # get the numerical column names based on the user specified columns above 
        org_cols_num = list(set(self.all_factors) - set(self.dummy_cols)) + extra_factors

        t[self.response] = self.response_transform_func(t[self.response])

        if ((t[self.response]==-np.inf).sum() + (t[self.response]==np.inf).sum())>0:
            t = t.replace(-np.inf, np.nan)
            t = t.replace(np.inf, np.nan)
            l_before = len(t)
            t = t.dropna()
            l_after = len(t)
            warnings.warn("After transforming the response there were undefined values. After removing them {} rows reduced to {}".format(l_before,l_after))


        all_independent_features = org_cols_num
        self.get_full_model_and_important_factors(t)

        lin_model_fit_for_group = dict()
        X_avg_for_group = dict()
        avg_for_group = dict()
        for this_grp in t[self.grp_col].unique():
            lin_model_fit_grp,X_grp_average,grp_avg = self.single_model(t,this_grp)
            lin_model_fit_for_group[this_grp] = lin_model_fit_grp
            X_avg_for_group[this_grp] = X_grp_average
            avg_for_group[this_grp] = grp_avg


        temp_grps = self.grp_groups.copy()

        temp_grps.remove(self.denom)
        diff = avg_for_group[self.denom] - avg_for_group[temp_grps[0]]

        diff_coeffs = lin_model_fit_for_group[self.denom].params - lin_model_fit_for_group[temp_grps[0]].params

        # potential discrimination diff
        unexplained_diff = X_avg_for_group[temp_grps[0]].dot(diff_coeffs)
        X_diff = X_avg_for_group[self.denom].subtract(X_avg_for_group[temp_grps[0]])
        explained_diff = X_diff.dot(lin_model_fit_for_group[self.denom].params)


        differences_in_factors = dict()

        for idx,factor in enumerate(self.all_factors):
            try:
                d = X_diff.loc[0,[x for x in X_diff.columns if factor in x]].dot(lin_model_fit_for_group[self.denom].params[[x for x in X_diff.columns if factor in x]])
                differences_in_factors[factor] = d
                if self.verbose != 0:
                    print('Difference in {} accounts for a contribution to the difference of {}'.format(factor,d))
            except:
                pass

        final_df = pd.DataFrame(differences_in_factors.values())
        final_df.index = differences_in_factors.keys()
        final_df.reset_index(inplace=True)
        
        final_df.columns = ['Factor','Effect']
        final_df['Abs_Effect'] = final_df['Effect'].abs()
        final_df['percent_Effect'] = (final_df['Effect']/final_df['Effect'].abs().sum())*100
        final_df['Direction'] = final_df['Effect'].apply(lambda x: 'Increases Gap' if x > 0 else 'Decreases Gap')
        final_df.sort_values(by='Abs_Effect',ascending = True,inplace = True)

        unexplained_percent = (unexplained_diff/diff)*100
        percent_due_measured_differences = (explained_diff/diff)*100


        df = pd.DataFrame([unexplained_percent,percent_due_measured_differences],index=['Unexplained pct','Explained pct'])
        df.reset_index(inplace=True)
        df.columns = ['Component','Percent']
        df['Placeholder'] = 'Placeholder'




        import plotly.express as px

        fig = px.bar(df, x="Placeholder", y="Percent", color="Component",
                    hover_data=['Percent'], barmode = 'stack',title='Proportion of the difference we can explain')

        dic_output['figs'].append(fig)
        
        
        if self.output_location: 
            #fig.write_image(os.path.join(self.output_location,'model_explained_prop_{}.png'.format(self.ctry)))
            fig.write_html(os.path.join(self.output_location,'model_explained_prop.html'))

        import plotly.express as px

        fig = px.bar(final_df, x="Direction", y="percent_Effect", color="Factor",
                    hover_data=['percent_Effect'], barmode = 'stack',title='Proportion of explained effects (positive is contributing to the Gap - {} are greater)'.format(self.denom))

        dic_output['figs'].append(fig)
        
        if self.output_location: 
            #fig.write_image(os.path.join(self.output_location,'model_explained_prop_barplot_{}.png'.format(self.ctry)))
            fig.write_html(os.path.join(self.output_location,'model_explained_prop_barplot.html'))

        diff_to_explain = self.response_inverse_transform_func(avg_for_group[self.denom]) - self.response_inverse_transform_func(avg_for_group[temp_grps[0]])
        s = 'Difference to explain after log transform = {:.3f}'.format(diff_to_explain)
        if self.verbose != 0:
            print('Difference to explain after log transform = {:.3f}'.format(diff_to_explain))
        
        if self.output_location: 
            f = open(os.path.join(self.output_location,'model_diagnostics_{}.txt'.format(this_grp)), "a")
            f.write(s)
            f.close()

        dic_output['s'].append(s)

        # for each factor, make it equal to the mean for men
        avg_log_pay_for_group = dict()
        
        factor_effects = dict()
        for factor in self.all_factors:
            t1 = X_avg_for_group[temp_grps[0]].copy()
            for col in t1.columns:
                if factor in col:
                    t1[col] = X_avg_for_group[self.denom][col]
                    factor_effects[factor] = t1.dot(lin_model_fit_for_group[self.denom].params)
        avg_log_pay_for_group[self.denom] = X_avg_for_group[self.denom].dot(lin_model_fit_for_group[self.denom].params)
        avg_log_pay_for_group[temp_grps[0]] = X_avg_for_group[temp_grps[0]].dot(lin_model_fit_for_group[self.denom].params)

        explainable_gap = self.response_inverse_transform_func(avg_log_pay_for_group[self.denom]) - self.response_inverse_transform_func(avg_log_pay_for_group[temp_grps[0]])
        explainable_gap = -explainable_gap
        unexplained_gap = diff_to_explain - explainable_gap

        df_factor_effects = pd.DataFrame(factor_effects)
        df_factor_effects = df_factor_effects.T.reset_index()
        df_factor_effects.columns = ['Factor','ln({}yearlypay)'.format(temp_grps[0])]
        df_factor_effects['ln({}yearlypay_as{})'.format(temp_grps[0],self.denom)] = avg_log_pay_for_group[temp_grps[0]][0]

        df_factor_effects['Difference_in_variable'] = df_factor_effects[['ln({}yearlypay)'.format(temp_grps[0]),'ln({}yearlypay_as{})'.format(temp_grps[0],self.denom)]].apply(lambda x: self.response_inverse_transform_func(x[0])-self.response_inverse_transform_func(x[1]),axis=1)
        df_factor_effects.sort_values(by='Difference_in_variable',inplace=True,ascending=True)

        fig=plt.figure(figsize=(10,5))
        ax=fig.add_subplot(1,1,1)

        sns.barplot(y='Factor',x='Difference_in_variable',data=df_factor_effects,ax=ax,orient='h')
        ax.set_title('Change in {} for {} if their factors were treated the same as {} ({} paid more)'.format(self.response,self.denom,temp_grps[0],temp_grps[0]))

        ax.set_ylabel('Factor equated between {} and {}'.format(self.denom,temp_grps[0]))
        #plt.xticks(rotation=90)
        
        if self.output_location:
            fig.savefig(os.path.join(self.output_location,'model_importancebarplot_{}.png'.format(this_grp)))

        dic_output['figs'].append(fig)

        t1 = df_factor_effects[['Factor','Difference_in_variable']].copy()
        t1.reset_index(drop=True,inplace=True)
        t1.loc[-1] = ['Gap', -diff_to_explain]  # adding a row
        t1.index = t1.index + 1  # shifting index
        t1.sort_index(inplace=True)

        remaining = diff_to_explain - df_factor_effects['Difference_in_variable'].sum()

        t1.loc[-1] = ['REMAINING', remaining]  # adding a row
        t1['Difference_in_variable'] = -t1['Difference_in_variable']

        fig = go.Figure(go.Waterfall(
            name = "pos/neg", orientation = "v",
            measure = ["relative"]*len(t1['Difference_in_variable']),
            x = t1['Factor'],
            #textposition = "outside",
            #text = ["+60", "+80", "", "-40", "-20", "Total"],
            y = t1['Difference_in_variable'],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},base = 0,
            decreasing = {"marker":{"color":"red", "line":{"color":"red", "width":0}}},
            increasing = {"marker":{"color":"green"}}
        ))
        fig.update_layout(
                title = "Importance of factors in the final model for {} ({} are greater)".format(self.response,self.denom),
                showlegend = True
        )
        fig.add_shape(
            type="rect", fillcolor="blue", line=dict(color="blue"), opacity=1,
            x0=-0.4, x1=0.4, xref="x", y0=0.0, y1=fig.data[0].y[0], yref="y"
        )

        fig.add_shape(
            type="rect", fillcolor="blue", line=dict(color="blue"), opacity=1,
            x0=len(t1)-1-0.388, x1=len(t1)-1+0.39, xref="x",
            y0=0, y1=-fig.data[0].y[-1], yref="y"
        )
        dic_output['figs'].append(fig)
        
        if self.output_location: 
            #fig.write_image(os.path.join(self.output_location,'model_waterfall_{}.png'.format(self.ctry)))
            fig.write_html(os.path.join(self.output_location,'model_waterfall_{}.html'.format(self.ctry)))

        self.figs = dic_output['figs']
        self.s = dic_output['s']
     

def unit_test_1():
    print('Unit test 1...')
    import sys
    import os
    import warnings

    np.random.seed(101)
    #warnings.filterwarnings("ignore")

    warnings.filterwarnings('ignore',
                            '.*divide by zero.*')

    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare']]
    df = df.dropna()
    #df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 1 if x =='female' else 0)
    my_oaxaca = Oaxaca(df, grp_col='Sex', response='Fare', grp_groups=['male','female'],
           dummy_cols=['Survived', 'Pclass', 'SibSp','Parch'],
           all_factors=['Survived', 'Pclass', 'Age', 'SibSp','Parch'],
                       response_transform_func=np.log,
                       response_inverse_transform_func=np.exp,verbose=0)
    my_oaxaca.run_oaxaca_analysis()

    result_required = [3.490755216840938, -0.10703341622945511, 1.7817109779453175, -0.6170001903131179, 0.5109449084371861, -0.6641981700074011, -0.3343154517849804, -0.2507091851612552, 0.4738562106245683, 0.44288919159484136]
    result_actual = list(my_oaxaca.lin_model_fit_3.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    print('Success!')


def unit_test_2():
    print('Unit test 2...')
    import sys
    import os
    import warnings

    np.random.seed(101)
    #warnings.filterwarnings("ignore")

    warnings.filterwarnings('ignore',
                            '.*divide by zero.*')

    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'ibd')
    ibd_csv = os.path.join(data_dir, 'IBD.csv')
    df = pd.read_csv(ibd_csv)
    #df = df[['Gender', 'Rank','Status']]
    #df = df.dropna()
    #df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 1 if x =='female' else 0)
    my_oaxaca = Oaxaca(df=df, grp_col='Gender', response='Rank', grp_groups=['All','Female'],
           dummy_cols='infer',
           all_factors=['Status'],
                       response_transform_func=(lambda x: x),
                       response_inverse_transform_func=(lambda x: x),verbose=0)
    my_oaxaca.run_oaxaca_analysis()

    result_required = [468.7401770256098, 3.984140007368431, -59.70088512804106, -113.54010650666291]
    result_actual = list(my_oaxaca.lin_model_fit_3.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

    print('Success!')

if __name__ == '__main__':
    unit_test_1()
    unit_test_2()

