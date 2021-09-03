import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.formula.api as smf
import scipy
import plotly.express as px
import plotly.graph_objects as go
import plotly
import os

class Oaxaca:
    '''
    Example usage:
    df=dsd.titanic_df
    df['dummy_col'] = '1'

    my_oaxaca = dsm.Oaxaca(df=df,grp_col='Sex',ctry_col='dummy_col',response='Fare',ctry='1',grp_groups=['male','female'],
                      grp_groups2 = None,dummy_cols = ['Survived','Pclass','Embarked'],
                d=dict(),
                all_factors = ['Survived','Pclass','Age','Embarked'], org_cols_cat = [],response_transform_func = (lambda x: x),response_inverse_transform_func = (lambda x: x),
                impute_cols = [],output_location=None)

    figs,s,m1,m2=my_oaxaca.run_oaxaca_analysis()
    '''

    def __init__(self,df,grp_col='Gender',ctry_col = 'CountryDescription',response = 'YearlyPay',ctry = 'United Kingdom',
                grp_groups = ['Male','Female'],grp_groups2 = ['Male','Female'],dummy_cols = ['FUNCTION_DESC'],
                d=dict((('Age',[2]),('LENGTH_OF_SERVICE_IN_YRS',[2]))),
                all_factors = ['FUNCTION_DESC','Age','LENGTH_OF_SERVICE_IN_YRS','Mobile','Workweek_span',
                      'HistoricalPartTime','HistoricalIA','MidCareerRecruit','Admin Role','NumHighPerformance',
            'NumDivisions','NumPMUs','NumRoles'], org_cols_cat = [],response_transform_func = np.log,response_inverse_transform_func = np.exp,
                impute_cols = [],output_location=None):
        # some variables for this particular analysis
        self.grp_col = str(grp_col)
        self.ctry_col = str(ctry_col)
        self.response = str(response)
        self.ctry = str(ctry)

        self.model_grp_summary = dict()

        self.grp_groups = list(map(lambda x: str(x),grp_groups))
        self.grp_groups2 = self.grp_groups

        # the columns we want to tackle each category individually
        self.dummy_cols = dummy_cols # WORKLEVEL_CODE, 'IA_MASTER_TYPE_DESC'

        # the instructions to create higher power factors
        self.d=d
        if self.d is None:
            self.d = dict()

        # all the factors to consider
        self.all_factors = all_factors
        # the factors to be treated as categorical variables (don't include the ones specified as dummy columns above)
        self.org_cols_cat = org_cols_cat

        # the function to use to transform the response variable
        self.response_transform_func = response_transform_func
        self.response_inverse_transform_func = response_inverse_transform_func

        # the columns needing imputation
        self.impute_cols = impute_cols
        
        # the main dataframe
        self.df = df.copy()
        self.df = self.df[self.df[self.grp_col].isin(self.grp_groups)]
        self.df = self.df.replace(-np.inf, np.nan)
        self.df = self.df.replace(np.inf, np.nan)
        self.df[grp_col] = self.df[grp_col].astype('str')
        if ctry_col != '':
            self.df[ctry_col] = self.df[ctry_col].astype('str')
        self.df[response] = self.df[response].astype('float')

        if self.ctry_col == '':
            self.df['all_data'] = 'all_data'
            self.ctry_col = 'all_data'
            self.ctry = 'all_data'

        # the full model
        self.lin_reg = None
        
        # the output location for the results
        self.output_location = output_location
        

    def get_dummies_for_oaxaca(self,t):
        '''
        Gets a list of columns to create dummies for while keeping track of the list of created columns
        :param df: a dataframe
        :param ls: a list of columns to create dummies for
        :returns: a tuple (df,dummy_factors). A transformed df as well as a list of create columns
        '''

        temp = t.copy()
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
        for factor,ls_power in self.d.items():
            for power in ls_power:
                this_factor = factor+str(power)
                temp[this_factor] = np.power(temp[factor],power)
                power_factors.append(this_factor)

        return temp,power_factors



    def impute_data(self,t):
        '''
        Fills in missing data in the columns specified. Using simple means
        :param df: a dataframe
        :param ls: a list of columns to impute
        :returns: temp. A dataframe
        '''
        temp = t.copy()
        for col in self.impute_cols:
            temp[col] = temp[col].apply(lambda x: temp[col].mean() if pd.isna(x) else x)

        return temp


    def get_country_denoms(self,t):
        '''
        Gets the grp with the largest response (yearly pay) value
        :param df: a dataframe
        :param grp_col: the column with the groups. i.e. Gender
        :param ctry_col: the column with the country
        :param response: the column with the response. i.e. YearlyPay
        :returns: country_denoms. A dictionary with the country:group specifying the group with the greatest response value
        '''

        grp = t[[self.grp_col,self.ctry_col,self.response]].groupby(by=[self.ctry_col,self.grp_col]).mean().reset_index()
        if self.grp_col == 'Gender':
            grp['Gender'] = grp['Gender']
        grp = grp.sort_values(self.response, ascending=False).drop_duplicates([self.ctry_col])
        country_denoms = dict(np.array(grp[[self.ctry_col,self.grp_col]]))

        return country_denoms


    def pay_gap(self,t,denom,grp_col1='Male',grp_col2='Female'):
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
        t_temp = t[t[self.ctry_col]==self.ctry][[self.grp_col,self.response]].groupby(by=self.grp_col).mean().reset_index().copy()
        if self.grp_col == 'Gender':
            t_temp['Gender'] = t_temp['Gender']
            #if grp_col1 == 'Male':
            #    grp_col1 = 'men'
            #if grp_col1 == 'Female':
            #    grp_col1 = 'women'
            #if grp_col2 == 'Male':
            #    grp_col2 = 'men'
            #if grp_col2 == 'Female':
            #    grp_col2 = 'women'
        g1 = t_temp[t_temp[self.grp_col]==grp_col1][self.response].iloc[0]
        g2 = t_temp[t_temp[self.grp_col]==grp_col2][self.response].iloc[0]
        
        s = ''
        
        if denom == grp_col1:
            s = 'Raw Gap ({denom}-{num})/{denom} = {res:.1f}% or {denom} - {num} = {res2:.1f}'.format(denom=grp_col1,num=grp_col2,res=(g1-g2)*100/g1,res2=(g1-g2))
            print(s)
        else:
            s = 'Raw Gap ({denom}-{num})/{denom} = {res:.1f}% or {denom} - {num} = {res2:.1f}'.format(denom=grp_col2,num=grp_col1,res=(g2-g1)*100/g2,res2=(g2-g1))
            print(s)
            
        if self.output_location: 
            f = open(os.path.join(self.output_location,'raw_pay_gap_{}.txt'.format(self.ctry)), "w")
            f.write(s)
            f.close()
        return s


    def get_full_model_and_important_factors(self,t,countries_params,denom,all_independent_features):
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
        import warnings
        
        t = t.copy()

        first_pass = True
        ls_remaining = all_independent_features.copy()
        t[self.grp_groups[0]]=(t[self.grp_col]==self.grp_groups[0]).astype('int')
        ls = [self.grp_groups[0]]


        # this is to store the next selected variable values
        r2_values = []
        last_variable = None
        last_r2 = 0
        last_model = None
        chosen_model = []
        chosen_variables = []
        last_anova_pvalue = None
        last_X = None
        country_params = dict()
        country_params['denom'] = denom
        
        if self.output_location: 
            f = open(os.path.join(self.output_location,'model_diagnostics_{}.txt'.format(self.ctry)), "w")
            f.write('Model progression')
            f.close()

        while len(ls_remaining) > 0:
            last_variable = None
            for new_var in ls_remaining:

                # append the new variable to ls
                ls_new = ls + [new_var]

                # we create X and y
                X = t[t[self.ctry_col] == self.ctry][ls_new].copy()

                #X = pd.get_dummies(X,columns=list((set(self.org_cols_cat) & set(ls_new))),drop_first=True)
                #y = pd.Series(np.random.randint(0, 100, 1000))
                #x1 = pd.DataFrame(np.random.randint(0, 100, 1000))
                #x2 = pd.DataFrame(np.random.randint(0, 100, 1000))
                #X = pd.concat([x1, x2], axis=1)
                #X.columns = ['x1', 'x2']

                X = sm.add_constant(X)
                y = t[t[self.ctry_col] == self.ctry][self.response]

                # create the model and fit it
                lin_model = sm.OLS(y,X,missing = 'drop')
                self.lin_model_fit_3 = lin_model.fit()
                if not first_pass:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        anova_ftest = sm.stats.anova_lm(chosen_model[-1],self.lin_model_fit_3)
                    if (anova_ftest['Pr(>F)'].max() < 0.05) or first_pass:
                        if (self.lin_model_fit_3.rsquared > last_r2) or first_pass:
                            last_r2 = self.lin_model_fit_3.rsquared
                            last_variable = new_var
                            last_model = self.lin_model_fit_3
                            last_anova_pvalue = anova_ftest['Pr(>F)'].max()
                            last_X = X.copy()
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
                            last_X = X.copy()

            if last_variable is None:
                break
                
            #df_temp = last_model.pvalues.copy()
            #not_sig = list(df_temp[df_temp > 0.05].index)
            #sig = list(df_temp[df_temp <= 0.05].index)
            
            #s2=''
            #for col in not_sig:
            #    idx = ls.index(col)
            #    print(not_sig)
            #    print(col)
            #    print(ls)
            #    print(idx)
            #    s2 = '\Removed variable {} with r2 {} and pvalue {}'.format(ls[idx],r2_values[idx])
            #    print(s2)
            #    del ls[idx]
            #    del r2_values[idx]
            #    del chosen_variables[idx]
            #    del chosen_model[idx]
                

            country_params['Factors'] = chosen_variables

            s = ''
            if first_pass:
                ls_remaining.remove(last_variable)
                ls.append(last_variable)
                first_pass = False
                r2_values.append(last_r2)
                chosen_variables.append(last_variable)
                chosen_model.append(last_model)
                s = '\nChose variable {} with r2 {} and pvalue {}'.format(last_variable,last_r2,last_anova_pvalue)
                print(s)
            else:
                ls_remaining.remove(last_variable)
                ls.append(last_variable)
                r2_values.append(last_r2)
                chosen_variables.append(last_variable)
                chosen_model.append(last_model)

                s = '\nChose variable {} with r2 {} and pvalue {}'.format(last_variable,last_r2,last_anova_pvalue)
                print(s)
                
            if self.output_location: 
                f = open(os.path.join(self.output_location,'model_diagnostics_{}.txt'.format(self.ctry)), "a")
                f.write(s)
                f.close()

        countries_params[self.ctry] = country_params
        if len(chosen_model) > 0:
            self.lin_model_fit_3 = chosen_model[-1]
            if self.output_location:
                f = open(os.path.join(self.output_location,'model_diagnostics_{}_full_model.txt'.format(self.ctry)), "w")
                f.write(self.lin_model_fit_3.summary().as_text())
                f.close()

            print(self.lin_model_fit_3.summary())
        return countries_params,chosen_variables,r2_values,chosen_model


    def single_model(self,t,this_grp,countries_params,silent=False):
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
        t_grp = t_grp[(t_grp[self.ctry_col] == self.ctry) & (t_grp[self.grp_col] == this_grp)].copy()

        # we create X and y
        X = t_grp[countries_params[self.ctry]['Factors']].copy()
        #X = pd.get_dummies(X,columns=list((set(self.org_cols_cat) & set(countries_params[self.ctry]['Factors']))),drop_first=True)
        X = sm.add_constant(X)
        y = t_grp[self.response]

        # create the model and fit it
        lin_model = sm.OLS(y,X,missing = 'drop')
        lin_model_fit_grp = lin_model.fit()

        if not silent:
            print(lin_model_fit_grp.summary())

        self.model_grp_summary[this_grp] = lin_model_fit_grp.summary()
            
        if self.output_location: 
            f = open(os.path.join(self.output_location,'model_diagnostics_{}_{}.txt'.format(self.ctry,this_grp)), "w")
            f.write(lin_model_fit_grp.summary().as_text())
            f.close()

        X_grp = X.copy()
        X_grp_average = pd.DataFrame(X.mean(axis=0)).T
        grp_avg = lin_model_fit_grp.predict(X_grp_average)[0]

        return lin_model_fit_grp,X_grp_average,grp_avg
    
    
    def run_oaxaca_analysis(self):
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

        # impute missing data
        t=self.impute_data(t)

        # get the group for this country that has a higher response
        country_denoms = self.get_country_denoms(t)

        # this will contain the country info
        countries_params= dict()

        # extract the higher paying group
        denom = country_denoms[self.ctry]

        # report the raw gap
        dic_output['s'].append(self.pay_gap(t,denom,self.grp_groups[0],self.grp_groups[1]))

        # get the numerical column names based on the user specified columns above 
        org_cols_num = list(set(self.all_factors)-set(self.org_cols_cat) - set(self.dummy_cols))+extra_factors

        t[self.response] = self.response_transform_func(t[self.response])
        t = t.replace(np.inf,np.nan).replace(-np.inf,np.nan)

        all_independent_features = org_cols_num + self.org_cols_cat
        countries_params,chosen_variables,r2_values,chosen_model = self.get_full_model_and_important_factors(t,countries_params,denom,all_independent_features)

        lin_model_fit_for_group = dict()
        X_avg_for_group = dict()
        avg_for_group = dict()
        for this_grp in t[self.grp_col].unique():
            lin_model_fit_grp,X_grp_average,grp_avg = self.single_model(t,this_grp,countries_params,silent=False)
            this_grp2 = this_grp
            #if this_grp2 == 'Male':
            #    this_grp2 = 'men'
            #elif this_grp2 == 'Female':
            #    this_grp2 = 'women'
            lin_model_fit_for_group[this_grp2] = lin_model_fit_grp
            X_avg_for_group[this_grp2] = X_grp_average
            avg_for_group[this_grp2] = grp_avg


        temp_grps = self.grp_groups2.copy()

        temp_grps.remove(denom)
        diff = avg_for_group[denom] - avg_for_group[temp_grps[0]]

        diff_coeffs = lin_model_fit_for_group[denom].params - lin_model_fit_for_group[temp_grps[0]].params

        # potential discrimination diff
        unexplained_diff = X_avg_for_group[temp_grps[0]].dot(diff_coeffs)
        X_diff = X_avg_for_group[denom].subtract(X_avg_for_group[temp_grps[0]])
        explained_diff = X_diff.dot(lin_model_fit_for_group[denom].params)


        differences_in_factors = dict()

        for idx,factor in enumerate(self.all_factors):
            try:
                d = X_diff.loc[0,[x for x in X_diff.columns if factor in x]].dot(lin_model_fit_for_group[denom].params[[x for x in X_diff.columns if factor in x]])
                differences_in_factors[factor] = d
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
            fig.write_html(os.path.join(self.output_location,'model_explained_prop_{}.html'.format(self.ctry)))

        import plotly.express as px

        fig = px.bar(final_df, x="Direction", y="percent_Effect", color="Factor",
                    hover_data=['percent_Effect'], barmode = 'stack',title='{} - Proportion of explained effects (positive is contributing to the Gap - {} are greater)'.format(self.ctry,denom))

        dic_output['figs'].append(fig)
        
        if self.output_location: 
            #fig.write_image(os.path.join(self.output_location,'model_explained_prop_barplot_{}.png'.format(self.ctry)))
            fig.write_html(os.path.join(self.output_location,'model_explained_prop_barplot_{}.html'.format(self.ctry)))

        diff_to_explain = self.response_inverse_transform_func(avg_for_group[denom]) - self.response_inverse_transform_func(avg_for_group[temp_grps[0]])
        #diff_to_explain = avg_for_group[denom] - avg_for_group[temp_grps[0]]

        s = 'Difference to explain after log transform = {:.3f}'.format(diff_to_explain)
        print('Difference to explain after log transform = {:.3f}'.format(diff_to_explain))
        
        if self.output_location: 
            f = open(os.path.join(self.output_location,'model_diagnostics_{}.txt'.format(self.ctry,this_grp)), "a")
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
                    t1[col] = X_avg_for_group[denom][col]
                    factor_effects[factor] = t1.dot(lin_model_fit_for_group[denom].params)
        avg_log_pay_for_group[denom] = X_avg_for_group[denom].dot(lin_model_fit_for_group[denom].params)
        avg_log_pay_for_group[temp_grps[0]] = X_avg_for_group[temp_grps[0]].dot(lin_model_fit_for_group[denom].params)

        explainable_gap = self.response_inverse_transform_func(avg_log_pay_for_group[denom]) - self.response_inverse_transform_func(avg_log_pay_for_group[temp_grps[0]])

        unexplained_gap = diff_to_explain - explainable_gap

        df_factor_effects = pd.DataFrame(factor_effects)
        df_factor_effects = df_factor_effects.T.reset_index()
        df_factor_effects.columns = ['Factor','ln({}yearlypay)'.format(temp_grps[0])]
        df_factor_effects['ln({}yearlypay_as{})'.format(temp_grps[0],denom)] = avg_log_pay_for_group[temp_grps[0]][0]

        df_factor_effects['Difference_in_variable'] = df_factor_effects[['ln({}yearlypay)'.format(temp_grps[0]),'ln({}yearlypay_as{})'.format(temp_grps[0],denom)]].apply(lambda x: self.response_inverse_transform_func(x[0])-self.response_inverse_transform_func(x[1]),axis=1)
        df_factor_effects.sort_values(by='Difference_in_variable',inplace=True,ascending=True)

        fig=plt.figure(figsize=(10,5))
        ax=fig.add_subplot(1,1,1)

        sns.barplot(y='Factor',x='Difference_in_variable',data=df_factor_effects,ax=ax,orient='h')
        ax.set_title('{} - Change change for {} if their factors were treated the same as {} ({} paid more)'.format(self.ctry,temp_grps[0],denom,denom))

        ax.set_ylabel('Factor equated between {} and {}'.format(denom,temp_grps[0]))
        #plt.xticks(rotation=90)
        
        if self.output_location:
            fig.savefig(os.path.join(self.output_location,'model_importancebarplot_{}.png'.format(self.ctry,this_grp)))

        dic_output['figs'].append(fig)

        t1 = df_factor_effects[['Factor','Difference_in_variable']].copy()
        t1.reset_index(drop=True,inplace=True)
        t1.loc[-1] = ['Pay Gap', -diff_to_explain]  # adding a row
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
                title = "Importance of factors in the final model for {} in {} ({} are greater)".format(self.response,self.ctry,denom),
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


        return dic_output['figs'],dic_output['s'],self.lin_model_fit_3.summary(),self.model_grp_summary
     

def unit_test_1():
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
    df['Dummy'] = '1'
    my_oaxaca = Oaxaca(df, grp_col='Sex', response='Fare', grp_groups=['male', 'female'],
           dummy_cols=['Survived', 'Pclass', 'SibSp','Parch'],
           all_factors=['Survived', 'Pclass', 'Age', 'SibSp','Parch'],d=dict(),ctry_col='Dummy',ctry='1',
                       response_transform_func=np.log,
                       response_inverse_transform_func=np.exp
                       )
    figs,_,_,_ = my_oaxaca.run_oaxaca_analysis()

    result_required = [3.490755216840938, -0.10703341622945511, 1.7817109779453175, -0.6170001903131179, 0.5109449084371861, -0.6641981700074011, -0.3343154517849804, -0.2507091851612552, 0.4738562106245683, 0.44288919159484136]
    result_actual = list(my_oaxaca.lin_model_fit_3.params)

    result_required = list(map(lambda x: round(x, 2), result_required))
    result_actual = list(map(lambda x: round(x, 2), result_actual))

    assert (sorted(result_required) == sorted(result_actual))

#     result_required = 0.4061624649859944
#     result_actual = my_logistic_regresion_class.result_with_feature_selection.predict(my_logistic_regresion_class.X_with_feature_selection).mean()
#
#     result_required = round(result_required, 2)
#     result_actual = round(result_actual, 2)
#
#     assert (result_required == result_actual)
#
#     result_required = '''Performance (0 is negative 1 is positive)
# 5-Fold Cross Validation Results:
# Test Set accuracy = 0.8
# f1 = 0.74
# precision = 0.78
# recall = 0.72
# auc = 0.85'''
#
#     result_actual = my_logistic_regresion_class.log_reg_diagnostic_performance(my_logistic_regresion_class.X_with_feature_selection,
#                                                                                my_logistic_regresion_class.y_with_feature_selection)
#
#     assert (result_required == result_actual)
#
#     result_required = [0.4061624649859944, 0.236473141666754, 0.7141621577909735, 0.3998399415589254, 0.3537524130019424]
#     result_actual = list(my_logistic_regresion_class.get_interpretation(my_logistic_regresion_class.result_with_feature_selection,
#                                                                         my_logistic_regresion_class.X_with_feature_selection.columns)[
#                              'Probability'])
#
#     result_required = list(map(lambda x: round(x, 2), result_required))
#     result_actual = list(map(lambda x: round(x, 2), result_actual))
#
#     assert (result_required == result_actual)
#
#     result_required = [365, 59, 78, 212]
#     result_actual = my_logistic_regresion_class.logistic_regression_get_report(
#         my_logistic_regresion_class.result_with_feature_selection,
#         my_logistic_regresion_class.X_with_feature_selection,
#         my_logistic_regresion_class.y_with_feature_selection,verbose=False)
#
#     result_required = list(map(lambda x: round(x, 2), result_required))
#     result_actual = list(map(lambda x: round(x, 2), list(np.array(result_actual).flatten())))
#
#     assert (result_required == result_actual)

    print('Success!')

if __name__ == '__main__':
    unit_test_1()

