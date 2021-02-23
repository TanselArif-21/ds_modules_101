# Libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def heatmap(x, y, value, figuresize=(10,10), fontsize=20,colour_from_to=(15,235),minimum_size=None, maximum_size=None,
            size_scale=200,marker_style='o',show_grid_lines = True):
    '''
    A function to be used to generate a heatmap using icons with the colour indicating both the direction of the value and the
    size indicating the magnitude.
    :param x: Series object of size n containing a list of string names. Forms the x-axis label names in the heatmap
    :param y: Series object of size n containing a list of string names. Forms the y-axis label names in the heatmap
    :param size: Series object of size n containing the values corresponding to the pairs of string names in x and y
    :param figuresize: Tuple. The size of the resulting figure. Default (10,10)
    :param fontsize: Int. The size of the labels. Default 20
    :param colour_from_to: Tuple. Controls the spectrum of colours. Defaults to (15,235). Max is 256
    :param minimum_size: The value to assign the minimum colour to. For correlations this is -1. If None, it is inferred from the data
    :param maximum_size: The value to assign the maximum colour to. For correlations this is 1. If None, it is inferred from the data
    :param size_scale: How big the icons can get
    :param marker_style: The style of the markers. Default is 'o' for circle. See here for options: https://matplotlib.org/3.1.1/api/markers_api.html
    
    Example Usage 1:
    from ds_modules_101 import Plotting as dsp
    from ds_modules_101.Data import titanic_df
    import pandas as pd

    data = titanic_df.copy()
    data = data[['Survived','Pclass','SibSp','Parch','Fare']].copy()
    columns = ['Survived','Pclass','SibSp','Parch','Fare'] 
    corr = data[columns].corr()
    corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']
    a = dsp.heatmap(
        x=corr['x'],
        y=corr['y'],
        value=corr['value']
    )
    
    corr looks like this:
        x         y         value
    0   Survived  Survived  1.000000
    1     Pclass  Survived -0.338481
    2      SibSp  Survived -0.035322
    3      Parch  Survived  0.081629
    4       Fare  Survived  0.257307
    5   Survived    Pclass -0.338481
    6     Pclass    Pclass  1.000000
    7      SibSp    Pclass  0.083081
    8      Parch    Pclass  0.018443
    9       Fare    Pclass -0.549500
    10  Survived     SibSp -0.035322
    11    Pclass     SibSp  0.083081
    12     SibSp     SibSp  1.000000
    13     Parch     SibSp  0.414838
    14      Fare     SibSp  0.159651
    15  Survived     Parch  0.081629
    16    Pclass     Parch  0.018443
    17     SibSp     Parch  0.414838
    18     Parch     Parch  1.000000
    
    Example Usage 2:
    from ds_modules_101 import Plotting as dsp
    from ds_modules_101.Data import titanic_df
    import pandas as pd
    
    dsp.heatmap(['Survived','Parch','Survived','Parch','Survived','Parch','Pclass','Pclass','Pclass'],['Parch','Survived','Survived','Parch','Pclass','Pclass','Pclass','Survived','Parch'],[0.1,-0.5,1,1,0.8,0.2,0.2,0.1,0.3])
    '''
    # convert to size
    size = np.abs(value)

    # get a figure object and set the size
    fig, ax = plt.subplots(figsize=figuresize)

    # get a colour palette within range colour_from_to split by 256
    palette = sns.diverging_palette(*colour_from_to,n=256)
    
    # set the minimum value
    if minimum_size is None:
        minimum_size = np.min(value)
    
    # set the maximum value
    if maximum_size is None:
        maximum_size = np.max(value)
    
    # protect against the max size being the same as the min size
    if maximum_size == minimum_size:
        maximum_size = minimum_size + 0.0000001
        
    color_min, color_max = [minimum_size, maximum_size]

    # create a mapping from the string names to integers in sorted order
    
    x_labels = [v for v in sorted(np.unique(x))]
    y_labels = [v for v in sorted(np.unique(y))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 

    # create a grid layout with 1 row and 20 columns. This is so that the last column is left for the legend
    plot_grid = plt.GridSpec(1, 20, hspace=0.2, wspace=0.1)
    ax = plt.subplot(plot_grid[:,:-1])

    def convert_to_color(val):
        val_position = float((val - color_min)) / (color_max - color_min)
        ind = int(val_position * (256 - 1))
        return palette[ind]

    if 'map' in x.__dir__():
        ax.scatter(
            x=x.map(x_to_num),
            y=y.map(y_to_num),
            s=size * size_scale,
            c=value.apply(convert_to_color),
            marker=marker_style
        )
    else:
        ax.scatter(
            x=pd.Series(list(map(lambda i: x_to_num[i],x))),
            y=pd.Series(list(map(lambda i: y_to_num[i],y))),
            s=pd.Series(list(map(lambda i: i*size_scale,size))),
            c=list(map(lambda i: convert_to_color(i),value)),
            marker=marker_style
        )

    # we only want the ticks between the major ticks to show
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    
    if show_grid_lines:
        ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
        ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right',fontsize=fontsize)
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels,fontsize=fontsize)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5]) 
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

    # Add a legend/bar for the color
    ax2 = plt.subplot(plot_grid[:,-1])
    col_x = [0]*len(palette)
    bar_y=np.linspace(color_min, color_max, 256)

    bar_height = bar_y[1] - bar_y[0]
    ax2.barh(
        y=bar_y,
        width=[5]*len(palette),
        left=col_x,
        height=bar_height,
        color=palette,
        linewidth=0
    )
    ax2.set_xlim(1, 2)
    ax2.grid(False)
    ax2.set_facecolor('white')
    ax2.set_xticks([])
    ax2.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))
    ax2.yaxis.tick_right()
    ax2.set_yticklabels(np.linspace(min(bar_y), max(bar_y), 3),fontsize=fontsize)

    return fig

def correlation_heatmap(df,columns,method='pearson',minimum_size=-1,maximum_size=1,**kwargs):
    '''
    A function to just do a correlation heatmap between the specified columns
    :param df: The data frame object
    :param method: The correlation type. 'pearson', 'kendall', 'spearman' or callable Method of correlation
    :param columns: A list of column names to do a correlation between
    :param minimum_size: float. Where you want to colour scale to start from. i.e. the minimum value. Default = -1
    :param maximum_size: float. Where you want to colour scale to end. i.e. the maximum value. Default = +1
    :param kwargs: Key word arguments to be passed to the heatmap function
    
    :return fig: A figure object
    
    Example Usage:
    from ds_modules_101 import Plotting as dsp
    from ds_modules_101.Data import titanic_df
    import pandas as pd
    
    dsp.correlation_heatmap(titanic_df,['Survived','Pclass','SibSp','Parch','Fare'],figuresize=(10, 10),minimum_size=-1,
    maximum_size=1)
    '''
    
    data = df[columns].copy()
    
    # get a correlation matrix between all columns
    corr = data.corr(method)
    
    # convert the dataframe by unpivotting so that the first 2 columns are the pair of names and the 3rd column their correlation
    corr = pd.melt(corr.reset_index(), id_vars='index')
    
    # rename the columns
    corr.columns = ['x', 'y', 'value']
    
    # get the heatmap
    fig = heatmap(
        x=corr['x'],
        y=corr['y'],
        value=corr['value'],minimum_size=minimum_size,maximum_size=maximum_size,**kwargs
    )
    return fig
    
def Heatmap_Survey_group_difference(df,comparison_col_name,comparison_order,choice_col_name,category_col_name,category_order = None,
                                   minimum_size = 5,annot_loc = 10,annotate_dprime = True,annotate_mean_diffs = False,annotate_pvalue_significance = True,
                                   output_diagnostics = False,significance_level = 0.01,significance_level2 = 0.05,significance_level3 = 0.1,footnote_x = 0, footnote_y = -0.5,footnote_size=13,annotation_font_size=16,
                                   ticks_font_size=20,custom_title = None, title_append = ''):

    '''
    This function is a heatmap of survey data representing response choices to a particular question. For each category, it
    looks at whether there is a difference in the choice between 2 comparison groups. There is functionality to output pvalues
    based on the proportion (mean) that group has chosen this choice as well as d prime results comparing the relative 
    potency of that choice within each of the groups. The function does not care if this is a single choice question or multiple.

    :param df: The dataframe
    :param comparison_col_name: String. This column should contain at least 2 groups. This column will be used to compare choice answers between
    :param comparison_order: List of size 2. This must be supplied. The 2 groups to consider
    :param choice_col_name: String. The column name representing the choice
    :param category_col_name: String. The column to be used to focus the analysis within
    :param category_order: List. The order to place the categories. If not supplied this will be inferred from the df
    :param minimum_size: Int. Specifies the minimum required sample size when displaying results. Defaults to 5
    :param annot_loc: Float. Controls where the annotations are. The larget the value the more the annotations go to the right
    :param annotate_dprime: Boolean. Default True
    :param annotate_mean_diffs: Boolean. Default False
    :param annotate_pvalue_significance. Boolean. Default True
    :param output_diagnostics: Boolean. Default False
    :param significance_level: Float. Default 0.01
    :param significance_level2: Float. Default 0.05. Used for annotation of 1, 2 or 3 stars
    :param significance_level3: Float. Default 0.1. Used for annotation of 1, 2 or 3 stars
    :param footnote_x: The x coordinate of the footnote
    :param footnote_y: The y coordinate of the footnote
    :param footnote_size: The label size of the footnote
    
    :returns: fig
    
    Example Usage:
    from ds_modules_101 import Plotting as dsp
    from ds_modules_101.Data import titanic_df
    import pandas as pd
    
    # get only specific columns
    temp = titanic_df[[pclass,sex,embarked]].copy()

    # filter to have only sex in these
    temp = temp[temp[sex].isin([male,female])][[sex,pclass,embarked]].copy()


    choice_col_name = 'Pclass'
    category_col_name = 'Sex'
    comparison_col_name = 'Embarked'

    category_order = ['male','female']

    comparison_order = ['C','S']
    
    f = dsp.Heatmap_Survey_group_difference(temp,comparison_col_name,comparison_order,choice_col_name,category_col_name,category_order = None,
                                       minimum_size = 5,annot_loc = 10,annotate_dprime = True,annotate_mean_diffs = False,annotate_pvalue_significance = True,
                                       output_diagnostics = False,significance_level = 0.05)
    '''
    single_star = False
    if max([significance_level,significance_level2,significance_level3]) == significance_level:
        single_star = True
        if output_diagnostics:
            print('Only using single star pvalue annotation because the significance levels provided are not incremental')
    elif max([significance_level,significance_level2]) == significance_level:
        single_star = True
        if output_diagnostics:
            print('Only using single star pvalue annotation because the significance levels provided are not incremental')
    elif max([significance_level2,significance_level3]) == significance_level2:
        single_star = True
        if output_diagnostics:
            print('Only using single star pvalue annotation because the significance levels provided are not incremental')
    
    if category_order is None:
        category_order = list(df[category_col_name].unique())

    # remove any rows that are missing entries
    temp = df[~df.isna().any(axis=1)].copy()

    # add a key
    temp['key'] = 1

    # get the unique entries in this column
    choices = temp[[choice_col_name]].drop_duplicates(keep='first')

    # create a key
    choices['key'] = 1

    # join all rows on the key
    temp = pd.merge(left = temp,right = choices,on='key',how='left',suffixes=('','_y'))

    # create a column which specifies whether this class was the choice
    temp['choice'] = temp[[choice_col_name,choice_col_name+'_y']].apply(lambda x: 1 if x[0] == x[1] else 0,axis=1)
    temp.drop(choice_col_name+'_y',axis=1,inplace=True)

    # get the order of the class
    order = sorted(list(temp[choice_col_name].unique()))

    # add a count column
    temp['Count'] = 1

    # get the total number of people in each of these groups as well as the total number of people who are each choice_col_name
    grp = temp.groupby(by = [category_col_name,comparison_col_name,choice_col_name]).sum().reset_index()

    # get the total in 2 fields
    totals = temp.groupby(by = [category_col_name,comparison_col_name]).sum().reset_index()
    if choice_col_name in list(totals.columns):
        totals.drop(choice_col_name,axis=1,inplace=True)
    
    # join the totals onto the groups
    grp = pd.merge(left = grp,right = totals, on=[category_col_name,comparison_col_name],suffixes=('_level','_levelgroup'))

    # calculate the pct in the groups
    grp['pct in group'] = grp[['Count_level','Count_levelgroup']].apply(lambda x: x[0]*100/x[1],axis=1)

    # join the group table to get differences between category_col_name
    grp_cross_ethnicity = pd.merge(left=grp,right=grp,on=[category_col_name,choice_col_name])


    grp_cross_ethnicity = grp_cross_ethnicity[(grp_cross_ethnicity[comparison_col_name+'_x'] == comparison_order[0]) & (grp_cross_ethnicity[comparison_col_name+'_y'] == comparison_order[1])]
    grp_cross_ethnicity = grp_cross_ethnicity[(grp_cross_ethnicity['choice_level_x'] > minimum_size) & (grp_cross_ethnicity['choice_level_y'] > minimum_size)]
    grp_cross_ethnicity['ScaledProp'] = grp_cross_ethnicity[['pct in group_y','pct in group_x']].apply(lambda x: round(x[0]*100/(x[0]+x[1]),2),axis=1)

    # remove rows smaller than 10
    grp_cross_ethnicity = grp_cross_ethnicity[(grp_cross_ethnicity['choice_level_x'] >= minimum_size) & (grp_cross_ethnicity['choice_level_y'] >= minimum_size)].copy()

    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(1,1,1)
    df = pd.pivot_table(data=grp_cross_ethnicity,
                        index=category_col_name,
                        values='ScaledProp',
                        columns=choice_col_name)
    sns.heatmap(df,annot=False, cmap="RdYlGn",annot_kws={"size": 18},ax=ax,vmin=20, vmax=80)

    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 20,rotation=45)
    ax.set_xlabel(ax.get_xlabel(), fontsize = 20)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 20)
    title = 'Proportion importance between {0} = {1} and {2} ({1}:{2})'.format(comparison_col_name,comparison_order[0],comparison_order[1])
    if custom_title is not None:
        title = custom_title
    title = title + ' ' + title_append
    ax.set_title(title)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 20,rotation=45)

    # Get axis labels and locations
    xlabs = list(map(lambda x: x.get_text(),ax.get_xticklabels()))
    xtiks = list(ax.get_xticks())
    ylabs = list(map(lambda x: x.get_text(),ax.get_yticklabels()))
    ytiks = list(ax.get_yticks())

    # Draw significance
    stress = comparison_order

    for ix,xlab in enumerate(xlabs):
        for iy,ylab in enumerate(ylabs):
            temp['value'] = temp[choice_col_name].apply(lambda x: 1 if str(x) == xlab else 0)
            cohort1 = temp[(temp[comparison_col_name] == comparison_order[0]) & (temp[category_col_name] == ylab) & (temp['choice'] == 1)]

            cohort2 = temp[(temp[comparison_col_name] == comparison_order[1]) & (temp[category_col_name] == ylab) & (temp['choice'] == 1)]

            pos1 = cohort1[cohort1['value'] == 1]
            pos2 = cohort2[cohort2['value'] == 1]

            pos1_len = len(pos1)
            pos2_len = len(pos2)

            cohort1_len = len(cohort1)
            cohort2_len = len(cohort2) 

            prop1 = pos1_len/cohort1_len
            prop2 = pos2_len/cohort2_len

            scaledprop1 = prop1/(prop1 + prop2)
            scaledprop2 = prop2/(prop1 + prop2)

            pvalue = stats.ttest_ind(cohort1['value'],cohort2['value'], equal_var = False)[1]
            #pvalue = proportions_ztest(count=[len(pos1),len(pos2)],nobs=[len(cohort1),len(cohort2)])[1]
            if output_diagnostics:
                print('{0} and {1}: \npvalue = {2} \n{3}count_{4}count = {5}_{6} \n{3}true_{4}true = {7}_{8} \nproportions = {9}:{10} \nScaled Proportions = {11}:{12} \nd-prime significant: {13}\n\n'.format(xlab,ylab,pvalue,comparison_order[0],comparison_order[1],cohort1_len,cohort2_len,pos1_len,pos2_len,prop1,prop2,scaledprop1,scaledprop2,(scaledprop1 >= 0.6) or (scaledprop2 >= 0.6)))

            if annotate_pvalue_significance:
                if (pos1_len > minimum_size) and (pos2_len > minimum_size):
                    if pvalue <= significance_level:
                        #sns.scatterplot(x=[xtiks[ix] + (xtiks[1]-xtiks[0])/4],y=[ytiks[iy]],s=200,marker="***",color='black',ax=ax)
                        plt.text(x=xtiks[ix] + (xtiks[1]-xtiks[0])/4,y=ytiks[iy], s='***', fontsize=annotation_font_size)
                    elif (pvalue <= significance_level2) and not single_star:
                        #sns.scatterplot(x=[xtiks[ix] + (xtiks[1]-xtiks[0])/4],y=[ytiks[iy]],s=200,marker="**",color='black',ax=ax)
                        plt.text(x=xtiks[ix] + (xtiks[1]-xtiks[0])/4,y=ytiks[iy], s='**', fontsize=annotation_font_size)
                        a=1
                    elif (pvalue <= significance_level3) and not single_star:
                        #sns.scatterplot(x=[xtiks[ix] + (xtiks[1]-xtiks[0])/4],y=[ytiks[iy]],s=200,marker="*",color='black',ax=ax)
                        plt.text(x=xtiks[ix] + (xtiks[1]-xtiks[0])/4,y=ytiks[iy], s='*', fontsize=annotation_font_size)

            if annotate_dprime:
                if (pos1_len > minimum_size) and (pos2_len > minimum_size):
                    plt.text(x=xtiks[ix] - (xtiks[1]-xtiks[0])/annot_loc,y=ytiks[iy], s='{}:{}'.format(int(np.round(scaledprop1*100,0)),int(np.round(scaledprop2*100,0))), fontsize=annotation_font_size)
            elif annotate_mean_diffs:
                if (pos1_len > minimum_size) and (pos2_len > minimum_size):
                    plt.text(x=xtiks[ix] - (xtiks[1]-xtiks[0])/annot_loc,y=ytiks[iy], s='{}'.format(np.round(prop1-prop2,1)), fontsize=annotation_font_size)

    t = 'Any cell not appearing has sample size < {0} for one of the groups\n'.format(minimum_size)

    if annotate_dprime:
        t = t + 'Annotations: d prime scaled proportions \nColour: Red indicates that the choice is more potent in {0}\n'.format(comparison_order[0])

    if annotate_pvalue_significance:
        if single_star:
            t = t + 'Pvalue: A *** indicates significance between {0} groups at the {1} level'.format(comparison_col_name,significance_level)
        else:
            t = t + 'Pvalue: A *, ** and *** indicates significance between {0} groups at the {1}, {2}, {3} levels respectively'.format(comparison_col_name,significance_level3,significance_level2,significance_level)


    a=fig.text(footnote_x,footnote_y,t,size=footnote_size)


    cbar = ax.collections[0].colorbar
    ts = cbar.get_ticks()
    cbar.set_ticks(ts)
    cbar.set_ticklabels(['{}:{}'.format(100-int(i),int(i)) for i in ts])
    cbar.ax.tick_params(labelsize=ticks_font_size)
    
    return fig
    
def Heatmap_group_difference(df,bucket_col_name,comparison_col_name,category_col_name,value_col_name,
                             bucket_order=None,
                             comparison_order=None,
                             category_order=None,
                             minimum_size = 5,annot_loc = 10,
                             annotate_pvalue_significance = True,
                             output_diagnostics = False,significance_level = 0.01,significance_level2 = 0.05,
                             significance_level3 = 0.1,footnote_x = 0, footnote_y = -0.5,footnote_size=13,
                             annotation_font_size=16,ticks_font_size=20,custom_title = None, title_append = '',dp=3):

    '''
    This function is a heatmap of the mean differences between 2 groups in the data. For each category and bucket, it
    looks at whether there is a difference in the choice between 2 comparison groups.

    :param df: The dataframe
    :param bucket_col_name: String. The column in the data frame that will appear along the x axis of the heatmap
    :param value_col_name: String. This column should contain numeric values
    :param bucket_order: List. The order of the buckets we want to appear
    :param comparison_col_name: String. The column containing 2 groups to compare their means
    :param comparison_order: List. The order to compare the two groups. i.e. if group 1 mean > group 2 mean or the other direction
    :param category_col_name: String. The column to be used to focus the analysis within
    :param category_order: List. The order to place the categories. If not supplied this will be inferred from the df
    :param minimum_size: Int. Specifies the minimum required sample size when displaying results. Defaults to 5
    :param annot_loc: Float. Controls where the annotations are. The larget the value the more the annotations go to the right
    :param annotate_pvalue_significance. Boolean. Default True
    :param output_diagnostics: Boolean. Default False
    :param significance_level: Float. Default 0.01
    :param significance_level2: Float. Default 0.05. Used for annotation of 1, 2 or 3 stars
    :param significance_level3: Float. Default 0.1. Used for annotation of 1, 2 or 3 stars
    :param footnote_x: The x coordinate of the footnote
    :param footnote_y: The y coordinate of the footnote
    :param footnote_size: The label size of the footnote
    :param dp: int. the decimal places to display the mean diffs
    
    :returns: fig
    
    Example:
    from ds_modules_101 import Plotting as dsp
    from ds_modules_101.Data import titanic_df
    import pandas as pd

    bucket_col_name = 'Pclass'
    #bucket_order=['1','2','3']
    category_col_name = 'Sex'
    comparison_col_name = 'Embarked'
    category_order = ['male','female']
    comparison_order = ['C','S']
    value_col_name='Fare'

    # get only specific columns
    temp = titanic_df[[bucket_col_name,category_col_name,comparison_col_name,value_col_name]].copy()

    # filter to have only sex in these
    temp = temp[temp[category_col_name].isin(category_order)][[bucket_col_name,category_col_name,comparison_col_name,value_col_name]].copy()


    f = Heatmap_group_difference(temp,bucket_col_name,comparison_col_name,category_col_name,value_col_name,
        bucket_order=bucket_order,comparison_order=comparison_order,category_order=category_order,minimum_size = 5,annot_loc = 10,annotate_pvalue_significance = True,
                                       output_diagnostics = False,significance_level = 0.05)
    
    '''
    single_star = False
    if max([significance_level,significance_level2,significance_level3]) == significance_level:
        single_star = True
        if output_diagnostics:
            print('Only using single star pvalue annotation because the significance levels provided are not incremental')
    elif max([significance_level,significance_level2]) == significance_level:
        single_star = True
        if output_diagnostics:
            print('Only using single star pvalue annotation because the significance levels provided are not incremental')
    elif max([significance_level2,significance_level3]) == significance_level2:
        single_star = True
        if output_diagnostics:
            print('Only using single star pvalue annotation because the significance levels provided are not incremental')
    
    if category_order is None:
        category_order = list(df[category_col_name].unique())
        
    if bucket_order is None:
        bucket_order = list(df[bucket_col_name].unique())
        
    if comparison_order is None:
        comparison_order = list(df[comparison_col_name].unique())

    # remove any rows that are missing entries
    temp = df[~df.isna().any(axis=1)].copy()

    for col in [bucket_col_name,comparison_col_name,category_col_name]:
        temp[col] = temp[col].astype('str')
    ### For each category col and bucket col, we need to calculate the difference between the comparison column of
    ### the value column
    
    value_col_name_mean = value_col_name + '_mean'
    value_col_name_count = value_col_name + '_count'
    value_col_name_std = value_col_name + '_std'
    function_dict = {value_col_name_mean: "mean", value_col_name_count: "count",value_col_name_std: "std"}
    temp[value_col_name_mean] = temp[value_col_name]
    temp[value_col_name_count] = temp[value_col_name]
    temp[value_col_name_std] = temp[value_col_name]

    temp_grp = temp.groupby(by=[comparison_col_name,bucket_col_name,category_col_name]).aggregate(function_dict).reset_index()
    
    temp_merged = pd.merge(left=temp_grp[temp_grp[comparison_col_name]==comparison_order[0]].drop(comparison_col_name,axis=1),
         right=temp_grp[temp_grp[comparison_col_name]==comparison_order[1]].drop(comparison_col_name,axis=1),
        on=[bucket_col_name,category_col_name],suffixes=('_x','_y'),how='inner')
    
    temp_merged['Diff_mean']=temp_merged[value_col_name_mean+'_x']-temp_merged[value_col_name_mean+'_y'] 
    
    temp_merged['pvalue'] = 1
    for i in list(temp_merged[bucket_col_name].unique()):
        for j in list(temp_merged[category_col_name].unique()):
            cohort1 = temp[(temp[bucket_col_name]==i) & (temp[category_col_name]==j) & (temp[comparison_col_name]==comparison_order[0])][value_col_name]
            cohort2 = temp[(temp[bucket_col_name]==i) & (temp[category_col_name]==j) & (temp[comparison_col_name]==comparison_order[1])][value_col_name]
            pvalue = stats.ttest_ind(cohort1.dropna(),cohort2.dropna(), equal_var = False)[1]
            temp_merged.loc[(temp_merged[bucket_col_name]==i) & (temp_merged[category_col_name]==j),'pvalue'] = pvalue
            mean1 = temp_merged.loc[(temp_merged[bucket_col_name]==i) & (temp_merged[category_col_name]==j),value_col_name_mean+'_x'].reset_index().iloc[0,1]
            mean2 = temp_merged.loc[(temp_merged[bucket_col_name]==i) & (temp_merged[category_col_name]==j),value_col_name_mean+'_y'].reset_index().iloc[0,1]
            diff_mean = temp_merged.loc[(temp_merged[bucket_col_name]==i) & (temp_merged[category_col_name]==j),'Diff_mean'].reset_index().iloc[0,1]
    
            if output_diagnostics:
                print('{0} and {1}: \npvalue = {2} \n{3}count_{4}count = {5}_{6} \nmeans = {7} vs {8}  \ndiff = {9}\n\n'.format(i,j,pvalue,comparison_order[0],comparison_order[1],len(cohort1),len(cohort2),mean1,mean2,diff_mean))
    
    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(1,1,1)
    df = pd.pivot_table(data=temp_merged,
                        index=category_col_name,
                        values='Diff_mean',
                        columns=bucket_col_name)
    sns.heatmap(df,annot=False, cmap="RdYlGn",annot_kws={"size": 18},ax=ax,vmin=None, vmax=None)
    
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 20,rotation=45)
    ax.set_xlabel(ax.get_xlabel(), fontsize = 20)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 20)
    title = 'Difference in means of {3} between {0} = {1} and {2} ({1}:{2})'.format(comparison_col_name,comparison_order[0],comparison_order[1],value_col_name)

    title = title + ' '
    ax.set_title(title)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 20,rotation=45)

    # Get axis labels and locations
    xlabs = list(map(lambda x: x.get_text(),ax.get_xticklabels()))
    xtiks = list(ax.get_xticks())
    ylabs = list(map(lambda x: x.get_text(),ax.get_yticklabels()))
    ytiks = list(ax.get_yticks())
    
    for ix,xlab in enumerate(xlabs):
        for iy,ylab in enumerate(ylabs):
            len1=temp_merged[(temp_merged[category_col_name]==ylab) & (temp_merged[bucket_col_name]==xlab)][value_col_name_count+'_x'].reset_index().iloc[0,1]
            len2=temp_merged[(temp_merged[category_col_name]==ylab) & (temp_merged[bucket_col_name]==xlab)][value_col_name_count+'_y'].reset_index().iloc[0,1]
            pvalue=temp_merged[(temp_merged[category_col_name]==ylab) & (temp_merged[bucket_col_name]==xlab)]['pvalue'].reset_index().iloc[0,1]
            diff_mean=temp_merged[(temp_merged[category_col_name]==ylab) & (temp_merged[bucket_col_name]==xlab)]['Diff_mean'].reset_index().iloc[0,1]

            if annotate_pvalue_significance:
                if (len1 > minimum_size) and (len2 > minimum_size):
                    if pvalue <= significance_level:
                        #sns.scatterplot(x=[xtiks[ix] + (xtiks[1]-xtiks[0])/4],y=[ytiks[iy]],s=200,marker="***",color='black',ax=ax)
                        plt.text(x=xtiks[ix] + (xtiks[1]-xtiks[0])/4,y=ytiks[iy], s='***', fontsize=annotation_font_size)
                    elif (pvalue <= significance_level2) and not single_star:
                        #sns.scatterplot(x=[xtiks[ix] + (xtiks[1]-xtiks[0])/4],y=[ytiks[iy]],s=200,marker="**",color='black',ax=ax)
                        plt.text(x=xtiks[ix] + (xtiks[1]-xtiks[0])/4,y=ytiks[iy], s='**', fontsize=annotation_font_size)
                        a=1
                    elif (pvalue <= significance_level3) and not single_star:
                        #sns.scatterplot(x=[xtiks[ix] + (xtiks[1]-xtiks[0])/4],y=[ytiks[iy]],s=200,marker="*",color='black',ax=ax)
                        plt.text(x=xtiks[ix] + (xtiks[1]-xtiks[0])/4,y=ytiks[iy], s='*', fontsize=annotation_font_size)

            if (len1 > minimum_size) and (len2 > minimum_size):
                plt.text(x=xtiks[ix] - (xtiks[1]-xtiks[0])/annot_loc,y=ytiks[iy], s='{}'.format(np.round(diff_mean,dp)), fontsize=annotation_font_size)

        t = 'Any cell not appearing has sample size < {0} for one of the groups\n'.format(minimum_size)

    if annotate_pvalue_significance:
        if single_star:
            t = t + 'Pvalue: A *** indicates significance between {0} groups at the {1} level'.format(comparison_col_name,significance_level)
        else:
            t = t + 'Pvalue: A *, ** and *** indicates significance between {0} groups at the {1}, {2}, {3} levels respectively'.format(comparison_col_name,significance_level3,significance_level2,significance_level)


    a=fig.text(footnote_x,footnote_y,t,size=footnote_size)


    cbar = ax.collections[0].colorbar
    ts = cbar.get_ticks()
    cbar.set_ticks(ts)
    cbar.set_ticklabels(['{}:{}'.format(100-int(i),int(i)) for i in ts])
    cbar.ax.tick_params(labelsize=ticks_font_size)
    
    return fig
    
def Heatmap_Survey_group_difference_v2(df,unique_choices,comparison_col_name,comparison_order,choice_col_name,category_col_name,category_order = None,is_multi_choice=False,
                                   minimum_size = 5,annot_loc = 10,annotate_dprime = True,annotate_mean_diffs = False,annotate_pvalue_significance = True,
                                   output_diagnostics = False,significance_level = 0.01,significance_level2 = 0.05,significance_level3 = 0.1,footnote_x = 0, footnote_y = -0.5,footnote_size=13,annotation_font_size=16,
                                   ticks_font_size=20,custom_title = None, title_append = ''):

    '''
    This function is a heatmap of survey data representing response choices to a particular question. For each category, it
    looks at whether there is a difference in the choice between 2 comparison groups. There is functionality to output pvalues
    based on the proportion (mean) that group has chosen this choice as well as d prime results comparing the relative 
    potency of that choice within each of the groups. The function does not care if this is a single choice question or multiple.

    :param df: The dataframe
    :param unique_choices: List of unique choices in the choice_col_name column we are interested in
    :param comparison_col_name: String. This column should contain at least 2 groups. This column will be used to compare choice answers between
    :param comparison_order: List of size 2. This must be supplied. The 2 groups to consider
    :param choice_col_name: String. The column name representing the choice
    :param category_col_name: String. The column to be used to focus the analysis within
    :param is_multi_choice: A boolean specifying whether choice_col_name is multi choice. If choice_col_name contains a list per row then it is multi choice
    :param category_order: List. The order to place the categories. If not supplied this will be inferred from the df
    :param minimum_size: Int. Specifies the minimum required sample size when displaying results. Defaults to 5
    :param annot_loc: Float. Controls where the annotations are. The larget the value the more the annotations go to the right
    :param annotate_dprime: Boolean. Default True
    :param annotate_mean_diffs: Boolean. Default False
    :param annotate_pvalue_significance. Boolean. Default True
    :param output_diagnostics: Boolean. Default False
    :param significance_level: Float. Default 0.01
    :param significance_level2: Float. Default 0.05. Used for annotation of 1, 2 or 3 stars
    :param significance_level3: Float. Default 0.1. Used for annotation of 1, 2 or 3 stars
    :param footnote_x: The x coordinate of the footnote
    :param footnote_y: The y coordinate of the footnote
    :param footnote_size: The label size of the footnote
    
    :returns: fig
    
    Example Usage:
    from ds_modules_101 import Plotting as dsp
    from ds_modules_101.Data import titanic_df
    import pandas as pd
    
    # get only specific columns
    temp = titanic_df[[pclass,sex,embarked]].copy()

    # filter to have only sex in these
    temp = temp[temp[sex].isin([male,female])][[sex,pclass,embarked]].copy()


    choice_col_name = 'Pclass'
    category_col_name = 'Sex'
    comparison_col_name = 'Embarked'

    category_order = ['male','female']

    comparison_order = ['C','S']
    
    f = dsp.Heatmap_Survey_group_difference(temp,comparison_col_name,comparison_order,choice_col_name,category_col_name,category_order = None,
                                       minimum_size = 5,annot_loc = 10,annotate_dprime = True,annotate_mean_diffs = False,annotate_pvalue_significance = True,
                                       output_diagnostics = False,significance_level = 0.05)
    '''
    single_star = False
    if max([significance_level,significance_level2,significance_level3]) == significance_level:
        single_star = True
        if output_diagnostics:
            print('Only using single star pvalue annotation because the significance levels provided are not incremental')
    elif max([significance_level,significance_level2]) == significance_level:
        single_star = True
        if output_diagnostics:
            print('Only using single star pvalue annotation because the significance levels provided are not incremental')
    elif max([significance_level2,significance_level3]) == significance_level2:
        single_star = True
        if output_diagnostics:
            print('Only using single star pvalue annotation because the significance levels provided are not incremental')
    
    if category_order is None:
        category_order = list(df[category_col_name].unique())

    # remove any rows that are missing entries
    temp = df[~df.isna().any(axis=1)].copy()

    # create a column for each choice which specifies whether this class was the choice
    for col in unique_choices:
        if is_multi_choice:
            temp[col] = temp[choice_col_name].apply(lambda x: 1 if col in x else 0)
        else:
            temp[col] = temp[choice_col_name].apply(lambda x: 1 if col == x else 0)
    
    # get the order of the class
    order = sorted(unique_choices)

    # add a count column
    temp['Count'] = 1
    
    # get the total number of people in each of these groups as well as the total number of people who are each choice_col_name
    mean = temp[[category_col_name,comparison_col_name] + unique_choices].groupby(by = [category_col_name,comparison_col_name]).mean().reset_index()

    # get the total in 2 fields
    totals = temp.groupby(by = [category_col_name,comparison_col_name]).sum().reset_index()
    if choice_col_name in list(totals.columns):
        totals.drop(choice_col_name,axis=1,inplace=True)
    
    if comparison_order is None:
        comparison_order = list(temp[comparison_col_name].unique())
        
    # join the totals onto the groups
    grp = pd.merge(left = mean,right = totals, on=[category_col_name,comparison_col_name],suffixes=('_mean','_sum'))

    # join the group table to get differences between category_col_name
    grp_cross_ethnicity = pd.merge(left=grp,right=grp,on=[category_col_name])

    grp_cross_ethnicity = grp_cross_ethnicity[(grp_cross_ethnicity[comparison_col_name+'_x'] == comparison_order[0]) & (grp_cross_ethnicity[comparison_col_name+'_y'] == comparison_order[1])]

    grp_cross_ethnicity = grp_cross_ethnicity[(grp_cross_ethnicity['Count_x'] > minimum_size) & (grp_cross_ethnicity['Count_y'] > minimum_size)]
    
    for col in unique_choices:
        grp_cross_ethnicity[col+'_ScaledProp'] = grp_cross_ethnicity[[col+'_mean_x',col+'_mean_y']].apply(lambda x: round(x[0]*100/(x[0]+x[1]),2),axis=1)
    
    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(1,1,1)
    temp2 = grp_cross_ethnicity[[category_col_name]+list([col for col in grp_cross_ethnicity.columns if 'ScaledProp' in col])].copy()
    temp2.set_index(category_col_name,inplace=True)
    temp2.columns = list(map(lambda x: x.replace('_ScaledProp',''),temp2.columns))
    sns.heatmap(temp2,annot=False, cmap="RdYlGn",annot_kws={"size": 18},ax=ax,vmin=20, vmax=80)

    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 20,rotation=45)
    ax.set_xlabel(ax.get_xlabel(), fontsize = 20)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 20)
    title = 'Proportion importance between {0} = {1} and {2} ({1}:{2})'.format(comparison_col_name,comparison_order[0],comparison_order[1])
    if custom_title is not None:
        title = custom_title
    title = title + ' ' + title_append
    ax.set_title(title)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 20,rotation=45)

    # Get axis labels and locations
    xlabs = list(map(lambda x: x.get_text(),ax.get_xticklabels()))
    xtiks = list(ax.get_xticks())
    ylabs = list(map(lambda x: x.get_text(),ax.get_yticklabels()))
    ytiks = list(ax.get_yticks())

    # Draw significance
    stress = comparison_order

    for ix,xlab in enumerate(xlabs):
            for iy,ylab in enumerate(ylabs):
                cohort1 = temp[(temp[comparison_col_name] == comparison_order[0]) & (temp[category_col_name] == ylab)]
                cohort2 = temp[(temp[comparison_col_name] == comparison_order[1]) & (temp[category_col_name] == ylab)]

                pos1_len = cohort1[xlab].sum()
                pos2_len = cohort2[xlab].sum()

                cohort1_len = len(cohort1)
                cohort2_len = len(cohort2) 

                prop1 = pos1_len/cohort1_len
                prop2 = pos2_len/cohort2_len

                scaledprop1 = prop1/(prop1 + prop2)
                scaledprop2 = prop2/(prop1 + prop2)

                pvalue = stats.ttest_ind(cohort1[xlab],cohort2[xlab], equal_var = False)[1]

                if output_diagnostics:
                    print('{0} and {1}: \npvalue = {2} \n{3}count_{4}count = {5}_{6} \n{3}true_{4}true = {7}_{8} \nproportions = {9}:{10} \nScaled Proportions = {11}:{12} \nd-prime significant: {13}\n\n'.format(xlab,ylab,pvalue,comparison_order[0],comparison_order[1],cohort1_len,cohort2_len,pos1_len,pos2_len,prop1,prop2,scaledprop1,scaledprop2,(scaledprop1 >= 0.6) or (scaledprop2 >= 0.6)))

                if annotate_pvalue_significance:
                    if (pos1_len > minimum_size) and (pos2_len > minimum_size):
                        if pvalue <= significance_level:
                            #sns.scatterplot(x=[xtiks[ix] + (xtiks[1]-xtiks[0])/4],y=[ytiks[iy]],s=200,marker="***",color='black',ax=ax)
                            plt.text(x=xtiks[ix] + (xtiks[1]-xtiks[0])/4,y=ytiks[iy], s='***', fontsize=annotation_font_size)
                        elif (pvalue <= significance_level2) and not single_star:
                            #sns.scatterplot(x=[xtiks[ix] + (xtiks[1]-xtiks[0])/4],y=[ytiks[iy]],s=200,marker="**",color='black',ax=ax)
                            plt.text(x=xtiks[ix] + (xtiks[1]-xtiks[0])/4,y=ytiks[iy], s='**', fontsize=annotation_font_size)
                            a=1
                        elif (pvalue <= significance_level3) and not single_star:
                            #sns.scatterplot(x=[xtiks[ix] + (xtiks[1]-xtiks[0])/4],y=[ytiks[iy]],s=200,marker="*",color='black',ax=ax)
                            plt.text(x=xtiks[ix] + (xtiks[1]-xtiks[0])/4,y=ytiks[iy], s='*', fontsize=annotation_font_size)

                if annotate_dprime:
                    if (pos1_len > minimum_size) and (pos2_len > minimum_size):
                        plt.text(x=xtiks[ix] - (xtiks[1]-xtiks[0])/annot_loc,y=ytiks[iy], s='{}:{}'.format(int(np.round(scaledprop1*100,0)),int(np.round(scaledprop2*100,0))), fontsize=annotation_font_size)
                elif annotate_mean_diffs:
                    if (pos1_len > minimum_size) and (pos2_len > minimum_size):
                        plt.text(x=xtiks[ix] - (xtiks[1]-xtiks[0])/annot_loc,y=ytiks[iy], s='{}'.format(np.round(prop1-prop2,1)), fontsize=annotation_font_size)

    t = 'Any cell not appearing has sample size < {0} for one of the groups\n'.format(minimum_size)

    if annotate_dprime:
        t = t + 'Annotations: d prime scaled proportions \nColour: Red indicates that the choice is more potent in {0}\n'.format(comparison_order[0])

    if annotate_pvalue_significance:
        if single_star:
            t = t + 'Pvalue: A *** indicates significance between {0} groups at the {1} level'.format(comparison_col_name,significance_level)
        else:
            t = t + 'Pvalue: A *, ** and *** indicates significance between {0} groups at the {1}, {2}, {3} levels respectively'.format(comparison_col_name,significance_level3,significance_level2,significance_level)


    a=fig.text(footnote_x,footnote_y,t,size=footnote_size)


    cbar = ax.collections[0].colorbar
    ts = cbar.get_ticks()
    cbar.set_ticks(ts)
    cbar.set_ticklabels(['{}:{}'.format(100-int(i),int(i)) for i in ts])
    cbar.ax.tick_params(labelsize=ticks_font_size)
    
    return fig
