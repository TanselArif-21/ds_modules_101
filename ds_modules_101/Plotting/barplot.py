import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def stacked_bar_chart(df_in,x_col,y_col,category_col = None,hex_numbers = None,figsize=(10,10)):
    '''    
    This function return a stacked barchart for each category. If the category is None, the a single stacked barchart is 
    given.
    
    :param df_in: The dataframe
    :param x_col: String. The name of the column to be along the x axis. Should be categories and not real numbers.
    :param y_col: String. The name of the column to be the different segments. Should be categories and not real numbers.
    :param category_col: String. The name of the column to be the different categories. Should be categories and not real numbers. If not given, only a single stacked barchart will be created.
    :param hex_numbers: A list of hex numbers corresponding to the different number of segments (y_col). If not supplied, the hex numbers are automatically chosen
    :param figsize: A tuple. i.e. (10,10)
    :param barWidth: The width of the bars.
    
    :returns: fig
    
    Example Usage:
    
    from ds_modules_101 import Plotting as dsp
    from ds_modules_101.Data import titanic_df
    import pandas as pd


    # get only specific columns
    temp = titanic_df[['Pclass','Sex','Age','Embarked']].copy()

    temp['AgeBand'] = temp['Age'].apply(lambda x: '<18' if x < 18 else '18-24' if x < 25 else '25-34' if x < 35 else '35+')
    
    x_col = 'AgeBand'
    y_col = 'Pclass'
    category_col = 'Sex'
    hex_numbers = ['#b5ffb9','#f9bc86','#a3acff']

    f=stacked_bar_chart(temp,x_col,y_col,category_col,hex_numbers = hex_numbers)
    '''
    
    # take a copy
    temp = df_in.copy()
    
    # if a category is not given, create a dummy one
    if category_col is None:
        category_col = 'DummyCatColumn'
        temp[category_col] = 'Stacked Bar Chart'
        
    # make a copy of the dataframe for only columns we need then sort it for constistency
    temp = temp[[x_col,y_col,category_col]].copy()
    temp['count'] = 1
    temp2 = temp.groupby(by=[x_col,y_col,category_col]).aggregate({'count':np.sum}).reset_index()
    temp2.sort_values(by=[x_col,y_col,category_col],ascending=True,inplace=True)

    # get the unique segments per bar and the unique x columns
    segs = list(temp2[y_col].unique())
    xs = list(temp2[x_col].unique())
    cats = list(temp2[category_col].unique())
    
    
    # get the list of x locations on each plot
    r = list(range(len(xs)))

    # get hex colours if not given
    if hex_numbers is None:
        colours_ints=list(map(lambda x: int(x),np.linspace(0,16777215,num=len(segs)+5)))
        hex_numbers = [str(hex(i)) for i in colours_ints]
        hex_numbers = list(map(lambda x:'#'+ x[2:],hex_numbers))
        hex_numbers = hex_numbers[1:]

    # create a figure and set the figure size
    fig = plt.figure(figsize=figsize)
    
    # for each category, create a stacked bar plot
    for l,et in enumerate(cats):
        # add a subplot
        ax = fig.add_subplot(len(cats),1,l+1)

        # build a dictionary which will contain values for each segment and convert it into a dataframe
        raw_data = dict()
        for i in segs:
            raw_data[i] = temp2[(temp2[category_col] == et) & (temp2[y_col] == i)]['count'].reset_index(drop=True)
        df = pd.DataFrame(raw_data)
        df.fillna(0,inplace=True)

        # get the total values per x xolumn
        totals = [i for i in df.sum(axis=1)]
        
        # convert the raw value to percentage for each segment
        bars = []
        for k,l in enumerate(list(raw_data.keys())):
            bars.append([i / j * 100 for i,j in zip(df[l], totals)])

        # create the first segment on the bottom
        sns.barplot(x=r, y=bars[0], color=hex_numbers[0], edgecolor='white',ax=ax, label='{}'.format(segs[0]))

        # for each subsequent segment, use the previous segment as the bottom
        for idx,bar in enumerate(zip(bars[1:],segs[1:])):
            sns.barplot(x=r, y=bars[idx+1], bottom=[sum(i) for i in list(zip(*(bars[0:idx+1])))], color=hex_numbers[idx+1], edgecolor='white',ax=ax, label='{}'.format(bar[1]))

        #  custom x axis
        plt.xticks(r, xs)
        ax.set_title(et)

    # set the overall x label and sort out the legend
    plt.xlabel(x_col)
    fig.get_axes()[0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1,title=y_col)
    fig.tight_layout()
    
    return fig

def stacked_bar_chart2(t,x_axis_name,col_to_stack_name,y_label = 'Proportion', x_label=None,
                       title = 'Stacked bar plot',colormap = 'Greys',title_color = 'grey',
                       suptitle = 'Stacked bar plot',suptitle_color = 'grey',legend_title = '',
                       show_value_threshold = 0,legend_loc=(1.2, 0.9),order_col=None,
                       x_label_size=15,y_label_size=15,title_size=10,suptitle_size=15,
                       annotation_size=20,legend_size=15,xtick_size=15,annotation_weight='bold',width = 0.70,
                       figsize=(10,10),suptitle_loc=None,calculation_column=None,calculation_method='sum',
                       normalise_to_1=True,list_of_colors=None,diagnostic=False):

    '''
    This function plots a fancy stacked bar chart and takes away a lot of the pre-processing required from the user.

    :param t: A dataframe
    :param x_axis_name: String. X-axis column
    :param col_to_stack_name: String. Column to stack
    :param y_label: String. The y axis label
    :param x_label: String. The x axis label
    :param title: String. Plot title
    :param colormap: String. The colormap to use. Examples are 'viridix','Greys','YlGn'. More here -> https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :param title_color: String. The color of the title. i.e. 'black', 'blue', 'grey'...
    :param suptitle: String. The overall title
    :param suptitle_color: String. The color of the suptitle. i.e. 'black', 'blue', 'grey'
    :param legend_title: String. The title for the legend
    :param show_value_threshold: Float. Some stacks might be too small to display the values. Omit values below this threshold
    :param legend_loc: Tuple. Specify the location (from top left) of the legend. i.e. (1.2,1)
    :param order_col: String. The column to order the data by
    :param x_label_size: Float. The size of the x label
    :param y_label_size: Float. The size of the y label
    :param title_size: Float. The title size
    :param suptitle_size: Float. The suptitle size
    :param annotation_size: Float. The size of the annotations
    :param legend_size: Float. The size of the legend text
    :param xtick_size: Float. The size of the xtick labels
    :param annotation_weight: String. The weight of the annotation font. e.g. 'bold'
    :param width: Float. The width of the bars
    :param figsize: Tuple. The size of the figure. e.g (10,10)
    :param suptitle_loc: Tuple. The x/y coords of the suptitle
    :param calculation_column: String. The column to use as the calculation
    :param calculation_method: String. The method to use for the calculation of the stack bars. e.g. 'mean','sum','count'
    :param normalise_to_1: Bool. Whether or not each stacked bar should add to 1
    :param list_of_colors: List. List of hex colors to apply to each stack

    EXAMPLE USAGE:
    from ds_modules_101 import Plotting as dsp
    from ds_modules_101.Data import titanic_df
    import pandas as pd


    # get only specific columns
    df = titanic_df[['Pclass','Sex','Age','Embarked']].copy()

    fig = stacked_bar_chart2(df, x_axis_name='Pclass', col_to_stack_name='Embarked', y_label='Proportion',
                             title='Stacked bar plot', colormap='YlGn', title_color='grey',
                             suptitle='Stacked bar plot', suptitle_color='grey', legend_title='legend title',
                             show_value_threshold=3, order_col='Pclass')
    #fig.savefig('dfafd.png', bbox_inches='tight')
    plt.show()
    '''

    # get a colormap for the stacking colors
    from matplotlib import cm
    if colormap is not None:
        colormap = cm.get_cmap(colormap, 100)
    else:
        colormap = cm.get_cmap('Greys', 100)

    # default the x axis name to the column name
    if x_label is None:
        x_label = x_axis_name

    # order the data according to user specification
    if order_col is not None:
        if order_col not in t.columns:
            print('order by column not in columns list')
            t = t.sort_values(by=x_axis_name)
        else:
            t = t.sort_values(by=order_col)

    # get only the columns we need
    t = t[[x_axis_name, col_to_stack_name]].copy()

    # if there are nans label them
    t = t.fillna('NaN')

    # get the values to stack from the user. Otherwise just do a count
    if calculation_column is not None:
        t['Count'] = calculation_column
    else:
        t['Count'] = 1


    # get the lists of values in the x axis as well as the stacking axis
    cols = []
    x_axis_list = list(t[x_axis_name].unique())
    x_axis_list_string = list(map(lambda x: str(x), x_axis_list))
    col_to_stack_list = list(filter(lambda x: not pd.isna(x), t[col_to_stack_name].unique()))

    # for each value of the x axis get the values of the stacking axis
    for col_val in x_axis_list:
        # col1 = t[t[x_axis_name] == col_val][
        #     [col_to_stack_name]].value_counts().reset_index()

        # calculate the stacking values according to calculation method
        if calculation_method.lower() == 'sum':
            col1 = t[t[x_axis_name] == col_val][
                [col_to_stack_name,'Count']].groupby(by=[col_to_stack_name]).sum().reset_index()
        elif calculation_method == 'mean':
            col1 = t[t[x_axis_name] == col_val][
                [col_to_stack_name, 'Count']].groupby(by=[col_to_stack_name]).mean().reset_index()
        elif calculation_method == 'count':
            col1 = t[t[x_axis_name] == col_val][
                [col_to_stack_name, 'Count']].groupby(by=[col_to_stack_name]).count().reset_index()
        else:
            print('Unknown calculation method {}. Using sum instead'.format(calculation_method))
            col1 = t[t[x_axis_name] == col_val][
                [col_to_stack_name, 'Count']].groupby(by=[col_to_stack_name]).sum().reset_index()

        # rename the column as 'Count' if it was named as 0 (this might not be needed after groupby)
        col1 = col1.rename(columns={0: 'Count'})

        # get the total rows for this x axis category
        col1['Total'] = len(t[t[x_axis_name] == col_val])

        # scale so that each stacked bar adds to 1 if needed
        if normalise_to_1:
            col1['Proportion'] = col1['Count'] / col1['Total']
        else:
            col1['Proportion'] = col1['Count']
        #col1 = col1.sort_values(by=col_to_stack_name)
        cols.append(col1)

    # get the list of labels for the legend
    labels = list(map(lambda x: str(x), col_to_stack_list))

    # grab the calculated proportions and put them together into a 2 dimensional list (x-axis by stacked axis)
    proportions = []
    for val in col_to_stack_list:
        proportion = []
        for col in cols:
            # if there is an error, this val doesn't exist in this col. So put proportion 0
            try:
                proportion.append(col[col[col_to_stack_name] == val]['Proportion'].values[0])
            except:
                proportion.append(0)
                continue

        proportions.append(proportion)

    # get the figure/axis objects
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    # create a linear segue of colors
    color_list = np.linspace(0.3, 0.8, len(col_to_stack_list))

    # this for loop draws the bars. i goes along the x axis and j goes along the stacked axis.
    # Bottom is where the next bar starts drawing in that same x axis
    for i in list(range(len(proportions)))[-1::-1]:
        bottom = 0
        for j in range(i):
            bottom = np.array(proportions[j]) + bottom

        if list_of_colors is None:
            ax.bar(x_axis_list_string, proportions[i], width, bottom=bottom,
                   label=labels[i], color=colormap(color_list[i]))
        else:
            ax.bar(x_axis_list_string, proportions[i], width, bottom=bottom,
                   label=labels[i], color=list_of_colors[i])

    # figure aesthetics
    ax.set_ylabel(y_label, fontsize=y_label_size)
    ax.set_xlabel(x_label, fontsize=x_label_size)
    # ax.set_title(title, loc='left', fontsize=15, color=title_color)
    ax.legend(loc='upper right', bbox_to_anchor=legend_loc, fontsize=legend_size, title=legend_title, borderaxespad=0.)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.get_yaxis().set_ticks([])
    plt.xticks(fontsize=xtick_size)
    if suptitle_loc is None:
        suptitle_loc = (ax.get_position().x0,ax.get_position().y1 + 0.05)
    plt.suptitle(suptitle, fontsize=suptitle_size, color=suptitle_color,
                 horizontalalignment='left',
                 verticalalignment='top', x=suptitle_loc[0], y=suptitle_loc[1])
    plt.title(title, loc='left', fontsize=title_size, color=title_color)

    # this loop annotates. i goes along the x axis and j goes along the stacked axis. k loops through the previous
    # stacks in that same x axis category so it can reach the middle of the stack bar it needs to annotate to
    for i in list(range(len(cols))):
        for j in range(len(col_to_stack_list))[-1::-1]:
            text_loc = 0
            for k in range(j + 1):
                if k == j:
                    text_loc = text_loc + proportions[k][i] / 2
                else:
                    text_loc = text_loc + proportions[k][i]

            if proportions[j][i] * 100 > show_value_threshold:
                ax.text(i, text_loc,
                        '{}%'.format(int(np.round(proportions[j][i] * 100, 0))),
                        ha='center', va='center', color='white', fontsize=annotation_size, fontweight=annotation_weight)

    if diagnostic:
        t_counts = t[[x_axis_name,col_to_stack_name]].value_counts().reset_index()
        t_counts = t_counts.rename(columns={0:'Counts'})
        print(t_counts)

    #
    # ax.text(0.5, 0.2,s , color='black',
    #         bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

    return fig

def show_values_on_bars(axs,fontsize=20,color='black',decimal_places = 2):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
            v = round(p.get_height(),decimal_places)
            if decimal_places == 0:
                v = int(v)
            value = '{}'.format(v)
            ax.text(_x,_y,value,ha="center",fontsize=fontsize,color=color)

    if isinstance(axs,np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

def show_values_on_bars_h(axs,fontsize=20,color='black',decimal_places = 2):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _y = p.get_y()+ (p.get_height()/2)
            _x = p.get_x()+ p.get_width()*1.02
            v = round(p.get_width(),decimal_places)
            if decimal_places == 0:
                v = int(v)
            value = '{}'.format(v)
            ax.text(_x,_y,value,ha="center",va='center',fontsize=fontsize,color=color,rotation=-90)

    if isinstance(axs,np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def values_on_bars_test_1():
    a = pd.Series(np.random.randint(low=0, high=100, size=100))
    b = pd.Series(['Group1', 'Group2'] * 50)
    df = pd.DataFrame([a, b]).T
    df.columns = ['Value', 'Column']
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    sns.barplot(data=df, x='Column', y='Value', ax=ax)
    show_values_on_bars(ax, decimal_places=0)
    fig.show()

def values_on_bars_test_2():
    a = pd.Series(np.random.randint(low=0, high=100, size=100))
    b = pd.Series(['Group1', 'Group2'] * 50)
    df = pd.DataFrame([a, b]).T
    df.columns = ['Value', 'Column']
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    sns.barplot(data=df, y='Column', x='Value', ax=ax, orient='h')
    show_values_on_bars_h(ax, decimal_places=0)
    fig.show()

def stacked_barplot_test_1():
    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)

    # get only specific columns
    temp = df[['Pclass', 'Sex', 'Age', 'Embarked']].copy()

    temp['AgeBand'] = temp['Age'].apply(
        lambda x: '<18' if x < 18 else '18-24' if x < 25 else '25-34' if x < 35 else '35+')

    x_col = 'AgeBand'
    y_col = 'Pclass'
    category_col = 'Sex'
    hex_numbers = ['#b5ffb9', '#f9bc86', '#a3acff']

    f = stacked_bar_chart(temp, x_col, y_col, category_col, hex_numbers=hex_numbers)

def stacked_bar_plot_test_2():
    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)

    fig = stacked_bar_chart2(df, x_axis_name='Pclass', col_to_stack_name='Embarked', y_label='Proportion',
                             title='Stacked bar plot', colormap='YlGn', title_color='grey',
                             suptitle='Stacked bar plot', suptitle_color='grey', legend_title='legend title',
                             show_value_threshold=3, order_col='Pclass',diagnostic=True)
    #fig.savefig('dfafd.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':

    import os
    import sys
    from matplotlib import cm

    values_on_bars_test_1()
    values_on_bars_test_2()
    stacked_barplot_test_1()
    stacked_bar_plot_test_2()
