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