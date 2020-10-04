# Libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    
    Example Usage:
    from ds_modules_101 import Plotting as dsp
    from ds_modules_101.Data import titanic_df
    
    data = titanic_df.copy()
    data = data[['Survived','Pclass','SibSp','Parch','Fare']].copy()
    columns = ['Survived','Pclass','SibSp','Parch','Fare'] 
    corr = data[columns].corr()
    corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']
    a = heatmap(
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

def correlation_heatmap(df,columns,**kwargs):
    '''
    A function to just do a correlation heatmap between the specified columns
    :param df: The data frame object
    :param columns: A list of column names to do a correlation between
    :param kwargs: Key word arguments to be passed to the heatmap function
    
    :return fig: A figure object
    
    Example Usage:
    from ds_modules_101 import Plotting as dsp
    from ds_modules_101.Data import titanic_df
    import pandas as pd
    
    dsp.correlation_heatmap(titanic_df,['Survived','Pclass','SibSp','Parch','Fare'],figuresize=(10, 10))
    '''
    
    data = df[columns].copy()
    
    # get a correlation matrix between all columns
    corr = data.corr()
    
    # convert the dataframe by unpivotting so that the first 2 columns are the pair of names and the 3rd column their correlation
    corr = pd.melt(corr.reset_index(), id_vars='index')
    
    # rename the columns
    corr.columns = ['x', 'y', 'value']
    
    # get the heatmap
    fig = heatmap(
        x=corr['x'],
        y=corr['y'],
        value=corr['value'],**kwargs
    )
    return fig