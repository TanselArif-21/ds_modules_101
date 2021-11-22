import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os,sys


def cleveland_plot(df,group_by,grp_col,val_col,groups,sortbycol=None,min_group_size=20,height = 10,width=10,number_of_splits = 5,
                    legend_x = 1.05,legend_y = 0,legend_x2 = 1.05,legend_y2 =1,labelsize = 1,
                    xlim = None, ylim = None):
    
    '''
    Example Usage:
    from ds_modules_101 import Plotting as dsp
    from ds_modules_101 import Data as dsd
    df = dsd.titanic_df

    # this code looks at differences in 'Fare' between 'male' and 'female' in column 'Sex' split by 'Pclass'
    dsp.cleveland_plot(df=df,group_by='Pclass',grp_col='Sex',val_col='Fare',groups=['male','female'])
    '''

    group_by = group_by
    grp_col = grp_col
    grp1 = groups[0]
    grp2 = groups[1]
    sortbycol = sortbycol

    t = df.copy()
    t['count'] = 1
    t = t[['count',val_col]+[group_by]+[grp_col]].groupby(by=[group_by]+[grp_col]).agg({val_col:np.mean,'count':sum}).reset_index()
    #t = t[t['count']>=20].copy()
    t_men = t[t[grp_col]==grp1].copy()
    t_men.rename(columns={val_col:val_col+'_{}'.format(grp1),'count':'count_{}'.format(grp1)},inplace=True)
    t_women = t[t[grp_col]==grp2].copy()
    t_women.rename(columns={val_col:val_col+'_{}'.format(grp2),'count':'count_{}'.format(grp2)},inplace=True)
    t = pd.merge(left=t_women,right=t_men,on=group_by)
    t = t[(t['count_{}'.format(grp2)] >= min_group_size) & (t['count_{}'.format(grp1)] >= min_group_size)].copy()
    out_t = t.copy()

    # Reorder it following the values of the first value:
    if sortbycol is not None:
        if sortbycol == val_col:
            ordered_df = t.sort_values(by=val_col+'_{}'.format(grp2))
        else:
            ordered_df = t.sort_values(by=sortbycol)
    else:
        ordered_df = t.copy()
    
    my_range=range(1,len(t)+1)

    fig = plt.figure(figsize=(width,height))
    ax = fig.add_subplot(1,1,1)
    # The horizontal plot is made using the hline function
    ax.hlines(y=my_range, xmin=ordered_df[val_col+'_{}'.format(grp2)], xmax=ordered_df[val_col+'_{}'.format(grp1)], color='grey', alpha=0.4)
    scatter = ax.scatter(ordered_df[val_col+'_{}'.format(grp2)], my_range, color='skyblue', alpha=1, label=grp2,s=ordered_df['count_{}'.format(grp2)]*number_of_splits)
    ax.scatter(ordered_df[val_col+'_{}'.format(grp1)], my_range, color='green', alpha=0.4 , label=grp1,s=ordered_df['count_{}'.format(grp1)]*number_of_splits)
    ax.add_artist(ax.legend(bbox_to_anchor=(legend_x2, legend_y2), loc="lower right",fontsize=20*labelsize))

    # Add title and axis names
    plt.xticks(fontsize=16*labelsize)
    plt.yticks(my_range, ordered_df[group_by],fontsize=15*labelsize)
    ax.set_title("Gap in {}".format(val_col),fontsize=20*labelsize)
    ax.set_xlabel(val_col,fontsize=20*labelsize)
    ax.set_ylabel(group_by,fontsize=20*labelsize)

    if xlim is not None:
        ax.set_xlim(xlim[0],xlim[1])

    if ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])

    min_women = t['count_{}'.format(grp2)].min()
    max_women = t['count_{}'.format(grp2)].max()
    min_men = t['count_{}'.format(grp1)].min()
    max_men = t['count_{}'.format(grp1)].max()

    ls_sizes = sorted([min_men,min_women,max_men,max_women])
    ls_sizes = list(map(lambda x: x*number_of_splits,ls_sizes))

    kw = dict(prop="sizes", num=ls_sizes, color=scatter.cmap(0.7))
    print(kw)
    legend2 = ax.legend(*scatter.legend_elements(**kw),
                        bbox_to_anchor=(legend_x, legend_y), loc="lower right", handletextpad=None,handleheight=None,fontsize=20*labelsize,title='Num. Data Points')
    for i in range(len(legend2.get_texts())):
        txt = legend2.get_texts()[i].get_text().replace('$\mathdefault{','').replace('}$','')
        txt = '{:.0f}'.format(round(float(txt)/number_of_splits,0))
        legend2.get_texts()[i].set_text(txt)
    # Show the graph
    #plt.show()
    
    return fig,out_t

def cleveland_plot_v2(df,y ,comparison = None,
                      order_by = None,color1 = '#d9d9d9',color2 = '#d57883',
                      dot_colors = None,
                      dot_sizes = None,
                      annot_location_x = 2,annot_location_y = - 0.02,legend_location_x = 1,legend_location_y = 1.01,
                      title = 'A cleveland plot',title_location = 'left',xlim = None,annotation_type = 'normal',
                      annotation_decimals = 0,legend_min_size = 10,legend_max_size = 10,legend_title_size = 10,
                      legend_font_size = 10,show_dot_size_legend = True):

    # sort the plot values
    if order_by is not None:
        df = df.set_index(y).sort_values(order_by)
    else:
        df = df.set_index(y)

    # get the minimum for each row between the comparison groups
    df['min'] = df[comparison].min(axis=1)
    df['max'] = df[comparison].max(axis=1)

    # get the figure
    plt.figure(figsize=(12, 6))
    y_range = np.arange(1, len(df.index) + 1)

    # if there're only 2 groups then we can color the line connecting them depending on which one is larger
    if len(comparison) == 2:
        colors = np.where(df[comparison[0]] > df[comparison[1]], color1, color2)
    else:
        colors = color1

    # connect the dots with a line
    plt.hlines(y=y_range, xmin=df['min'], xmax=df['max'],
               color=colors, lw=10)

    # draw the dots
    for grp, c, s in zip(comparison, dot_colors, dot_sizes):
        plt.scatter(df[grp], y_range, color=c, s=s, label=grp, zorder=3)

    # do the annotations
    if annotation_type == 'percentage':
        for (_, row), y in zip(df.iterrows(), y_range):
            val = round(row['change'], annotation_decimals) * 100
            if annotation_decimals == 0:
                val = int(val)
            plt.annotate(str(val) + '%', (row["max"] + annot_location_x, y + annot_location_y))
    elif annotation_type == 'normal':
        for (_, row), y in zip(df.iterrows(), y_range):
            val = round(row['change'], annotation_decimals)
            if annotation_decimals == 0:
                val = int(val)
            plt.annotate(str(val), (row["max"] + annot_location_x, y + annot_location_y))

    # get the data points the smallest/biggest dot contains
    min_size = np.min(np.array(dot_sizes).squeeze())
    max_size = np.max(np.array(dot_sizes).squeeze())

    # draw the dot size legend if we want it
    if show_dot_size_legend:
        # create a legend for the dot sizes
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=str(int(min_size)),
                                  markerfacecolor='g', markersize=legend_min_size),
                           Line2D([0], [0], marker='o', color='w', label=str(int(max_size)),
                                  markerfacecolor='g', markersize=legend_max_size)]

        legend1 = plt.legend(handles=legend_elements, loc='lower right', title="Size", fontsize=legend_font_size,
                             bbox_to_anchor=(1, 0))

    # draw the main legend
    plt.legend(ncol=2, bbox_to_anchor=(legend_location_x, legend_location_y), loc="lower right",
               frameon=False)

    # attach the dot size legend if we want it
    if show_dot_size_legend:
        plt.gca().add_artist(legend1)
        plt.setp(legend1.get_title(), fontsize=legend_title_size)

    # label the y ticks
    plt.yticks(y_range, df.index)

    # add a title to the plot
    plt.title(
        title,
        loc=title_location)

    # set the xlim if it was supplied
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])

    # get the figure
    f = plt.gcf()

    return f

def unit_test_1():
    print('Unit test 1...')

    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)

    fig, _ = cleveland_plot(df=df, group_by='Pclass', grp_col='Sex', val_col='Fare', groups=['male', 'female'],
                            sortbycol=None, min_group_size=20, height=10, width=10, number_of_splits=5,
                            legend_x=1.05, legend_y=0, legend_x2=1.05, legend_y2=1, labelsize=1,
                            xlim=None, ylim=None)
    plt.show()

def unit_test_2():
    print('Unit test 2...')

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io
    sns.set(style="whitegrid")  # set style

    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)
    df_orig = df.copy()
    df_counts = df_orig[['Sex', 'Pclass', 'Fare']].groupby(['Sex', 'Pclass']).count().reset_index()
    df_counts = df_counts.rename(columns={'Fare': 'Size'})

    t = df[['Sex', 'Pclass', 'Fare']].copy()
    t_male = t[t['Sex']=='male'][['Pclass', 'Fare']].copy()
    t_male = t_male.rename(columns={'Fare':'Fare_male'})
    t_male = t_male.groupby(by='Pclass').mean().reset_index()
    t_male = pd.merge(t_male,df_counts[df_counts['Sex']=='male'][['Pclass','Size']],how='inner',on='Pclass',suffixes=('','_male'))
    t_male = t_male.rename(columns={'Size':'Size_male'})
    t_female = t[t['Sex'] == 'female'][['Pclass', 'Fare']].copy()
    t_female = t_female.rename(columns={'Fare': 'Fare_female'})
    t_female = t_female.groupby(by='Pclass').mean().reset_index()
    t_female.loc[t_female['Pclass'] == 3,'Fare_female'] = t_female['Fare_female'] - 10
    t_female = pd.merge(t_female, df_counts[df_counts['Sex'] == 'female'][['Pclass', 'Size']], how='inner', on='Pclass',
                      suffixes=('', '_female'))
    t_female = t_female.rename(columns={'Size': 'Size_female'})

    t_other = t_female.drop(columns=['Size_female'])
    t_other['Fare_female'] = t_other['Fare_female'] - 5
    t_other = t_other.rename(columns={'Fare_female':'Fare_other'})
    t_other['Size_other'] = 200

    df = pd.merge(t_male,t_female,how='inner',on='Pclass')
    df = pd.merge(df, t_other, how='inner', on='Pclass')
    df['change'] = df['Fare_female'] - df['Fare_male']

    # the y axis
    y = 'Pclass'

    # the groups to compare
    comparison = ['Fare_male','Fare_female','Fare_other']

    # the column to order by
    order_by = 'Fare_male'

    # line color 1
    color1 = '#d9d9d9'

    # line color 2
    color2 = '#d57883'

    # dot colors
    dot_colors = ['#0096d7','#003953','#003999']

    # dot sizes can be list of lists
    #dot_sizes = [200,200,200]
    dot_sizes = [df['Size_male'], df['Size_female'], df['Size_other']]

    # annotation location
    annot_location_x = 2
    annot_location_y = - 0.02

    # legend location
    legend_location_x = 1
    legend_location_y = 1.01

    # title
    title = 'A cleveland plot'
    title_location = 'left'

    # xlim
    xlim = (0,150)

    # show annotation
    annotation_type = 'normal'

    # annotation decimal places to show
    annotation_decimals = 0

    # legend dot sizes
    legend_min_size = 10
    legend_max_size = 10

    # legend sizes
    legend_title_size = 10
    legend_font_size = 10

    # show legend for dot sizes
    show_dot_size_legend = True

    # sort the plot values
    if order_by is not None:
        df = df.set_index(y).sort_values(order_by)
    else:
        df = df.set_index(y)

    # get the minimum for each row between the comparison groups
    df['min'] = df[comparison].min(axis=1)
    df['max'] = df[comparison].max(axis=1)

    # get the figure
    plt.figure(figsize=(12, 6))
    y_range = np.arange(1, len(df.index) + 1)

    # if there're only 2 groups then we can color the line connecting them depending on which one is larger
    if len(comparison) == 2:
        colors = np.where(df[comparison[0]] > df[comparison[1]], color1, color2)
    else:
        colors = color1

    # connect the dots with a line
    plt.hlines(y=y_range, xmin=df['min'], xmax=df['max'],
               color=colors, lw=10)

    # draw the dots
    for grp,c,s in zip(comparison,dot_colors,dot_sizes):
        plt.scatter(df[grp], y_range, color=c, s=s, label=grp, zorder=3)

    # do the annotations
    if annotation_type == 'percentage':
        for (_, row), y in zip(df.iterrows(), y_range):
            val = round(row['change'], annotation_decimals) * 100
            if annotation_decimals == 0:
                val = int(val)
            plt.annotate(str(val)+'%', (row["max"] + annot_location_x, y + annot_location_y))
    elif annotation_type == 'normal':
        for (_, row), y in zip(df.iterrows(), y_range):
            val = round(row['change'],annotation_decimals)
            if annotation_decimals == 0:
                val = int(val)
            plt.annotate(str(val), (row["max"] + annot_location_x, y + annot_location_y))

    # get the data points the smallest/biggest dot contains
    min_size = np.min(np.array(dot_sizes).squeeze())
    max_size = np.max(np.array(dot_sizes).squeeze())

    # draw the dot size legend if we want it
    if show_dot_size_legend:
        # create a legend for the dot sizes
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=str(int(min_size)),
                                  markerfacecolor='g', markersize=legend_min_size),
                           Line2D([0], [0], marker='o', color='w', label=str(int(max_size)),
                                  markerfacecolor='g', markersize=legend_max_size)]


        legend1 = plt.legend(handles=legend_elements, loc='lower right',title="Size",fontsize=legend_font_size,
                             bbox_to_anchor=(1, 0))

    # draw the main legend
    plt.legend(ncol=2, bbox_to_anchor=(legend_location_x, legend_location_y), loc="lower right",
                         frameon=False)

    # attach the dot size legend if we want it
    if show_dot_size_legend:
        plt.gca().add_artist(legend1)
        plt.setp(legend1.get_title(), fontsize=legend_title_size)

    # label the y ticks
    plt.yticks(y_range, df.index)

    # add a title to the plot
    plt.title(
        title,
        loc=title_location)

    # set the xlim if it was supplied
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])

    # get the figure
    f = plt.gcf()
    #f.subplots_adjust(right=1000)
    plt.tight_layout()
    plt.show()

def unit_test_3():
    print('Unit test 2...')

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io
    sns.set(style="whitegrid")  # set style

    current_dir = '/'.join(sys.path[0].split('/')[:-1])  # sys.path[0]
    data_dir = os.path.join(current_dir, 'Data', 'titanic')
    titanic_csv = os.path.join(data_dir, 'titanic.csv')
    df = pd.read_csv(titanic_csv)
    df_orig = df.copy()
    df_counts = df_orig[['Sex', 'Pclass', 'Fare']].groupby(['Sex', 'Pclass']).count().reset_index()
    df_counts = df_counts.rename(columns={'Fare': 'Size'})

    t = df[['Sex', 'Pclass', 'Fare']].copy()
    t_male = t[t['Sex'] == 'male'][['Pclass', 'Fare']].copy()
    t_male = t_male.rename(columns={'Fare': 'Fare_male'})
    t_male = t_male.groupby(by='Pclass').mean().reset_index()
    t_male = pd.merge(t_male, df_counts[df_counts['Sex'] == 'male'][['Pclass', 'Size']], how='inner', on='Pclass',
                      suffixes=('', '_male'))
    t_male = t_male.rename(columns={'Size': 'Size_male'})
    t_female = t[t['Sex'] == 'female'][['Pclass', 'Fare']].copy()
    t_female = t_female.rename(columns={'Fare': 'Fare_female'})
    t_female = t_female.groupby(by='Pclass').mean().reset_index()
    t_female.loc[t_female['Pclass'] == 3, 'Fare_female'] = t_female['Fare_female'] - 10
    t_female = pd.merge(t_female, df_counts[df_counts['Sex'] == 'female'][['Pclass', 'Size']], how='inner', on='Pclass',
                        suffixes=('', '_female'))
    t_female = t_female.rename(columns={'Size': 'Size_female'})

    t_other = t_female.drop(columns=['Size_female'])
    t_other['Fare_female'] = t_other['Fare_female'] - 5
    t_other = t_other.rename(columns={'Fare_female': 'Fare_other'})
    t_other['Size_other'] = 200

    df = pd.merge(t_male, t_female, how='inner', on='Pclass')
    df = pd.merge(df, t_other, how='inner', on='Pclass')
    df['change'] = df['Fare_female'] - df['Fare_male']

    # the y axis
    y = 'Pclass'

    # the groups to compare
    comparison = ['Fare_male', 'Fare_female', 'Fare_other']

    # the column to order by
    order_by = 'Fare_male'

    # line color 1
    color1 = '#d9d9d9'

    # line color 2
    color2 = '#d57883'

    # dot colors
    dot_colors = ['#0096d7', '#003953', '#003999']

    # dot sizes can be list of lists
    # dot_sizes = [200,200,200]
    dot_sizes = [df['Size_male'], df['Size_female'], df['Size_other']]

    # annotation location
    annot_location_x = 2
    annot_location_y = - 0.02

    # legend location
    legend_location_x = 1
    legend_location_y = 1.01

    # title
    title = 'A cleveland plot'
    title_location = 'left'

    # xlim
    xlim = (0, 150)

    # show annotation
    annotation_type = 'normal'

    # annotation decimal places to show
    annotation_decimals = 0

    # legend dot sizes
    legend_min_size = 10
    legend_max_size = 10

    # legend sizes
    legend_title_size = 10
    legend_font_size = 10

    # show legend for dot sizes
    show_dot_size_legend = True

    f = cleveland_plot_v2(df=df, y=y, comparison=comparison,
                      order_by=order_by, color1='#d9d9d9', color2='#d57883',
                      dot_colors=dot_colors,
                      dot_sizes=dot_sizes,
                      annot_location_x=2, annot_location_y=- 0.02, legend_location_x=1, legend_location_y=1.01,
                      title='A cleveland plot', title_location='left', xlim=xlim, annotation_type='normal',
                      annotation_decimals=0, legend_min_size=10, legend_max_size=10, legend_title_size=10,
                      legend_font_size=10, show_dot_size_legend=True)

    plt.show()

if __name__ == '__main__':
    unit_test_1()
    unit_test_2()
    unit_test_3()
