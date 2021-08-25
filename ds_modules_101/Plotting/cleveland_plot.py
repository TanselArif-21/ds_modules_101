import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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