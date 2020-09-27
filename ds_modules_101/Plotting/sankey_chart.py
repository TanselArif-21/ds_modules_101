# Libraries
import numpy as np
import plotly.graph_objects as go

def getSankey(title,label,source,target,value,color_of_group_outline = None,color_of_groups = None,color_of_lines = None,padding_between_groups=20,thickness_of_groups=20,
			 font_size=10,font_color=None,fig_height=None,fig_width=None):
	'''
	This function creates a Sankey Chart. 
	:param title: String. The title of the chart
	:param label: List of strings. Each element corresponds to a group name
	:param source: List of indices corresponding to the label list. This corresponds to the source of a movement
	:param target: List of indices corresponding to the label list. This corresponds to the target of a movement. Same size as source
	:param values: List of values. Each value corresponds to the magnitude of a move from the same index in source to the same index in target. Same size as target and source
	:param color_of_group_outline: A single color or a list of colors. The color for the outlines of the groups
	:param color_of_groups: A single color or a list of colors. The color for the groups
	:param color_of_lines: A single color or a list of colors. The color for the lines connecting the groups
	:param padding_between_groups: An integer. The distance between each group in the display
	:param thickness_of_groups: An integer. The thickness of the groups
	:param font_size: An integer. The size of the title and names etc...
	:param font_colo: An integer. The color of the title
	:param fig_height: An integer. The height of the figure object
	:param fig_width: An integer. The width of the figure object
	Example:
	import Graphs
	# Dataset 1: for Sankey
	label = ['A1','A2','A3','A4','A5','A1','A2','A3','A4','A5']
	source = [0,0,1,1,2,3,4,4] # These are the indices of the source group
	target = [6,8,7,6,8,9,5,6] # These are the indices of the target group
	value = list(np.random.randint(0,1000,5)) # These are the number of people going for the source group to the target group
	
	f = Graphs.getSankey('TempTitle',label=label,source=source,target=target,value=value,color_of_group_outline='grey',color_of_groups=['grey','black'],color_of_lines='light grey')
	f.show()	
	'''
	fig = go.Figure(data=[go.Sankey(
		node = dict(
			pad = padding_between_groups,
			thickness = thickness_of_groups,
			line = dict(color = color_of_group_outline, width = 0.5),
			label = label,
			color = color_of_groups
	),
	link = dict(
	  source = source,
	  target = target,
	  value = value,
		color=color_of_lines
	))])

	fig.update_layout(title_text=title, font=dict(size = font_size, color = font_color),height=fig_height, width=fig_width)
	return fig
	
	
def transform_data_for_sankey(df,from_column,to_column,source_and_target_separate = True):
	'''
	This function gets a dataframe in the format required for the sankey function. It is expected that there are 2 columns in the dataframe; a from column and a to column.
	
	:param df: The dataframe
	:param from_column: The column name to be the source column
	:param to_column: The column name to be the target column
	:param source_and_target_separate: If this is False, then we are visualising this as the same groupings. If it is True, then the from and to columns will be visualised as 2 separate groups
	
	Example:
	import Graphs
	import numpy as np
	import pandas as pd
	import Directory
	import os
	# Load dataset
	df = pd.read_csv(os.path.join(Directory.dataPath,'titanic','train.csv'))

	# Set the from and to columns
	from_column = 'SibSp'
	to_column = 'Survived'
	
	label,source,target,values = Graphs.transform_data_for_sankey(df,from_column,to_column,source_and_target_separate = True)
	
	f = Graphs.getSankey('Sankey Chart from {} to {}'.format(from_column,to_column),label=labels,source=source,target=target,value=value,color_of_group_outline='grey',color_of_groups=['grey','black'],color_of_lines='light grey',font_color='Black',font_size=10)
	f.show()
	'''

	# Get the reduce dataset
	df2 = df[[from_column,to_column]].copy()
	
	# This will be the count column in the group by
	df2['values'] = 1
	
	# Group by all combinations of these columns and get the counts
	df3 = df2.groupby(by=[from_column,to_column]).count().reset_index()
	
	# This is the original source column
	source = list(df3[from_column])
	
	# This is the unique entries in the source list
	source_unique = list(np.unique(np.array(source)))
	
	# This is the original target list
	target = list(df3[to_column])
	
	# This is the unique entries in the target list
	target_unique = list(np.unique(np.array(target)))
	
	# These are the values list
	values = list(df3['values'])
	
	# This is the shift value to shift from source to target if required (i.e. if we want the source to be seen as separate to the target)
	shift = len(source_unique)
	
	if not source_and_target_separate:
		shift = 0
	
	# This is the full labels list
	labels =  source_unique + target_unique
	
	# This is the encoded source list (i.e. referring to the indices of the labels list)
	source_encoded = list(map(lambda x: source_unique.index(x),source))
	
	# This is the encoded target list (i.e. referring to the indices of the labels list)
	target_encoded = list(map(lambda x: target_unique.index(x) + shift,target))   
	
	return labels, source_encoded, target_encoded, values