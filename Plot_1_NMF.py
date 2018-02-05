#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:47:45 2018

@author: nupur
"""

import pandas as pd
df1 = pd.read_csv('df1.csv')


from bokeh.plotting import figure
from bokeh.io import output_file, show
from bokeh.models.tickers import FixedTicker
from bokeh.plotting import ColumnDataSource
from bokeh.models import HoverTool

source = ColumnDataSource(df1)

p = figure(x_axis_label='Topics (Ref. README)', y_axis_label='Weights', tools='box_select')

x = range(1,36)
y = df1.iloc[8,1:]

p.circle(x, y,size=10,
         fill_color='grey', source=source,alpha=0.3, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white')
p.line(x, y,source=source,alpha=0.3, line_color=None)
p.xaxis.ticker = FixedTicker(ticks=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35])

hover = HoverTool(tooltips=[("(Modi)", "($x, $y)")], mode='vline')
p.add_tools(hover)
output_file('Plot1_NMF_recommendations.html')
show(p)
