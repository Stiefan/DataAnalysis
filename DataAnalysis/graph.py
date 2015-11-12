# -*- coding: utf-8 -*-

from collections import Counter

import csv
import numpy as np 
import matplotlib.pyplot as plt
import bokeh.plotting as bpl

from bokeh import mpl
from bokeh.plotting import figure, output_file, show
from bokeh.charts import Bar, output_file, show
from bokeh.models import HoverTool, BoxSelectTool, BoxZoomTool,ResetTool
from parse import parse

MY_FILE = "C:/Dev/repricing.csv"

def visualize_data():   
	data_file = parse(MY_FILE, "~")

	counter = Counter(item["Product"] for item in data_file)
   	
	data_list = [
	              counter["P-HYP"],
	              counter["RTVAST"],
	              counter["A-MKHYP"],
	              counter["P-HYPL"],
	            ]
	labels_tuple = tuple(["P-HYP", "RTVAST", "A-MKHYP", "P-HYPL"])

	plt.title("Test")
	plt.plot(data_list)
	plt.xticks(range(len(labels_tuple)), labels_tuple)
    
    #Save to bokeh
	plot = mpl.to_bokeh(name='test')
	bpl.show(plot)   
	output_file("test1.html")
	plt.clf()

"""def visualize_type():
	data_file = parse(MY_FILE,"~")

	counter = Counter(item["Product"] for item in data_file)
	print counter
	print counter.values()

	labels = tuple(counter.keys())
	print labels 

	hover = HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
            ("desc", "$desc"),
        ]
    )

	TOOLS = [BoxSelectTool(), hover, BoxZoomTool(),ResetTool()]

	#p = figure(plot_width=400, plot_height=400,
     #      title="Data Repricing Products", toolbar_location="below", tools = TOOLS)

	#p.circle([1, 2, 3, 4], counter.values(), size=10)
	p = Bar([1, 2, 3, 4],counter.values(),title='test')

	output_file("test1.html")
	show(p)

"""
def main():
	visualize_data()
	#visualize_type()

if __name__ == "__main__":
	main()