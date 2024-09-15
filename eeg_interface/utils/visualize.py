import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def visualize_time_series(df):
    graph = sns.lineplot(data=df).get_figure()
    graph.savefig('timeSeries.png')