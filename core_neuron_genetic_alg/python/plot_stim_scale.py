import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cycler
import argparse
import matplotlib.ticker as mticker
from matplotlib.ticker import AutoLocator


def title_and_save(fig,title, pdf):
    global fig_count
    if title:
        plt.title("Fig {}: ".format(fig_count) + title, fontsize=20)
    else:
        plt.title("Fig {}: ".format(fig_count), fontsize=20)
    pdf.savefig(fig, bbox_inches='tight')
    fig_count += 1
    plt.close(fig)


def set_custom_params_plt():

    colors = cycler('color',
                    ['#EE6666', '#3388BB', '#9988DD',
                    '#EECC55', '#88BB44', '#FFBBBB'])
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
        axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('patch', edgecolor='#E6E6E6')
    plt.rc('lines', linewidth=2)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


set_custom_params_plt()
df =  pd.read_csv("real_gpu_cori_outputs/fixed_compute_stim.txt_scale.csv")
df = df.sort_values('Stimuli')
import pdb; pdb.set_trace()
vals = df['Runtime']
labels  = df['Stimuli']
err = df['Runtime Stddev']

expected = (vals[0] * vals) / (labels[0] )

expected2 = (vals[0] * vals) / (labels[0] * 2)
plt.plot(labels, expected, label='O(n)', color='red')

plt.plot(labels, expected2, label='O(n/2)', color='purple')

plt.plot(labels, vals, label="actual", color='blue')
plt.errorbar(labels,vals, yerr=err)
plt.ylabel("Runtime (s)")
plt.xlabel("Number of Stimuli")
plt.legend()
plt.ylim(0,100)
plt.savefig("stimscale.png")

