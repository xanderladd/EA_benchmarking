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
    plt.rc('axes', facecolor='white', edgecolor='black',
        axisbelow=True, grid=True)
    plt.rc('grid', color='lightgrey', linestyle='solid')
    plt.rc('patch', edgecolor='grey')
    plt.rc('lines', linewidth=2, color='black')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams.update({'font.size': 18})



# set_custom_params_plt()
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
 
plt.rcParams.update({'font.size': 14})

df =  pd.read_csv("6GPU_outputs/fixed_compute_sf.txt_scale.csv")
df = df.sort_values('Score Functions')
vals = df['Runtime'].values
labels  = df['Score Functions'].values
err = df['Runtime Stddev'].values
# import pdb; pdb.set_trace()
expected = (vals[0] * labels / labels[0])  / 3

# expected2 = ((vals[0] * labels) / ( 4))
expected2 = (vals[0] * np.log(labels / labels[0]) / 2) + vals[0]

plt.plot(labels, expected, label='O(n/3)', color='firebrick')
plt.text(labels[-2]-10, expected[-2]+3, 'O(n/3)', color='firebrick', fontsize=17, rotation=35)

plt.plot(labels, expected2, label='O(log(n)/2)', color='darkred')
plt.text(labels[-2]-10, expected2[-2]-14, 'O(log(n)/4)', color='darkred',fontsize=17, rotation=5)

plt.plot(labels, vals, label="Observed", color='black')
plt.text(labels[-2]+10, vals[-2]-2, 'Observed', color='black',fontsize=17, rotation=40)

plt.fill_between(labels,vals-err, vals+err, alpha=.3)
ax = plt.gca()
ax.axvline(80,color='darkblue', linestyle='dashed', label='80 Score Functions', linewidth=2)
# plt.text(70,65, '80', color='darkblue',fontsize=20, rotation=90, zorder=10)
# plt.text(72,78, 'Score Functions', color='darkblue',fontsize=11, rotation=90, zorder=10)
plt.text(5,90, '80 Score Functions', color='darkblue',fontsize=17, zorder=10)

# plt.ylabel("Runtime (s)")
plt.xlabel("# of Score Functions", fontsize=14)
plt.ylabel("Run Time (s)", fontsize=14)
plt.xticks(labels)

# plt.legend(fontsize=20)
plt.ylim(bottom=0)
plt.savefig("sfscale.png", bbox_inches='tight')

