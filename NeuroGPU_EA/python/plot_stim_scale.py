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

# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 14})

# def set_custom_params_plt():

#     colors = cycler('color',
#                     ['#EE6666', '#3388BB', '#9988DD',
#                     '#EECC55', '#88BB44', '#FFBBBB'])
#     plt.rc('axes', facecolor='white', edgecolor='black',
#         axisbelow=True, grid=True)
#     plt.rc('grid', color='lightgrey', linestyle='solid')
#     plt.rc('patch', edgecolor='grey')
#     plt.rc('lines', linewidth=2, color='black')
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
#     plt.rcParams.update({'font.size': 30})

    
# set_custom_params_plt()
df =  pd.read_csv("6GPU_outputs/fixed_compute_stim.txt_scale.csv")
df = df.sort_values('Stimuli')
vals = df['Runtime'].values
labels  = df['Stimuli'].values
err = df['Runtime Stddev'].values
expected = (vals[0] * np.log(labels / labels[0]) / 2) + vals[0]

# expected2 = ((vals[0] * labels) / ( 4))
expected2 = (vals[0] * np.log(labels / labels[0]) / 4) + vals[0]

# show that log benchmark is a straight line when x axis is log scaled
# labels = np.array([1,2,4,8])
# expected = (vals[0] * np.log(labels)) + vals[0]
# expected2 = (vals[0] * labels) #+ vals[0]
# expected3 = (vals[0] * labels**2) #+ vals[0]
# plt.plot(labels, expected, label='log', color='firebrick')
# plt.plot(labels, expected2, label='linear', color='blue')
# # plt.plot(labels, expected3, label='exponential', color='firebrick')
# plt.legend()
# plt.savefig('chec_log_scale.png')
# exit()

plt.plot(labels, expected, label='O(log(n)/2)', color='firebrick')
plt.text(labels[-3], expected[-3]+1, 'O(log(n)/2)', color='firebrick', fontsize=18, rotation=20)
plt.plot(labels, expected2, label='O(log(n)/4)', color='darkred')
plt.text(labels[-2], expected2[-2]-3, 'O(log(n)/4)', color='darkred',fontsize=18)

plt.plot(labels, vals, label="Observed", color='black')
plt.text(labels[-2], vals[-2]+1, 'Observed', color='black',fontsize=18, rotation=5)

plt.xticks(labels)
plt.fill_between(labels,vals-err, vals+err, alpha=.3)

# plt.ylabel("Runtime (s)")
plt.xlabel("Number of Stimuli", fontsize=14)
plt.ylabel("Run Time (s)", fontsize=14)
# plt.legend(fontsize=20)
plt.ylim(bottom=0)
plt.savefig("stimscale.png", bbox_inches='tight')

