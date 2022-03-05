import matplotlib.pyplot as plt
import numpy as np
import h5py 
import os
from matplotlib import cycler
import matplotlib.ticker as mticker
import scalebary as sb
from matplotlib.transforms import Bbox

# def set_custom_params_plt():

#     colors = cycler('color',
#                     ['#EE6666', '#3388BB', '#9988DD',
#                     '#EECC55', '#88BB44', '#FFBBBB'])
#     # plt.rc('axes', facecolor='white', edgecolor='black',
#     #     axisbelow=True, grid=True)
#     # plt.rc('grid', color='lightgrey', linestyle='solid')
#     # plt.rc('patch', edgecolor='grey')
#     # plt.rc('lines', linewidth=2, color='black')
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
#     plt.rcParams.update({'font.size': 36})
plt.rcParams.update({'font.size': 35})
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# plt.rcParams.update({'font.size': 36})



    
# set_custom_params_plt()
# import pdb; pdb.set_trace()
target_volts_hdf5 = h5py.File('/global/cscratch1/sd/zladd/benchmarking/stims/allen_data_target_volts_488683423.hdf5','r')


stims_hdf5 = h5py.File('/global/cscratch1/sd/zladd/benchmarking/stims/allen_data_stims_488683423.hdf5','r')

print(target_volts_hdf5.keys())
print(stims_hdf5.keys(), "stim keys")

os.makedirs('figures/stims', exist_ok=True)
os.makedirs('figures/targ_v', exist_ok=True)

stim_path = 'figures/stims'
targ_v_path = 'figures/targ_v'

def f(axis):
    l = axis.get_majorticklocs()
    return len(l)>1 and (l[1] - l[0])


###### belwo generates plots of all stims and all volts
####### STIMS ###############
# for key in stims_hdf5.keys():
#     if "dt" in key:
#         continue
#     curr_stim = stims_hdf5[key][:]
#     if len(curr_stim) < 20000 or len(curr_stim) > 34000:
#         print('skipping : ', key)
#         print(len(curr_stim))
#         continue
#     elif len(curr_stim) != 30000:
#         curr_stim = curr_stim[:30000]
    

#     fig = plt.figure()
#     ax = fig.gca()
#     dt = stims_hdf5[key+'_dt'][:][0]
        
#     stim_end = ( len(curr_stim) * dt)  #/ 1000 --- use divisor for miliseconds
#     plt.plot(np.arange(len(curr_stim)) * dt , curr_stim, color='grey', linewidth=2)
#     tick_interval = stim_end // 10
#     ax.set_xticks(np.arange(0,stim_end + tick_interval, tick_interval))
# #     plt.ylabel("Stimulus mV")
# #     plt.xlabel("Timestep (ms)")
#     # if key == "53_1":
#     #     sb.add_scalebar(ax, sizey=.1, sizex=495, matchx=False, matchy=False, 
#     #                     labelx="mS", labely="nA", loc=1, pad=0, y_sep=-50,
#     #                     barwidth=2.5, bbox_to_anchor=Bbox.from_bounds(0,0, .2, .6), 
#     #                     bbox_transform=ax.figure.transFigure)
#     # else:
#     sb.add_scalebar(ax, labelx="mS", labely="nA", loc=1, pad=0, y_sep=-50, barwidth=2.5, bbox_to_anchor=Bbox.from_bounds(0,0, .2, .6), bbox_transform=ax.figure.transFigure)
#     # plt.title("Stim " + key)
#     fig.savefig(os.path.join(stim_path, key + ".png"), bbox_inches='tight')
#     plt.close(fig)
#     if key == '51_3':
#         fig = plt.figure(figsize=(5,3))
#         plt.plot(curr_stim[8000:10000])
#     #         locs, labels = plt.yticks() 
#     #         plt.yticks(locs, labels=np.arange(1000))
#         plt.ylabel("Stimulus mV")
#         plt.xlabel("Timestep (ms)")
#         fig.savefig(os.path.join('figures', key + "_stim.png"), bbox_inches='tight')
#         plt.close(fig)
    
    
    
######### TARGET V #######################
for key in target_volts_hdf5.keys():
    if "dt" in key:
        continue
    curr_volt = target_volts_hdf5[key][:]
    curr_stim = stims_hdf5[key][:]
    if len(curr_volt) < 20000 or len(curr_stim) > 34000:
        print('skipping : ', key)
        print(len(curr_volt))
        continue
    elif len(curr_volt) != 30000:
        curr_volt = curr_volt[:30000]
        curr_stim = curr_stim[:30000]
 
    fig = plt.figure()
    ax = fig.gca()
    dt = target_volts_hdf5[key+'_dt'][:][0]
    stim_end = ( len(curr_stim) * dt)  #/ 1000 --- use divisor for miliseconds

    tick_interval = stim_end // 10
    plt.plot(np.arange(len(curr_stim)) * dt, curr_volt, color='black')
    ax.set_xticks(np.arange(0,stim_end + tick_interval, tick_interval))

#     plt.ylabel("Stimulus mV")
#     plt.xlabel("Timestep (ms)")
    sb.add_scalebar(ax, labelx="mS", labely="mV", loc=1, pad=0, y_sep=-50, barwidth=2.5, bbox_to_anchor=Bbox.from_bounds(0,.1, .2, .5), bbox_transform=ax.figure.transFigure)
    # plt.title("Target Voltage # " + key)
    # plt.title(" # " + key)
    fig.savefig(os.path.join(targ_v_path, key + ".png"), bbox_inches='tight')
    plt.close(fig)
    
    if key == '51_3':
        fig = plt.figure(figsize=(8,4))
        plt.plot(curr_volt[8000:10000], color='black')
#         locs, labels = plt.yticks() 
#         plt.yticks(locs, labels=np.arange(1000))
        plt.ylabel("Membrane Potential (mV)", fontsize=24)
        plt.xlabel("Timestep (ms)", fontsize=24)
        plt.xticks(fontsize=20) 
        plt.yticks(fontsize=20) 
        fig.savefig(os.path.join('figures', key + ".png"), bbox_inches='tight')
        plt.close(fig)
        
    
    
    
# fig = plt.figure()        
# colors = ['grey','black', 'red' , 'blue', 'green']
# interesting_keys = ['55_3', '13','44','6','27']
# for idx, key in enumerate(interesting_keys):
#     if "dt" in key:
#         continue
#     if key == '13':
#         linewidth = 2
#     elif key != '55_3':
#         linewidth = 4
#     else:
#         linewidth = .7
#     curr_stim = stims_hdf5[key][:]
    
#     plt.plot(curr_stim, label =f'Stimuli {idx}',linewidth=linewidth, color=colors[idx])
#     plt.ylabel("Stimulus mV")
#     plt.xlabel("Timestep (ms)")
# plt.legend(loc=(.9,.5), fontsize=20)
# fig.savefig(os.path.join('figures',  "multi_stim.png"), bbox_inches='tight')
# plt.close(fig)


# fig = plt.figure()        
# interesting_keys = ['55_3', '13','44','4','27']
# for idx, key in enumerate(interesting_keys):
#     if idx == 0  or idx == 4:
#         continue
#     if "dt" in key:
#         continue
#     if key == '13':
#         linewidth = 2
#     elif key != '55_3':
#         linewidth = 2
#     else:
#         linewidth = .7
#     curr_v = target_volts_hdf5[key][:]
#     zorder = 1
#     if idx == 1:
#         zorder = 10
#     plt.plot(curr_v, label =f'Target Voltage for \n Stimuli {idx}',linewidth=linewidth, color=colors[idx], zorder=zorder)
#     plt.ylabel("Membrane Potential (mV)")
#     plt.xlabel("Timestep (ms)")
# plt.legend(loc=(1,.8),  fontsize=18)
# fig.savefig(os.path.join('figures',  "multi_volt.png"), bbox_inches='tight')
# plt.close(fig)