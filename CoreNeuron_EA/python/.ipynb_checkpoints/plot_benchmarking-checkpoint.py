import numpy as np
import matplotlib.pyplot as plt
import re
import os
import pstats
import sys
import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import argparse
import matplotlib
from matplotlib import cycler
import pickle
import matplotlib.ticker as mticker
import matplotlib
import glob


# font = {'family' : 'normal',
#         'size'   : 20}

# matplotlib.rc('font', **font)
# # plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"

default_sfs = str(20)
default_stims = str(1)
default_pop = str(500)
POP_SCALING_FACTOR=500
# with open("std_dev_backup.pkl",'rb') as f:
#     backup_stddev = pickle.load(f)


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

    # colors = cycler('color',
    #                 ['#EE6666', '#3388BB', '#9988DD',
    #                 '#EECC55', '#88BB44', '#FFBBBB'])
    # plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
    #     axisbelow=True, grid=True)
    # plt.rc('grid', color='w', linestyle='solid')
    # plt.rc('patch', edgecolor='#E6E6E6')
    # plt.rc('lines', linewidth=2)
    plt.rcParams['font.family'] = 'Arial'
    # plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams.update({'font.size': 18})


    
set_custom_params_plt()

def restore_default_mpl_params():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

def format_logname(node, pop, nCpu, stim, sf, path, how, title=None):
    if how == 'vanilla':
        stim = default_stims
        sf = default_sfs
        title = "Population"
    elif how == 'stims':
        pop = default_pop
        sf = default_sfs
        title = "Stims"
    elif how == "sfs":
        pop = default_pop
        stim = default_stims
        title = "Sfs"
    elif how == "full":
        assert title
        pass
    else:
        raise NotImplementedError
#         title=" PLOTTING METHOD NOT RECOGNIZED"
        
    if os.path.isfile("{}/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(path, node,nCpu, pop, stim, sf, node,nCpu,pop, stim, sf)):
        return "{}/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(path, node,nCpu, pop, stim, sf, node,nCpu,pop, stim, sf), title
    else:
        print("NO FILE ", "{}/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(path, node,nCpu, pop, stim, sf, node,nCpu,pop, stim, sf))
        return "{}/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(path, node,nCpu, pop, stim, sf, node,nCpu,pop, stim, sf), title

def format_gpu_util_name(node, pop, nCpu, stim, sf, path, how, title=None):
    if how == 'vanilla':
        stim = default_stims
        sf = default_sfs
        title = "Population"
    elif how == 'stims':
        pop = default_pop
        sf = default_sfs
        title = "Stims"
    elif how == "sfs":
        pop = default_pop
        stim = default_stims
        title = "Sfs"
    elif how == "full":
        assert title
        pass
    else:
        raise NotImplementedError
    gpu_logs = [os.path.join("{}/{}N_{}C_{}O_{}S_{}SF/".format(path,node,nCpu, pop, stim, sf), filename) \
                for filename in os.listdir("{}/{}N_{}C_{}O_{}S_{}SF/".format(path, node,nCpu, pop, stim, sf)) \
                if "gpu_utillization" in filename]
    return gpu_logs, len(gpu_logs)


def find_gpu_logs(basepath):
    files = os.listdir(basepath)
    gpu_logs = sorted([os.path.join(basepath, file) for file in files if "gpu_utillization" in file])
    return gpu_logs
    
    
def read_gpu_logs(fn):
    basepath = os.path.dirname(fn)
    gpu_logs = find_gpu_logs(basepath)
    fn = gpu_logs[-1]
    
    with open(fn, 'r') as gpu_f: 
        lines = gpu_f.readlines()
    gpu_df = pd.DataFrame([sub.split(",") for sub in lines])
    gpu_df.columns = gpu_df.iloc[0]
    gpu_df = gpu_df[1:]
    gpu_df = gpu_df.rename({' name': 'name', ' utilization.gpu [%]' : 'utilization'}, axis=1)
    # remove label rows

    gpu_df = gpu_df[gpu_df['name'] != ' name']
    gpu_df = gpu_df[gpu_df['timestamp'] != '\n']
    gpu_df  = gpu_df[gpu_df['utilization'] != ' utilization.memory [%]']
    # this will only work for the year
    gpu_df = gpu_df[gpu_df['timestamp'].str.count("2021") < 2]
    gpu_df['timestamp'] = pd.to_datetime(gpu_df['timestamp'], errors='coerce')#gpu_df['timestamp'].astype('datetime64[ns]')
    gpu_df = gpu_df.dropna(axis=0, subset=['timestamp']) 
    # WE GET ONE MEASUREMENT PER SECOND
    total_elapsed = (max(gpu_df.timestamp) - min(gpu_df.timestamp)).seconds 
    gpu_df['utilization'] = gpu_df['utilization'].str.replace(" \%","").astype(int)
    percent_utilization = (np.sum(gpu_df['utilization'] ) / (6 * total_elapsed))
    #gpu_df[['Date','Time']] = gpu_df.timestamp.str.split(expand=True)
    return percent_utilization, gpu_df


def processLog(f):
    with open(f, "r") as file:
        startEndPairs = []
        runtimes = []
        sfs = []
        neuroGPUStartTimes = []
        neuroGPUEndTimes = []
        procToSf = {}
        procStartDict = {}
        procEndDict = {}
        compStartDict = {}
        io_times = []
        evalTimes = []
        neuroGPUTimes = []
        gens = []
        file_lines =file.readlines()
        readingEnds = False
        readingStarts = False
        firstGen = True
        for line in file_lines:
            if "Date:" in line:
                print(line)
            if "absolute start" in line:
                firstGen = True
                numbers = re.findall(r'\d+', line)
                abs_start = [ '.'.join(x) for x in zip(numbers[0::2], numbers[1::2]) ][0]
            if "nCpus" in line:
                nCpus = int(re.match('.*?([0-9]+)$', line).group(1))
                #assert nCpus  == cpu, "expected {} but got {} cpus in log".format(cpu,nCpus)
            if "nGPUS" in line:
                curr_idx = line.find('nGPUS')
                curr_line = line[curr_idx:curr_idx+8]
                nGpus = int(re.match('.*?([0-9]+)$', curr_line).group(1))
                
                
            if "took:" in line:
                if firstGen:
                    firstGen = False
                    continue
                runtime = float(re.findall(r"[-+]?\d*\.\d+|\d+",line)[1])
                #print(runtime)
#                 if runtime > 100:
#                     runtime = 22
                runtimes.append(runtime)
            if "launched PIDS" in line:
                start = re.findall(r'\d+', line)[0] # second half is in miliseconds, don't need that precision
                
            if "finished PIDS" in line:
                end = re.findall(r'\d+', line)[0] 
                startEndPairs.append((start,end))
            if "process"  in line and "started" in line:
                stSplit = line.split(" ")
                sf = [stSplit[i] for i in range(2,len(stSplit)-2) if stSplit[i-1] == "is" and  stSplit[i+1] == "and"][0]
                sfs.append(sf)
                line = re.sub(r'(?<=is)(.*)(?=and)', "", line)
                numbers = re.findall(r'\d+', line)

                procToSf[numbers[0]] = sf
                if numbers[0] in procStartDict.keys():
                    procStartDict[numbers[0]].append(numbers[1])
                else:
                    procStartDict[numbers[0]] = [numbers[1]]
            if "returning" in line:
                numbers = re.findall(r'\d+', line)
                if numbers[0] in procEndDict.keys():
                    procEndDict[numbers[0]].append(numbers[1])
                else:
                    procEndDict[numbers[0]] = [numbers[1]]
            if "computing" in line:
                numbers = re.findall(r'\d+', line)
                if numbers[0] in compStartDict.keys():
                    compStartDict[numbers[0]].append(numbers[1])
                else:
                    compStartDict[numbers[0]] = [numbers[1]]
            if "evaluation:" in line and not firstGen:
                numbers = re.findall(r'\d+', line)
                numbers = [ '.'.join(x) for x in zip(numbers[0::2], numbers[1::2]) ]
                if "evalTimes" in locals():
                    evalTimes = np.append(evalTimes,  np.array(list(numbers), dtype=np.float32))
                else:
                    evalTimes = np.array(list(numbers), dtype=np.float32)
                avgEval = np.mean(evalTimes)
            if "neuroGPU" in line and "starts" not in line and "ends" not in line and not firstGen:
                numbers = re.findall(r'\d+', line)
                numbers = [ '.'.join(x) for x in zip(numbers[0::2], numbers[1::2]) ]
                
                if "neuroGPUTimes" in locals():
                    neuroGPUTimes = np.append(neuroGPUTimes,  np.array(list(numbers), dtype=np.float32))
                else:
                    neuroGPUTimes = np.array(list(numbers),dtype=np.float32)
                avgNGPU = np.mean(neuroGPUTimes)
            if ("neuroGPU" in line and "starts" in line and "ends" not in line) or readingEnds and not firstGen:
                readingEnds = True
                numbers = re.findall(r'\d+', line)
                numbers = [ '.'.join([x1,x2]) + "e+" + str(x3) for x1,x2,x3 in zip(numbers[0::3], numbers[1::3], numbers[2::3]) ]
                neuroGPUStartTimes += numbers
                if "]" in line:
                    readingEnds = False
            if ("neuroGPU" in line and "starts" not in line and "ends" in line) or readingStarts and not firstGen:
                readingStarts = True
                numbers = re.findall(r'\d+', line)
                numbers = [ '.'.join([x1,x2]) + "e+" + str(x3) for x1,x2,x3 in zip(numbers[0::3], numbers[1::3], numbers[2::3]) ]
                neuroGPUEndTimes += numbers
                if "]" in line:
                    readingStarts = False
            if "IO:" in line:
                numbers = re.findall(r'\d+', line)
                numbers = [ '.'.join([x1,x2]) + "e+" + str(x3) for x1,x2,x3 in zip(numbers[0::3], numbers[1::3], numbers[2::3]) ]
                io_times.append(numbers)
            if 'gen size' in line:
                numbers = re.findall(r'\d+', line)
                gens.append(float(numbers[0]))
#             if "gen1 took" in line:
#                 break
    try:
        res = {"procStartDict": procStartDict,"procEndDict": procEndDict,\
               "startEndPairs": startEndPairs,"runtimes": runtimes,\
               "compStartDict": compStartDict,"sfs": sfs,\
               "evalTimes": evalTimes,"neuroGPUTimes": neuroGPUTimes,\
              "procToSf": procToSf, "absStart": abs_start, \
               "neuroGPUStartTimes": neuroGPUStartTimes, \
               "neuroGPUEndTimes": neuroGPUEndTimes, "ioTimes": io_times,
              "nGpus": nGpus, 'gen_sizes':gens}
    except UnboundLocalError as e:
        print(e)
        print("MISREAD LOG : ", f, "  but I am in PERMISSIVE mode so it's ok")
#         raise e
        return {"procStartDict": {},"procEndDict": {},\
               "startEndPairs": [],"runtimes": [],\
               "compStartDict": {},"sfs": [],\
               "evalTimes": [],"neuroGPUTimes": [],\
              "procToSf": {}, "absStart": 0, \
               "neuroGPUStartTimes": [], \
               "neuroGPUEndTimes": [], "ioTimes": []}
    return res

    

def makeCustomProfile(node, nCpu, pop, stim, sf, vers, path, show=True):
    f, _ = format_logname(node, pop, nCpu, stim, sf, path, how='full', title="None")
    #f  = "runTimeLogs/runTime.log"
    logRes = processLog(f)
    print("making profile for {}".format(f))
    absStart = float(logRes['absStart'])
    start_data = np.array([float(start) for start in logRes["neuroGPUStartTimes"]]) 
    end_data = np.array(logRes["neuroGPUEndTimes"]).astype(float)
    print(len(start_data))
    times = logRes["neuroGPUTimes"]
    total_time = float(logRes['startEndPairs'][-1][1]) - float(logRes['absStart'])
    # bugged timer
    end_data = np.mean(times) + start_data
    procEndDict = logRes['procEndDict']
    sfsMap = logRes['procToSf']
    sfsMapMap = {}
    counter = 0
    for val in set(list(sfsMap.values())):
        sfsMapMap[val] = counter
        counter +=1

    nGpus = 6# THIS SHOULD BE IN LOG RES logRes['nGpus']
    compStartDict = logRes['compStartDict']
    procStartDict = logRes['procStartDict']
    
    startEndPairs = logRes['startEndPairs']
    #print(absStart)

    #print(startEndPairs)

    startEndPairs = [(float(pair0) - float(absStart), float(pair1) - float(absStart)) for pair0, pair1 in startEndPairs]
    #print(procStartDict)
    #print(startEndPairs)
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(13,9))
    
    # YLIM XLIM
    plt.ylim(0,300)
    plt.xlim(0,200)
    
    x_anchors= []
    x_ends = []

    def calc_y_anchor(x_anchor,width,x_anchors,x_ends):
        curr_ht = 0
        for x_anc, x_end in zip(x_anchors,x_ends):
            if x_anchor > x_anc and x_anchor < x_end:
                curr_ht += 10
            elif x_anc > x_anchor and x_anc < x_anchor+width:
                curr_ht += 10
        return curr_ht

    cur_start = 0
    y_base = 0
    firstGo = True
    count = 0 
    for start, end in startEndPairs:
        
        if firstGo:
            firstGo = False
            plt.axvline(x=start,color="blue", label="CPU Eval Start")
            plt.axvline(x=end,color="red", label="CPU Eval Done")
        else:
            plt.axvline(x=start,color="blue")
            plt.axvline(x=end,color="red")
       
        count += 1
        if count > 4:
            break
       

    idx = 1
    labels = list(compStartDict.keys())
    box_ht = 10
    runs = 0
    for procStart,procEnd,proc in \
    zip(list(procStartDict.values()),list(procEndDict.values()), list(procEndDict.keys()) ):
        for pStart,pEnd in zip(procStart,procEnd):
            x_anchor = float(pStart)  - float(absStart)#float(procStart) - float(absStart)
            y_anchor = y_base + 10
            if x_anchor > float(startEndPairs[cur_start][1]) and cur_start < len(startEndPairs) - 1 :
                cur_start += 1
                y_base = 0
                runs += 1
            else:
                y_base += 10
            
            if y_base > 1200: # MAX HEIGHT EXCEEDED
                #print('max height exc.')
                break
            width =(float(pEnd) - float(absStart)) - (float(pStart) - float(absStart)) #(float(procEnd) - float(absStart)) - (float(procStart) - float(absStart))
            #y_anchor = calc_y_anchor(x_anchor,width,x_anchors,x_ends)
            x_anchors.append(float(x_anchor)), x_ends.append(width)
            rect = patches.Rectangle((x_anchor, y_anchor), width, box_ht, \
                                     linewidth=2, edgecolor='black', facecolor='lightblue', fill=True, zorder=0)
            curr_sf = sfsMapMap[sfsMap[proc]]
#             ax.annotate(curr_sf, (x_anchor + 2.5, y_anchor + 5), color='black', weight='bold', \
#                         fontsize=7, ha='center', va='center', zorder=4)
            # Add the patch to the Axes
            ax.add_patch(rect)
            idx += 1
#         if y_base > 1200:
#             print("max height exceeded")
#             break
            if runs > 4:
                break

    # Create a Rectangle patches
    box_ht = 15 # constant box height
    cur_start = 0
    y_base = 0
    runs = 0
    for start,end,idx in zip(start_data,end_data, np.arange(len(end_data))):

        x_anchor = start-absStart
        if x_anchor > float(startEndPairs[cur_start][1]):
            cur_start += 2
            y_base = 0
            runs += 1
        else:
            y_base += 15
        y_anchor = y_base
        width =  end - start
        x_anchors.append(float(x_anchor)), x_ends.append(width)
        rect = patches.Rectangle((x_anchor, y_anchor), width, box_ht, \
                                 linewidth=2.5, edgecolor='black', facecolor='palegreen', fill=True, zorder=10)
#         ax.annotate("GPU {}".format(idx %  nGpus), (x_anchor + (total_time / 10), y_anchor + 8), color='black', weight='bold', 
#                     fontsize=10, ha='center', va='center', zorder=20)

        # Add the patch to the Axes
        #             break
        if runs > 4:
            break
        ax.add_patch(rect)
        
    plt.title("Profile for {} Node Parallel over Population (pop size {})".format(node, pop))
    #plt.title("Custom Profile for {} CPUs, {} Pop Size and {} Nodes".format(nCpus,nodes,popSize))
    plt.legend()
    plt.xlabel("time (s)")
    #plt.show()
    print("TODO: add legend later")
    out_dir = os.path.dirname(f)
    plt.savefig(os.path.join(out_dir,"custom_profile"), bbox_inches='tight')
    plt.close()
    sfsMap = logRes['procToSf']
    sfsMapMap = {}
    counter = 0
    for val in set(list(sfsMap.values())):
        sfsMapMap[val] = counter
        counter +=1

    make_legend(sfsMapMap)
    
    plt.savefig(os.path.join(out_dir,"legend"), bbox_inches='tight')
    plt.close()


def make_legend(top):
    fig, ax = plt.subplots(figsize=(8, 5))
    y = 9
    level = 0
    start = 9
    for name, val in top.items():
        ax.text(start, y - level, str(val) + "--> " + name, fontsize=20)
        level += 1

    ax.axis([0, 10, 0, 10])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')
    plt.show()
    
def time_to_50(abstart, start_end_pairs):
    try:
        end = start_end_pairs[49][1]
        res = float(end) - float(abstart)
    except IndexError:
        print(f"WARNING only {len(start_end_pairs)} start end pairs")
        res = 0
    return res



def calculate_gen_time(start_end_pairs):
    total = []
    num_pairs = len(start_end_pairs)
    # need to start one deep
    for idx in range(1,num_pairs):
        item1 = start_end_pairs[idx][1] 
        item2 = start_end_pairs[idx][0]
        if  len(item1) != 10 or len(item2)  != 10: # case where times are not formatted correctly
            continue
        # difference between gen start and end can't be less than 1 second
        if float(start_end_pairs[idx][1]) - float(start_end_pairs[idx][0]) < 1:
            continue
        
        if  float(start_end_pairs[idx][1]) == float(start_end_pairs[idx][0]):
            continue

        total.append(float(start_end_pairs[idx][1]) - float(start_end_pairs[idx][0]))
    total = np.array(total)
    return np.mean(total), np.std(total)

def calculate_deap_time(start_end_pairs):
    total = []
    num_pairs = len(start_end_pairs)
    # need to start one deep
    for idx in range(1,num_pairs):
        item1 = start_end_pairs[idx][0]
        item2 = start_end_pairs[idx-1][1]
        if  len(item1) != 10 or len(item2)  != 10: # case where times are not formatted correctly
            continue
        
        if  float(item1) - float(item2) < 1 :
            continue
        total.append(float(start_end_pairs[idx][0]) - float(start_end_pairs[idx-1][1]))
    total = np.array(total)
    if len(total) > 25:
        print("MONKEY PATCH INCORRECTLY FORMATTED TIMES")
        total = total[25:]
    else:
        total = total[4:] # drop first 4 
    return np.mean(total), np.std(total)
    
def plot_CPUGPU_bottleneck(nCpus, nodes,pops, sfs, stims, versions, path, how='vanilla', title=None, show=True):
    """
    TODO: consider changing this to be a single plot output
    """
   
    #f  = "runTimeLogs/runTime.log"
    runtimes = []
    labels = []
    fig, axs = plt.subplots()
    plt.subplots_adjust(bottom=None, right=None, top=None, wspace=None, hspace=.5)
    plot_idxs = []
    # Figure size
    #fig, axs = plt.figure(figsize=(10,5))
    if title:
        figname = title + "_cpuVgpu.png"
        title = title + ": Time on CPU vs. GPU"
    pops, nodes = np.array(pops).astype(int), np.array(nodes).astype(int)
    inds = np.argsort(pops/nodes)[::]
    nodes,pops,nCpus,stims, sfs, verss =  nodes[inds],pops[inds],nCpus[inds],stims[inds], sfs[inds], versions[inds]
    seen = []
    for idx,(node,pop,nCpu,stim, sf, vers) in enumerate(zip(nodes,pops,nCpus, stims, sfs, versions)):
        if [pop,node] in seen:
            continue
        else:
            seen.append([pop,node])
        if not title:
            f, title = format_logname(node,pop,nCpu,stim, sf, path, how=how, title=title)
            figname = title + "_cpuVgpu.png"
            title = title + ": Time on CPU vs. GPU"
        else:
            f, _ = format_logname(node,pop,nCpu,stim, sf, path, how=how, title=title)
        
        xlabel = "Population/Node"
        logRes = processLog(f)
        if 'core_neuron' in os.path.abspath('./'):
            # if we are in core neuron we get 8 start end pairs on rank 1, but we
            # only care about the last
            total_time = time_to_50(logRes['absStart'], logRes['startEndPairs'][::8]) 
            mean_deap, std_deap = calculate_deap_time(logRes['startEndPairs'][::8])
        else:
            total_time = time_to_50(logRes['absStart'], logRes['startEndPairs']) 
            mean_deap, std_deap = calculate_deap_time(logRes['startEndPairs'])
            
#         total_time = time_to_50(logRes['absStart'], logRes['startEndPairs'])
#         total_time = total_time / 50
#         mean_deap, std_deap = calculate_deap_time(logRes['startEndPairs'])
        mean_runtime, std_runtime = np.mean(logRes['runtimes']), np.std(logRes['runtimes'])
        mean_eval, std_eval = np.mean(logRes['evalTimes']), np.std(logRes['evalTimes'])
        # import pdb; pdb.set_trace()
        mean_neuroGPU, std_neuroGPU =  np.mean(logRes['neuroGPUTimes']), np.std(logRes['neuroGPUTimes'])
        mean_deap = mean_deap - mean_neuroGPU # DEAP runtime consideration covers first simulation round too so we subtract it

                
        # USE PRE COMPUTED RESULTS INSTEAD FOR CLEAN MEASUREMENTS
        if title == 'Compute Scales and Problem Fixed: Time on CPU vs. GPU':
            deap_pre_recorded = np.genfromtxt(f'../../time_deap/deap_runtimes_{int(pop) // int(node)}.csv', delimiter=',')
            mean_deap, std_deap = np.mean(deap_pre_recorded), np.std(deap_pre_recorded)
        print(f"MEAN DEAP {mean_deap}, MEAN NGPU: {mean_neuroGPU}")

        # Width of a bar 
        width = 0.25       
        # Plotting
        plot_idx = idx * 1.5
        if idx == 0:
#             axs.bar(idx/1.5, total_time , width, label='Full Runtime',color="black")
            axs.bar(plot_idx/1.5, mean_deap , width, yerr=std_deap, label='BluePyOpt', color="orange",edgecolor='black', linewidth=1.5)

            axs.bar(plot_idx/1.5 + width, mean_neuroGPU , width, yerr=std_neuroGPU, label='GPU Simulation',color="palegreen",edgecolor='black', linewidth=1.5)
            axs.bar(plot_idx/1.5 + 2* width, mean_eval, width, yerr=std_eval, label='CPU Evaluation', color="lightblue",edgecolor='black', linewidth=1.5)
            
        else:
#             axs.bar(idx/1.5, total_time , width ,color="black")
            axs.bar(plot_idx/1.5, mean_deap , width, yerr=std_deap, color="orange",edgecolor='black', linewidth=1.5)

            axs.bar(plot_idx/1.5  + width, mean_neuroGPU , width, yerr=std_neuroGPU, color="palegreen", edgecolor='black', linewidth=1.5)
            axs.bar(plot_idx/1.5 + 2*width, mean_eval, width, yerr=std_eval, color="lightblue",edgecolor='black', linewidth=1.5)
        plot_idxs.append(plot_idx)
    # axs.legend(bbox_to_anchor=(1.25, 1), loc='upper right', ncol=1)
    axs.legend(loc='upper left', ncol=1, fontsize=14)
#     axs.set_xlabel(xlabel, font)
    axs.set_xlabel(xlabel)
    axs.set_ylim(0,140)

    axs.set_xticks(ticks=[p_idx/1.5 + .25 for p_idx in plot_idxs])
    
    
    pop_nodes  = np.unique(pops / nodes) # verify axes labels are unqiue
    inds = np.argsort(pop_nodes)[::]
    pops, nodes = pops[inds], nodes[inds]
#     axs.set_xticks(plot_idxs)
    
    axs.set_xticklabels(labels=np.array(np.array(pops).astype(int)/np.array(nodes).astype(int)).astype(int), rotation = 65)
    
    if how == 'vanilla':
        axs.set_xticklabels(labels=["{}/{}".format(node,pop) for node, pop in zip(nodes,pops)])
    elif how == 'stims':
         axs.set_xticklabels(labels=["{}/{}".format(node,stim) for node, stim in zip(nodes,stims)])
    elif how == 'sfs':
         axs.set_xticklabels(labels=["{}/{}".format(node,sf) for node, sf in zip(nodes,sfs)])
    axs.set_ylabel('Time (s)')
    # axs.set_title(title,fontweight='bold')
    fig.savefig(os.path.join(path, figname),  bbox_inches='tight')
    
    
def list_other_logs(f):
    path = os.path.dirname(f)
    files = [file for file in os.listdir(path) if ".log" in file and "gpu" not in file]
    return os.path.join(path,files[0])

def plotScaling(nCpus,nodes,pops, sfs, stims, versions, path, how='vanilla', title=None, show=True):
    #f  = "runTimeLogs/runTime.log"
    runtimes = []
    labels = []
    stds = []
    if title:
        figname = title + "_scaling.png"
        title = title #+ " Scaling"
    for idx,(node,pop,nCpu,stim, sf, vers) in enumerate(zip(nodes,pops,nCpus, stims, sfs, versions)):
        if not title:
            f, title = format_logname(node,pop,nCpu,stim, sf, path, how=how, title=title)
            figname = title + "_scaling.png"
            title = title + " Scaling"
        else:
             f, _ = format_logname(node,pop,nCpu,stim, sf, path, how=how, title=title)
        try:
            logRes = processLog(f)
        except:
            print("found no master log for ", f, " using first")
            prev_f = f
            f =  list_other_logs(prev_f)#re.sub(".log","_0.log", f)
            logRes = processLog(f)
            shutil.copyfile(f, prev_f)
        if len(logRes['runtimes']) < 1:
            continue
        runtime = np.mean(logRes['runtimes'])
        if len(logRes['runtimes']) > 1:
            stds.append(np.std(logRes['runtimes']))
            print("not using back up", node)
        else:
            stds.append(np.mean(backup_stddev[node]))
            print(" using back up standard deviation for {}.... get more trials".format(node))
        if (nodes[0] == nodes).all():
            label = "{}".format(pop)
        else:
            label = "{}".format(node)
        runtimes.append(runtime)
        labels.append(label)
    if (pops[0] == pops).all():
        lin_decr = runtimes[0]/ np.array([label.replace("N","") for label in labels]).astype(int)
        bench_name = 'Ideal'
    elif (nodes[0] == nodes).all():
        lin_decr = [runtimes[i] * (i+1) for i in range(len(runtimes))]
        bench_name = 'Exponential'
    else:
        bench_name = 'Ideal'
        lin_decr = np.repeat(runtimes[0],len(runtimes)) 
    
    fig = plt.figure()  
    
    plt.scatter(np.arange(len(runtimes)), lin_decr, color='orange', label=bench_name, s=15)
    plt.plot(np.arange(len(runtimes)), lin_decr,  color='orange')
    
    ax = fig.axes[0]
    plt.scatter(np.arange(len(runtimes)), runtimes, color='blue', label="Observed", s=15)
    plt.plot(np.arange(len(runtimes)), runtimes,  color='blue')
    runtimes, stds = np.array(runtimes), np.array(stds)
    plt.fill_between(np.arange(len(runtimes)), runtimes-stds, runtimes+stds, alpha=.5)

   
    plt.yscale("log")
    
    if (pops[0] == pops).all():
        plt.ylim(bottom=1)
        plt.xlabel("Nodes")
        #ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())

    elif (nodes[0] == nodes).all():
        
        plt.xlabel("Population")
    else:
        plt.ylim(bottom=10)
        plt.xlabel("Nodes")
        ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())

    plt.xticks(ticks=np.arange(len(runtimes)), labels=labels, rotation=45)
    

    plt.ylabel("Log(Total Runtime (s))")
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(path, figname),  bbox_inches='tight')
    
    

        
        
def compare_scaling(strong_df, weak_df, path):
    #f  = "runTimeLogs/runTime.log"
    #assert (strong_df['offspring'].values == weak_df['offspring'].values).all()
    fig = plt.figure()
    plt.title("Population Scaling Comparison")
    labels = strong_df['offspring'].values
    y = strong_df['Runtime'].values
    err = strong_df['Runtime Stddev'].values
    plt.plot(labels, y, color='blue', label="strong scaling")
    plt.fill_between(labels, y - err, y+ err, color='blue', alpha=.4)
    # revisit this line
    labels = weak_df['offspring'].values
    y = weak_df['Runtime'].values
    err = weak_df['Runtime Stddev'].values
    plt.plot(labels, y, color='red', label="weak scaling")
    plt.fill_between(labels, y - err, y+ err, color='red', alpha=.4)
    
    plt.ylabel("time (s)")
    plt.xlabel("pop size")
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(path, "scaling_time_compare"),  bbox_inches='tight')
    plt.close(fig)
    
    fig2 = plt.figure()
    plt.title("FOM comparison where FOM = pop size / nGPUs / runtime ")
    labels = strong_df['offspring'].values
    y = strong_df['FOM'].values
    plt.plot(labels, y, color='blue', label="strong scaling")
    plt.ylim(bottom=0)

    # revisit this line
    labels = weak_df['offspring'].values
    y = weak_df['FOM'].values
    err = weak_df['Runtime Stddev'].values
    plt.plot(labels, y, color='red', label="weak scaling")
    plt.ylabel("FOM")
    plt.xlabel("pop size")
    plt.legend()
    plt.savefig(os.path.join(path, "scaling_fom_compare"),  bbox_inches='tight')
    plt.close(fig)
    
    
def compare_stim_scaling(strong_df, weak_df, path):
    #f  = "runTimeLogs/runTime.log"
    strong_df = strong_df[strong_df['score functions'] == 20.0]
    strong_df = strong_df.sort_values(by='stims')
    fig = plt.figure()
    plt.title("Stim Scaling Comparison")
    labels = strong_df['stims'].values
    y = strong_df['Runtime'].values
    err = strong_df['Runtime Stddev'].values
    plt.plot(labels, y, color='blue', label="strong scaling")
    plt.fill_between(labels, y - err, y+ err, color='blue', alpha=.4)
    
    
    # MONKEY PATCH
    weak_df =weak_df[~(weak_df['nodes'] >  weak_df['stims'])]
    
    y = weak_df['Runtime'].values
    err = weak_df['Runtime Stddev'].values
    plt.plot(labels, y, color='red', label="weak scaling")
    plt.fill_between(labels, y - err, y+ err, color='red', alpha=.4)
    
    plt.ylabel("time (s)")
    plt.xlabel("number of stims")
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(path, "stim_scaling_time_compare"),  bbox_inches='tight')
    plt.close(fig)

    
def compare_sf_scaling(strong_df, weak_df, path):
    #f  = "runTimeLogs/runTime.log"
    strong_df = strong_df[strong_df['stims'] == 1.0]
    strong_df = strong_df[strong_df['offspring'] == 500.0]
    strong_df = strong_df[strong_df['score functions'] < 71]
    weak_df = weak_df[~((weak_df['score functions'] == 20.0) & (weak_df['nodes'] == 1.0))]
    strong_df = strong_df.sort_values(by='score functions')
    fig = plt.figure()
    plt.title("Score Function Scaling Comparison")
    labels = strong_df['score functions'].values
    y = strong_df['Runtime'].values
    err = strong_df['Runtime Stddev'].values
    plt.plot(labels, y, color='blue', label="strong scaling")
    plt.fill_between(labels, y - err, y+ err, color='blue', alpha=.4)
    
    y = weak_df['Runtime'].values
    err = weak_df['Runtime Stddev'].values
    plt.plot(labels, y, color='red', label="weak scaling")
    plt.fill_between(labels, y - err, y+ err, color='red', alpha=.4)
    
    plt.ylabel("time (s)")
    plt.xlabel("# of score functions")
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(path, "sf_scaling_time_compare"),  bbox_inches='tight')
    plt.close(fig)


def read_exps(exp_names, condition="vanilla", args=None):
    nodes = []
    pops = []
    nCpus = []
    sfs = []
    stims = []
    version_dict = {}
    use_constraint = False
    for exp_name in exp_names:
        try:
            curr_node, curr_core, curr_pop, curr_stims, curr_sfs, version = re.findall(r'\d+', exp_name) # TODO: use version appropriately
        except ValueError as e:
            print(exp_name, " doesn't confrom")
            continue
            
        shared_exp_name = exp_name[:-2]
        if shared_exp_name not in version_dict:
            version_dict[shared_exp_name] = [version]
        else:
            continue
            
        # here we can filter to only use relevant experiments
        # TODO: if one wanted to see scaling in multiple dimensions this will not work
        # need to expand conditions to allow something like "stims_sfs"
        if args and args.constraint_file:
            use_constraint = True
            constraints = {}
            with open(args.constraint_file, "r") as f: 
                lines = f.readlines()
            for line in lines:
                line = line.replace("\n", "")
                line = line.split("=")
                constraints[line[0]] = line[1].split(",")
        if use_constraint and ((curr_stims not in constraints['n_stims']) or  (curr_node not in constraints['N']) \
        or (curr_pop not in constraints['offspring']) or (curr_sfs not in constraints['n_sfs']) or (not len( np.where(np.array(constraints['offspring'])==curr_pop)[0]) > 1 and (np.where(np.array(constraints['offspring'])==curr_pop)[0] != np.where(np.array(constraints['N'])==curr_node)[0]).all())):
            continue
        elif not use_constraint:
            if "vanilla" in condition and (curr_stims != default_stims or curr_sfs != default_sfs):
                continue
           # MONKEY PATCH IF STATEMENT, ex; if you have a study with 2N 500 pop
            elif "vanilla" in condition and int(curr_pop) < (int(curr_node)  * POP_SCALING_FACTOR):
                continue
            elif condition == "stims" and (curr_pop != default_pop or curr_sfs != default_sfs):
                continue
            elif condition == "sfs" and (curr_pop != default_pop or curr_stims != default_stims):
                continue
        

        print("consuming ", curr_node, curr_pop, curr_stims, curr_sfs)
        nodes.append(curr_node), pops.append(curr_pop), nCpus.append(curr_core)
        sfs.append(curr_sfs), stims.append(curr_stims)

    max_version_list = [max(version_dict[key]) for key in version_dict]
    # TODO: fix this monkey patch on which version to use... this always use 0 versino
    #max_version_list = np.zeros(shape=len(nodes)).astype(int).astype(str)
    sort_inds = np.argsort(np.array(nodes).astype(int))
    if len(nodes)< 1:
        print(' NO EXPERIMENTS FOUND')
        print(1/0)
    if (nodes[0] == np.array(nodes)).all():
        sort_inds = np.argsort(np.array(pops).astype(int))
    if 'strong' in condition:
        sort_inds = np.argsort(np.array(pops).astype(int))
#     if condition == "vanilla":
#         sort_inds = np.argsort(np.array(pops).astype(int))
#     elif condition == "stims":
#         sort_inds = np.argsort(np.array(stims).astype(int))
#     elif condition == 'sfs':
#         sort_inds = np.argsort(np.array(sfs).astype(int))
        
    nodes = np.array(nodes)[sort_inds]
    pops = np.array(pops)[sort_inds]
    nCpus = np.array(nCpus)[sort_inds]
    sfs = np.array(sfs)[sort_inds]
    stims = np.array(stims)[sort_inds]
    max_version_list = np.zeros(shape=len(nodes)).astype(int).astype(str)
    max_version_listmax_version_list = np.array(max_version_list)[sort_inds]
    return nodes, pops, nCpus, sfs, stims, max_version_list
        
    
def wrapProfileMaker(nCpus,nodes,pops,stims,sfs, versions, path):
    for idx,(node,pop,nCpu,stim, sf, vers) in enumerate(zip(nodes,pops,nCpus,stims, sfs, versions)):
        if int(node) > 10:
            continue
        makeCustomProfile(node,nCpu,pop,stim ,sf, vers, path)
        
        
def generate_semicircle(center_x, center_y, radius, coverage, stepsize=0.1, num_points=0, direction='right'):
    """
    generates coordinates for a semicircle, centered at center_x, center_y
    """        
    assert coverage <= 1 and coverage >= 0, 'coverage is a percentage of circle to draw'
    
    right_coord = (center_x+(radius)) 
    left_coord = right_coord - (2*radius*coverage) 
    if num_points:
        x = np.linspace(left_coord, right_coord , num_points)
    else:
        x = np.arange(left_coord, right_coord , stepsize)
    
    x[-1] = right_coord # assures stepsize is covered (if it needs to be)

   
    y = np.sqrt(radius**2 - x**2)

    # since each x value has two corresponding y-values, duplicate x-axis.
    # [::-1] is required to have the correct order of elements for plt.plot. 
    x = np.concatenate([x,x[::-1]])

    # concatenate y and flipped y. 
    y = np.concatenate([y,-y[::-1]])
    
    if direction == 'left':
        x = -x - center_y # translate across center line

    return x, y + center_y

def draw_circle_errs(ax, xf,yf,y,radius,width_rat,std,coverage,color):
    mid_y = y[0] - ((y[0] - yf[-1]) / 2)
    scaled_radius = radius -  (radius  - radius*width_rat) / 2
    x_coord1 = np.sqrt(scaled_radius**2  - mid_y**2) - (std*scaled_radius)
    x_coord2 =  np.sqrt(scaled_radius**2 - mid_y**2) + (std*scaled_radius)
    if coverage < .5:
        x_coord1 = -x_coord1
        x_coord2 = -x_coord2

    ax.plot([x_coord1,x_coord2], [mid_y,mid_y ], linewidth=5, color=color)

    ax.plot([x_coord1,x_coord2], [-mid_y,-mid_y ], linewidth=5, color=color)

    
def draw_semicircles(ax, r1, r2, circle_frac, direction, color, x_ctr=0, y_ctr=0):
    x, y = generate_semicircle(x_ctr, y_ctr, r1, circle_frac , stepsize=0.01, direction=direction) # generate cirlce 2 GPU
    x_bot,y_bot = generate_semicircle(x_ctr, y_ctr, r2, circle_frac, stepsize=0.01, direction=direction) # generate cirlce 2 GPU
    xf = np.concatenate((x, x_bot[::-1]))
    yf = np.concatenate((y, y_bot[::-1]))
    p2 = ax.fill(xf, yf, edgecolor='black', facecolor=color, linewidth=1.5)
    
    return xf, yf, x, y

def draw_circles_with_row(row, ax):
    radii = [5,15,15]
    width = 4
    width_rat = .3 # ratio for width of circles
    
    
    eval_coverage = row['Mean Eval Time'] / row['Runtime']
    xf, yf, x, y = draw_semicircles(ax, radii[2], radii[2]*width_rat, circle_frac=eval_coverage, direction='right', color='lightblue')
    # draw_circle_errs(ax, xf,yf,y,radii[2], width_rat, row['Std Eval Time'] / row['Runtime'], eval_coverage, 'blue')
    ax.text(max(x) + 2 ,0, "%.2f %% \n CPU \n execution time" % (eval_coverage * 100), zorder=30)


    gpu_coverage = row['Mean Sim Time'] / row['Runtime']
    xf, yf, x, y = draw_semicircles(ax, radii[1], radii[1]*width_rat, circle_frac=gpu_coverage, direction='left', color='palegreen')
    ax.text(min(x) - radii[1]*.8 ,0, "%.2f %%  \n GPU \n execution \n time" % (gpu_coverage * 100), zorder=30) # adds another .7 radius for text

    # draw_circle_errs(ax, xf,yf,y,radii[1], width_rat, row['Std Sim Time'] , gpu_coverage, 'green')
    ax.text(-3.5,-1, "%.2f s \n total \n run time" % row['Runtime'], zorder=30)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')
    
    
    
    total_x,total_y = generate_semicircle(0, 0, radii[0], 1, stepsize=0.01, direction='right') # generate circle 1
    ax.fill(total_x,total_y,color='salmon', label='total', linewidth=2,edgecolor='black', zorder=10)

    
    ax.set_xlim(-25,25)
    
    # TODO:
    #     1. add confidence lines (skipped)
    #     2. change title to pop / node (done)
    #     3. add runtime and % anntoations (done)
    #     4. remove axes (done)
    #     5. only make half as many plots as you are now (done)
    #     6. maybe change center color (done)
    # OTHER TODO:
    #     1. GPU utilization line plot over time (DONE)
    #     2. Table with runtimes + max runtime, min runtime and total to 50
    
def plot_gpu_pies(df, figname):
    df = df.drop_duplicates(subset=["Nodes", "Total Gpus", "Offspring", "Stimuli", "Score Functions"])
    rows =int(np.sqrt(len(df)))
    cols = len(df) // rows
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(rows*4, cols*4))
    plt.subplots_adjust(bottom=None, right=None, top=None, wspace=.3, hspace=None)

    pops = [250,500,1500,3000]
    df = df.loc[np.where(np.isin(df['Offspring'].values, pops))[0]]
    for ind, ax in enumerate(axs.flatten()):
        row = df.iloc[ind]
        if row['Offspring'] in pops:
            draw_circles_with_row(row, ax)
        else:
            ax.set_visible(False)


        labels=['% of Time on GPU', '% of Time GPU Idle']
        # ax.pie(x, labels=labels)
        # ax.set_title("{} Nodes, {} GPUs, {} Stims, {} Pop".format(int(row['Nodes']),\
        #                                                           int(row['Total Gpus']),\
        #                                                           int(row['Stimuli']),\
        #                                                           int(row['Offspring'])))
        ax.set_title("{} Individuals / Node".format(int(row['Offspring'])), fontsize=20)
    plt.savefig("outputs/{}_Pie.png".format(figname))
    plt.close(fig)
          
def drop_constant(df, preserve_list=[]):
    res = df.loc[:, (df != df.iloc[0]).any()] 
    if len(preserve_list) > 0:
        preserved_cols = df.loc[:,preserve_list]
        dropped = list(preserved_cols.columns)
        res.loc[:,preserve_list] = preserved_cols
        cols = list(res)
        if 'Nodes' in dropped:
            cols.insert(0, cols.pop(cols.index('Nodes')))
            cols.insert(1, cols.pop(cols.index('Offspring')))
        elif 'Offspring' in dropped:
            cols.insert(2, cols.pop(cols.index('Offspring')))
        res = res.loc[:, cols]
    return res

def plus_minus_cols(df, main, std, drop=True):
    df[main] = df[main].astype(str).apply(lambda x: x[:5]) \
    + " ± " + df[std].astype(str).apply(lambda x: x[:5])
    if drop:
        df = df.drop(std,axis=1)
    return df


# def calculate_deap_time(start_end_pairs):
#     total = []
#     num_pairs = len(start_end_pairs)
#     # need to start one deep
#     # import pdb; pdb.set_trace()
#     for idx in range(1,num_pairs):
#         total.append(float(start_end_pairs[idx][1]) - float(start_end_pairs[idx][0]))
#     total = np.array(total)
#     # import pdb; pdb.set_trace()

#     return np.mean(total), np.std(total)

def generate_result_table(nCpus,nodes,pops, sfs, stims, versions, path, title=None, how='vanilla'):
    df = pd.DataFrame(columns=['Nodes','Total Cpus', 'Total Gpus',\
                               'Offspring', 'Stimuli', 'Score Functions', \
                               'Runtime', 'Runtime Stddev', 'FOM', 'FOM Std Dev','GPU Utilization', 'Mean Eval Time', 'Std Eval Time', 'Min Eval Time', 'Max Eval Time', 'Mean Sim Time', 'Std Sim Time', 'Min Sim Time', 'Max Sim Time', 'DEAP time', 'DEAP Time Std Dev', 'Time to 50 Gen', 'Mean Gen Size', 'Std Gen Size', 'Num Trials'])
    if title:
        figname = title + "_scale.tex"
        df_name =  title + "_scale.csv"
    
    fn_to_gpu_df = {}
    for idx,(node,pop,nCpu,stim, sf, vers) in enumerate(zip(nodes,pops,nCpus, stims, sfs, versions)):
        if not title:
            f, title = format_logname(node,pop,nCpu,stim, sf, path, how=how, title=title)
            gpu_logs, num_logs = format_gpu_util_name(node,pop,nCpu,stim, sf, path, how=how, title=title)
            figname = title + "_scale.tex"
            df_name =  title + "_scale.csv"
        else:
            f, _ = format_logname(node,pop,nCpu,stim, sf, path, how=how, title=title)
            gpu_logs, num_logs = format_gpu_util_name(node,pop,nCpu,stim, sf, path, how=how, title=title)
        
        
        if len(gpu_logs) > 0:
            fn = gpu_logs[0]
            try:
                percent_utilization, gpu_df = read_gpu_logs(fn)
                fn_to_gpu_df[fn] = gpu_df
            except Exception as e:
                print(e)
                percent_utilization = 0
            
       
        logRes = processLog(f)
        if 'core_neuron' in os.path.abspath('./'):
            # if we are in core neuron we get 8 start end pairs on rank 1, but we
            # only care about the last
            total_time = time_to_50(logRes['absStart'], logRes['startEndPairs'][::8]) 
        else:
            total_time = time_to_50(logRes['absStart'], logRes['startEndPairs']) 
        mean_deap, std_deap = calculate_deap_time(logRes['startEndPairs'])
        mean_runtime, std_runtime = np.mean(logRes['runtimes']), np.std(logRes['runtimes'])
        mean_eval, std_eval = np.mean(logRes['evalTimes']), np.std(logRes['evalTimes'])
        min_eval, max_eval = np.min(logRes['evalTimes']), np.max(logRes['evalTimes'])
        mean_neuroGPU, std_neuroGPU =  np.mean(logRes['neuroGPUTimes']), np.std(logRes['neuroGPUTimes'])
        min_neuroGPU, max_neuroGPU = np.min(logRes['neuroGPUTimes']), np.max(logRes['neuroGPUTimes'])
        mean_gen_size, std_gen_size =  np.mean(logRes['gen_sizes']), np.std(logRes['gen_sizes'])
        nGpus = logRes['nGpus']
        FOM = int(pop)/(nGpus*int(node))/np.array(logRes['runtimes'], dtype=np.float64)
        fom_mean = np.mean(FOM)
        fom_dev = np.std(FOM)
        num_trials = len(logRes['neuroGPUTimes'])
        # print(f"MEAN DEAP {mean_deap}, MEAN NGPU: {mean_neuroGPU}")
        mean_deap = mean_deap - mean_neuroGPU # DEAP runtime consideration covers first simulation round too so we subtract it
        pct_util = float(percent_utilization)
        if np.isnan(mean_runtime):
            continue
        df.loc[idx] = [int(node),int(nCpu)*int(node), nGpus*int(node), \
                     int(pop),int(stim), int(sf), float(mean_runtime), float(std_runtime), fom_mean, fom_dev, pct_util, mean_eval, std_eval, min_eval, max_eval, mean_neuroGPU, std_neuroGPU, min_neuroGPU, max_neuroGPU, mean_deap, std_deap, total_time, mean_gen_size, std_gen_size, num_trials]
    df = df.sort_values('Nodes', ascending=True) 
    
    
    # SAVE CSV
    df.to_csv(os.path.join(path, df_name))
    skip_latex=True
    if not skip_latex:
    # SAVE LATEX
        latex_df = drop_constant(df, preserve_list=['Nodes','Offspring'] )
        latex_df = plus_minus_cols(latex_df, main='Runtime',std='Runtime Stddev')
        latex_df = plus_minus_cols(latex_df, main='FOM',std='FOM Std Dev')
        latex_df['GPU Utilization'] = latex_df['GPU Utilization'].astype(str).apply(lambda x: x[:4])  + "%"
    #     formaters =  {"Runtime": "{:0.2f}".format, "Runtime Stddev":  "{:0.4f}".format,   "cori fom" : "{:0.2f}".format, "fom std dev" : "{:0.3f}".format,  'gpu_util': "{:0.2f}".format }
    #     df.to_latex(os.path.join(path, figname), formatters=formaters, float_format="%.0f", index=False)
        col_fmt = "|".join(np.repeat('c', len(df.columns)))
        col_fmt = "|" + col_fmt + "|"
        latex_df.to_latex(os.path.join(path, figname), float_format="%.0f", index=False, column_format=col_fmt)
    else:
        print("WARNING: skipped latex")

        
    print("WARNING: made a bunch of gpu dfs but not doing much with em .. could plot")

    return df, fn_to_gpu_df
    
def gpu_util_timeline(gpu_dict, name):
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(13,9))
    plt.subplots_adjust(bottom=None, right=None, top=None, wspace=None, hspace=.9)

    axs = axs.flatten()
    ax_counter = 0 
    pops = [250,500,1000,3000]
    for key in gpu_dict:
        parsed_key = re.findall(r'\d+', key)
        node, offspring, stim, sf = int(parsed_key[0]), int(parsed_key[2]), int(parsed_key[3]), int(parsed_key[4])
        if offspring not in pops:
            continue
        gpu_df = gpu_dict[key]
        gpu_df.loc[:,'timestamp'] = pd.to_datetime(gpu_df['timestamp'])
        gpu_df = gpu_df.sort_values(by='timestamp', ascending=True)
        first_idx = gpu_df['utilization'].gt(0).idxmax()  # start plotting where simulation starts
        gpu_df = gpu_df.iloc[first_idx:,:]

        gpu_df.index = gpu_df['timestamp']

        for i in range(8):
            curr_gpu_df = gpu_df.iloc[i::8, :]
            ds_gpu_df = curr_gpu_df.resample('3S').mean() # downsample to 3S timing
            ax = axs[ax_counter]
            x_vals = (ds_gpu_df.index.values - ds_gpu_df.index.values[0]).astype(int) / 1000000000
            # ^ nano second into secons
            ax.plot(x_vals, ds_gpu_df['utilization'].values, label=f'gpu{i}')
        ax.set_title(f'{offspring} offspring / node')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('GPU % Utilization')
        ax_counter += 1
    plt.legend(loc=(1,-.2))
    fig.savefig("outputs/{}_GPUtil.png".format(name), bbox_inches='tight')
            
    
def write_all_files(dest, srcs, overwrite=True):
    if not overwrite:
        assert srcs[0][-6:-4] == 'SF', f"this should not be a minion log, but it ends in a {srcs[0][-6:-4] } instead of SF.log"
        srcs = srcs[1:] 
        open_mode = 'a+'
    else:
        open_mode = 'w'
    with open(dest, open_mode) as outfile: # dest is in src, but gets cleaned out here, good
        for fname in srcs:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                    
def check_for_first_merge(log_path, old_log):
    """
    check that this 'old log' is should wipe out and replace log path
    you only want this if old list is the first log of it's kind to get processed
    """
    listing = sorted(glob.glob(os.path.dirname(log_path) + '*'))[1:] # check all matching folders
    assert len(listing) > 0, f"only master dir exists for {log_path}"
    matching = os.path.basename(listing[0]) == os.path.basename(old_log).replace('.log', '')
    return matching
                
    
def merge_experiments(src, dest, version, path):
    src_path = os.path.join(path,src)
    curr_node, curr_core, curr_pop, curr_stims, curr_sfs, version = re.findall(r'\d+', src) # TODO: use version appropriately
    prof_name = "{}N_{}C_{}O_{}S_{}SF.prof".format(curr_node, curr_core, curr_pop, curr_stims, curr_sfs)
    new_prof_name = "{}N_{}C_{}O_{}S_{}SF.{}.prof".format(curr_node, curr_core, curr_pop, curr_stims, curr_sfs, version)
    # move profile
    if not os.path.isdir(dest):
        os.makedirs(dest)
    try:
        shutil.copy(os.path.join(src_path,prof_name), os.path.join(dest,new_prof_name))
    except FileNotFoundError:
        print("no profile for ", src_path)
    # move gpu util log
    gpu_util_logname = "gpu_utillization.log"
    new_gpu_util_logname = "gpu_utillization.{}.log".format(version)
    try:
        shutil.copy(os.path.join(src_path,gpu_util_logname), os.path.join(dest,new_gpu_util_logname))
    except FileNotFoundError:
        print("no gpu util for ", src_path)
    # comebine regular log
    log_path = os.path.join(dest, "{}N_{}C_{}O_{}S_{}SF.log".format(curr_node, curr_core, curr_pop, curr_stims, curr_sfs, version))
    old_log = os.path.join(src_path, "{}N_{}C_{}O_{}S_{}SF_{}.log".format(curr_node, curr_core, curr_pop, curr_stims, curr_sfs, version))
    
    first_merge = check_for_first_merge(log_path, old_log)

    if os.path.isfile(log_path) and os.path.isfile(old_log) and first_merge: # you've already processed folder
        write_all_files(log_path, [log_path, old_log])
    elif os.path.isfile(log_path) and os.path.isfile(old_log):
        write_all_files(log_path, [log_path, old_log], overwrite=False)
    elif  os.path.isfile(old_log): # you haven't processed folder
        #assert int(version) == 1, "no master log and version is not 1?"
        write_all_files(log_path, [old_log])
    else:
        print("WARNING: No log merge for ", log_path)
        print("not deleting ... could be though")
        print(src_path, log_path)
#         print(1/0)
#         shutil.rmtree(src_path)
 


def collapse_exps(exp_names, path):
    nodes = []
    pops = []
    nCpus = []
    stims = []
    sfs = []
    exp_names = sorted(exp_names)
    seen = []
    for exp_name in exp_names:
        try:
            curr_node, curr_core, curr_pop, curr_stims, curr_sfs, version = re.findall(r'\d+', exp_name) # TODO: use version appropriately
        except ValueError as e:
            print(exp_name, " doesn't confrom")
            continue
        
        nodes.append(curr_node), pops.append(curr_pop), nCpus.append(curr_core)
        sfs.append(curr_sfs), stims.append(curr_stims)
        agg_exp_path =  os.path.join(path,"{}N_{}C_{}O_{}S_{}SF".format(curr_node, curr_core, curr_pop, curr_stims, curr_sfs))
        # changed == to !=
        exp_seen = False
        if  agg_exp_path in seen:
            exp_seen = True
        else:
            seen.append(agg_exp_path)

        if not exp_seen or not os.path.isdir(agg_exp_path):
            if os.path.isdir(agg_exp_path):
                shutil.rmtree(agg_exp_path)
            shutil.copytree(os.path.join(path,exp_name), agg_exp_path)
            log_name = [file for file in os.listdir(agg_exp_path) if ".log" in file and "gpu" not in file]
            try:
                os.rename(os.path.join(agg_exp_path,log_name[0]), os.path.join(agg_exp_path,log_name[0][:-6] + log_name[0][-4:] ))
                print(log_name)
            except:
                print("no log for : ", agg_exp_path)
                print(os.path.dirname(agg_exp_path))
#                 shutil.rmtree(os.path.dirname(agg_exp_path)

                continue
            
        else:
            merge_experiments(exp_name, agg_exp_path, version, path )
       
    return 
        
    
def sf_plot_strategy(exp_names, args, collapse=False):
    if collapse:
        collapse_exps(exp_names, args.path)
    nodes, pops,  nCpus, sfs, stims, versions = read_exps(exp_names, condition='sfs', args=args)
    plt.title("Population Size Scaling w. Nodes")
    # step 1
    plotScaling(nCpus,nodes,pops, sfs, stims, versions, args.path, how='sfs')
    # step 2
    #wrapProfileMaker(nCpus, nodes, pops, versions)
    
    # step 3
    plot_CPUGPU_bottleneck(nCpus,nodes,pops, sfs, stims, versions, args.path, how='sfs')

    # step 4
    generate_result_table(nCpus,nodes,pops, sfs, stims, versions, args.path, how='sfs')
    
def stim_plot_strategy(exp_names, args, collapse=False):
    
    if collapse:
        collapse_exps(exp_names, args.path)
    nodes, pops,  nCpus, sfs, stims, versions = read_exps(exp_names, condition='stims', args=args)
    plt.title("Population Size Scaling w. Nodes")
    # step 1
    how ='vanilla'
    if args.constraint_file:
        how = 'full'
    plotScaling(nCpus,nodes,pops, sfs, stims, versions, args.path, how=how)
    # step 2
    #wrapProfileMaker(nCpus, nodes, pops, versions)
    
    # step 3
    plot_CPUGPU_bottleneck(nCpus,nodes,pops, sfs, stims, versions, args.path, how=how)

    # step 4
    generate_result_table(nCpus,nodes,pops, sfs, stims, versions, args.path, how=how)
    
        
    
def vanilla_plot_strategy(exp_names, args, collapse=False):
    
    if collapse:
        collapse_exps(exp_names,args.path)
    print("NOT COLLAPSING CHANGE L8R")
    nodes, pops,  nCpus, sfs, stims, versions = read_exps(exp_names, args=args)
    plt.title("Population Size Scaling w. Nodes")
    set_custom_params_plt()
    print("CUSTOMING PARAMS")
    # step 1
    how ='vanilla'
    title = None
    figname='population'
    if args.constraint_file:
        how = 'full'
        title =  os.path.basename(args.constraint_file)
        figname =  os.path.basename(args.constraint_file)
        
    # plotScaling(nCpus,nodes,pops, sfs, stims, versions, args.path, how=how, title=title)
    # step 2
   # wrapProfileMaker(nCpus, nodes, pops, stims, sfs,  versions,  args.path)
    
    # step 3
    plot_CPUGPU_bottleneck(nCpus,nodes,pops, sfs, stims, versions, args.path, how=how, title=title)

    # step 4
    df, gpu_df_dict = generate_result_table(nCpus,nodes,pops, sfs, stims, versions, args.path, how=how, title=title)
    
    
    restore_default_mpl_params()
    # step 5      
    # plot_gpu_pies(df,figname)
    # gpu_util_timeline(gpu_df_dict, figname)
    
    
        
    
def strong_plot_strategy(exp_names, args, collapse=False):
    
    if collapse:
        collapse_exps(exp_names, args.path)
        
    how = "strong_vanilla" 
    weak_name = 'pop_scale.csv'
    if args.stims:
        how = "strong_stims" 
        weak_name= "stim_scale.csv"
    elif args.sfs:
        how = "strong_sfs" 
        weak_name  = 'sf_scale.csv'
    nodes, pops,  nCpus, sfs, stims, versions = read_exps(exp_names, condition=how, args=args)
    plt.title("Population Size Scaling w. Nodes")

    # step 4
    strong_df = generate_result_table(nCpus,nodes,pops, sfs, stims, versions, args.path, how=how)
    weak_df = pd.read_csv(os.path.join("weak_outputs",weak_name))
    if not args.stims and not args.sfs:
        compare_scaling(strong_df, weak_df)
    elif args.stims:
        compare_stim_scaling(strong_df, weak_df)
    elif args.sfs:
        compare_sf_scaling(strong_df, weak_df)
        
def check_collapse(exp_names, path):
    for exp_name in exp_names:
        if os.path.isfile(os.path.join(path, exp_name)):
            continue
        if not os.path.isdir(os.path.join(path, exp_name.split("SF")[0] + "SF")):
            return True
    return False

def find_largest_std(exp_names, args):
    stds = {}
    nodes, pops,  nCpus, sfs, stims, versions = read_exps(exp_names, condition="permissive")
    for node, pop, nCpu, sf, stim, version in zip(nodes, pops,  nCpus, sfs, stims, versions):
        f, title = format_logname(node, pop, nCpu, stim, sf, args.path, 'howwww')
        try:
            logRes = processLog(f)
        except:
            continue
#             print("found no master log for ", f, " using first")
#             prev_f = f
#             f = re.sub(".log","_0.log", f)
#             logRes = processLog(f)
#             shutil.copyfile(f, prev_f)
        mean_runtime, std_runtime = np.mean(logRes['runtimes']), np.std(logRes['runtimes'])
        # ignore an obvious case where algorithm glitched for some reason and took 300 seconds or something
        if node == '1' and std_runtime > 20:
            continue
        if node in stds:
            stds[node].append(std_runtime)
        else:
            stds[node] = [std_runtime]
    for key in stds:
        print(key, np.mean(stds[key]))
    with open("std_dev_backup.pkl",'wb') as f:
        pickle.dump(stds,f)
    exit()


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Benchmarking viz')


    parser.add_argument('--stims', action="store_true")
    parser.add_argument('--sfs', action="store_true")
    parser.add_argument('--strong', action="store_true")
    parser.add_argument('--constraint_file', type=str, required=False, default=None)
    parser.add_argument('--path', type=str, required=False, default="outputs")

    args = parser.parse_args()
    
    exp_names = [dirname for dirname in os.listdir(args.path) if "_" in dirname and "ipynb" not in dirname] # make this more strict later --> should match coresnodes_POPSIZE_iteration
    #find_largest_std(exp_names, args)
 
    collapse = check_collapse(exp_names, args.path)
#     if args.path != 'outputs':
    print(collapse, "SHOULD I COLLAPSE ?? IM SPITTING THESE RAPS TIL THE DAY THAT I DROP")
    if args.stims:
        stim_plot_strategy(exp_names, args, collapse=collapse)
    elif args.sfs:
        sf_plot_strategy(exp_names, args, collapse=collapse)
    else:
        vanilla_plot_strategy(exp_names, args, collapse=collapse)
        
    if args.strong:
        strong_plot_strategy(exp_names, args, collapse=collapse)
    
        
    
    
    