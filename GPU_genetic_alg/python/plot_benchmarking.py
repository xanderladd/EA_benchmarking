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



parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Benchmarking viz')


parser.add_argument('--stims', action="store_true")
parser.add_argument('--sfs', action="store_true")
parser.add_argument('--strong', action="store_true")
parser.add_argument('--constraint_file', type=str, required=False, default=None)
parser.add_argument('--path', type=str, required=False, default="outputs")




default_sfs = str(20)
default_stims = str(1)
default_pop = str(500)
POP_SCALING_FACTOR=500

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
        file_lines =file.readlines()
        readingEnds = False
        readingStarts = False
        for line in file_lines:
            if "Date:" in line:
                print(line)
            if "absolute start" in line:
                numbers = re.findall(r'\d+', line)
                abs_start = [ '.'.join(x) for x in zip(numbers[0::2], numbers[1::2]) ][0]
            if "nCpus" in line:
                nCpus = int(re.match('.*?([0-9]+)$', line).group(1))
                #assert nCpus  == cpu, "expected {} but got {} cpus in log".format(cpu,nCpus)
            if "took:" in line:
                runtime = float(re.findall(r"[-+]?\d*\.\d+|\d+",line)[1])
                #print(runtime)
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
            if "evaluation:" in line:
                numbers = re.findall(r'\d+', line)
                numbers = [ '.'.join(x) for x in zip(numbers[0::2], numbers[1::2]) ]
                if "evalTimes" in locals():
                    evalTimes = np.append(evalTimes,  np.array(list(numbers), dtype=np.float32))
                else:
                    evalTimes = np.array(list(numbers), dtype=np.float32)
                avgEval = np.mean(evalTimes)
            if "neuroGPU" in line and "starts" not in line and "ends" not in line:
                numbers = re.findall(r'\d+', line)
                numbers = [ '.'.join(x) for x in zip(numbers[0::2], numbers[1::2]) ]
                
                if "neuroGPUTimes" in locals():
                    neuroGPUTimes = np.append(neuroGPUTimes,  np.array(list(numbers), dtype=np.float32))
                else:
                    neuroGPUTimes = np.array(list(numbers),dtype=np.float32)
                avgNGPU = np.mean(neuroGPUTimes)
            if ("neuroGPU" in line and "starts" in line and "ends" not in line) or readingEnds:
                readingEnds = True
                numbers = re.findall(r'\d+', line)
                numbers = [ '.'.join([x1,x2]) + "e+" + str(x3) for x1,x2,x3 in zip(numbers[0::3], numbers[1::3], numbers[2::3]) ]
                neuroGPUStartTimes += numbers
                if "]" in line:
                    readingEnds = False
            if ("neuroGPU" in line and "starts" not in line and "ends" in line) or readingStarts:
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
#             if "gen1 took" in line:
#                 break
    try:
        res = {"procStartDict": procStartDict,"procEndDict": procEndDict,\
               "startEndPairs": startEndPairs,"runtimes": runtimes,\
               "compStartDict": compStartDict,"sfs": sfs,\
               "evalTimes": evalTimes,"neuroGPUTimes": neuroGPUTimes,\
              "procToSf": procToSf, "absStart": abs_start, \
               "neuroGPUStartTimes": neuroGPUStartTimes, \
               "neuroGPUEndTimes": neuroGPUEndTimes, "ioTimes": io_times}
    except UnboundLocalError as e:
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

    

def makeCustomProfile(node,nCpu,pop, vers, show=True):
    if vers == "0": # why is vers a string?
        f  = "outputs/{}C{}N_{}_{}/{}C{}N_{}.log".format(node,nCpu, pop, vers, node,nCpu, pop)
    else:
        f  = "outputs/{}C{}N_{}_{}/{}C{}N_{}_{}.log".format(node,nCpu, pop, vers, node,nCpu, pop, int(vers)-1)
    #f  = "runTimeLogs/runTime.log"
    logRes = processLog(f)
    absStart = float(logRes['absStart'])
    start_data = np.array([float(start) for start in logRes["neuroGPUStartTimes"]]) 
    end_data = np.array(logRes["neuroGPUEndTimes"]).astype(float)
    print(len(start_data))
    times = logRes["neuroGPUTimes"]
    # bugged timer
    end_data = np.mean(times) + start_data
    procEndDict = logRes['procEndDict']
    sfsMap = logRes['procToSf']
    sfsMapMap = {}
    counter = 0
    for val in set(list(sfsMap.values())):
        sfsMapMap[val] = counter
        counter +=1

    nGpus = 8# THIS SHOULD BE IN LOG RES logRes['nGpus']
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
    plt.ylim(0,600)
    #plt.xlim(0,120)
    
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
    for start, end in startEndPairs:
        if firstGo:
            firstGo = False
            plt.axvline(x=start,color="blue", label="CPU Eval Start")
            plt.axvline(x=end,color="red", label="CPU Eval Done")
        else:
            plt.axvline(x=start,color="blue")
            plt.axvline(x=end,color="red")

    idx = 1
    labels = list(compStartDict.keys())
    box_ht = 10
    for procStart,procEnd,proc in \
    zip(list(procStartDict.values()),list(procEndDict.values()), list(procEndDict.keys()) ):
        for pStart,pEnd in zip(procStart,procEnd):
            x_anchor = float(pStart)  - float(absStart)#float(procStart) - float(absStart)
            y_anchor = y_base + 10
            if x_anchor > float(startEndPairs[cur_start][1]) and cur_start < len(startEndPairs) - 1 :
                cur_start += 1
                y_base = 0
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
            ax.annotate(curr_sf, (x_anchor + 2.5, y_anchor + 5), color='black', weight='bold', \
                        fontsize=7, ha='center', va='center', zorder=4)
            # Add the patch to the Axes
            ax.add_patch(rect)
            idx += 1
#         if y_base > 1200:
#             print("max height exceeded")
#             break

    # Create a Rectangle patches
    box_ht = 15 # constant box height
    cur_start = 0
    y_base = 0
    for start,end,idx in zip(start_data,end_data, np.arange(len(end_data))):
        x_anchor = start-absStart
        if x_anchor > float(startEndPairs[cur_start][1]):
            cur_start += 2
            y_base = 0
        else:
            y_base += 15
        y_anchor = y_base
        width =  end - start
        x_anchors.append(float(x_anchor)), x_ends.append(width)
        rect = patches.Rectangle((x_anchor, y_anchor), width, box_ht, \
                                 linewidth=2.5, edgecolor='black', facecolor='palegreen', fill=True, zorder=10)
        ax.annotate("GPU {}".format(idx %  nGpus), (x_anchor + 10, y_anchor + 8), color='black', weight='bold', 
                    fontsize=10, ha='center', va='center', zorder=20)

        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.title("Profile for {} Node Parallel over Population (pop size {})".format(node, pop))
    #plt.title("Custom Profile for {} CPUs, {} Pop Size and {} Nodes".format(nCpus,nodes,popSize))
    plt.legend()
    plt.xlabel("time (s)")
    #plt.show()
    print("TODO: add legend later")
    plt.savefig("outputs/{}C{}N_{}_{}/custom_profile".format(node,nCpu, pop, vers), bbox_inches='tight')
    plt.close()
    sfsMap = logRes['procToSf']
    sfsMapMap = {}
    counter = 0
    for val in set(list(sfsMap.values())):
        sfsMapMap[val] = counter
        counter +=1

    make_legend(sfsMapMap)
    plt.savefig("outputs/{}C{}N_{}_{}/legend".format(node,nCpu, pop, vers), bbox_inches='tight')
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
    
    
def plot_CPUGPU_bottleneck(nCpus, nodes,pops, sfs, stims, versions, how='vanilla', show=True):
    """
    TODO: consider changing this to be a single plot output
    """
   
    #f  = "runTimeLogs/runTime.log"
    runtimes = []
    labels = []
    fig, axs = plt.subplots(figsize=(10,10))
    plt.subplots_adjust(bottom=None, right=None, top=None, wspace=None, hspace=.5)
    # Figure size
    #fig, axs = plt.figure(figsize=(10,5))

    for idx,(node,pop,nCpu,stim, sf, vers) in enumerate(zip(nodes,pops,nCpus, stims, sfs, versions)):
        if how == 'vanilla':
            f  = "outputs/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(node,nCpu, pop, default_stims, default_sfs, node,nCpu, pop, default_stims, default_sfs)
            label = "{}N_{}O".format(node, pop)
            title = "Population  Load Balance"
            xlabel="Nodes / Population Size"
            figname = "CpuGpuPop"
        elif how == 'stims':
            f  = "outputs/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(node,nCpu, default_pop, stim, default_sfs, node,nCpu,default_pop, stim, default_sfs)
            label = "{}N_{}S".format(node, stim)
            title = "Stim  Load Balance"
            xlabel="Nodes / # of Stim"
            figname = "CpuGpuStim"
        elif how == "sfs":
            f  = "outputs/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(node,nCpu, default_pop, default_stims, sf, node,nCpu,default_pop, default_stims, sf)
            label = "{}N_{}SF".format(node, sf)
            title = "Score Function Load Balance"
            xlabel="Nodes / # of SFs"
            figname = "CpuGpuSF"
        else:
            raise NotImplementedError
    
        logRes = processLog(f)
        mean_runtime, std_runtime = np.mean(logRes['runtimes']), np.std(logRes['runtimes'])
        mean_eval, std_eval = np.mean(logRes['evalTimes']), np.std(logRes['evalTimes'])
        mean_neuroGPU, std_neuroGPU =  np.mean(logRes['neuroGPUTimes']), np.std(logRes['neuroGPUTimes'])

        # Width of a bar 
        width = 0.17       
        # Plotting
        if idx == 0:
            axs.bar(idx/1.5, mean_neuroGPU , width, yerr=std_neuroGPU, label='Running time on GPU',color="green")
            axs.bar(idx/1.5 + width, mean_eval, width, yerr=std_eval, label='Running time on CPU', color="blue")
        else:
            axs.bar(idx/1.5, mean_neuroGPU , width, yerr=std_neuroGPU,color="green")
            axs.bar(idx/1.5 + width, mean_eval, width, yerr=std_eval, color="blue")
            
    axs.legend(bbox_to_anchor=(1.25, 1), loc='upper right', ncol=1)
    axs.set_xlabel(xlabel)
    axs.set_xticks(ticks=[i/1.5 for i in range(len(nCpus))])
    if how == 'vanilla':
        axs.set_xticklabels(labels=["{}/{}".format(node,pop) for node, pop in zip(nodes,pops)])
    elif how == 'stims':
         axs.set_xticklabels(labels=["{}/{}".format(node,stim) for node, stim in zip(nodes,stims)])
    elif how == 'sfs':
         axs.set_xticklabels(labels=["{}/{}".format(node,sf) for node, sf in zip(nodes,sfs)])
    axs.set_ylabel('time (s)')
    axs.set_title(title)
    fig.savefig(os.path.join("outputs", figname),  bbox_inches='tight')
    

def format_logname(node, pop, nCpu, stim, sf, how, title=None):
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
    return "outputs/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(node,nCpu, pop, stim, sf, node,nCpu,pop, stim, sf), title
    
def plotScaling(nCpus,nodes,pops, sfs, stims, versions, how='vanilla', title=None, show=True):
    #f  = "runTimeLogs/runTime.log"
    runtimes = []
    labels = []
    if title:
        figname = title + "_scaling.png"
        title = title + " Scaling"
    for idx,(node,pop,nCpu,stim, sf, vers) in enumerate(zip(nodes,pops,nCpus, stims, sfs, versions)):
        if not title:
            f, title = format_logname(node,pop,nCpu,stim, sf, how=how, title=title)
            figname = title + "_scaling.png"
            title = title + " Scaling"
        else:
             f, _ = format_logname(node,pop,nCpu,stim, sf, how=how, title=title)
#         if how == 'vanilla':
#             f  = "outputs/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(node,nCpu, pop, default_stims, default_sfs, node,nCpu, pop, default_stims, default_sfs)
# #             label = "{}N_{}O".format(node, pop)
#             title = "Population Scaling"
#             figname = "PopScaling"
#         elif how == 'stims':
#             f  = "outputs/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(node,nCpu, default_pop, stim, default_sfs, node,nCpu,default_pop, stim, default_sfs)
# #             label = "{}N_{}S".format(node, stim)
#             title = "Stim Scaling"
#             figname = "StimScaling"
#         elif how == "sfs":
#             f  = "outputs/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(node,nCpu, default_pop, default_stims, sf, node,nCpu,default_pop, default_stims, sf)
# #             label = "{}N_{}SF".format(node, sf)
#             title = "Score Function Scaling"
#             figname = "SfScaling"
#         elif how == "full":
#             f  = "outputs/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(node,nCpu, pop, stim, sf, node,nCpu,pop, stim, sf)
# #             label = "{}N_{}SF".format(node, sf)
#             title = "Score Function Scaling"
#             figname = "SfScaling"
#         else:
#             raise NotImplementedError
        label = "{}N".format(node)
        logRes = processLog(f)
        runtime = np.mean(logRes['runtimes'])
        runtimes.append(runtime)
        labels.append(label)
    plt.scatter(np.arange(len(runtimes)), runtimes)
    plt.plot(np.arange(len(runtimes)), runtimes)
    plt.xticks(ticks=np.arange(len(runtimes)), labels=labels, rotation=45)
    plt.ylabel("total runtime (s)")
    plt.title(title)
    plt.savefig(os.path.join("outputs", figname),  bbox_inches='tight')
    
    

        
        
def compare_scaling(strong_df, weak_df):
    #f  = "runTimeLogs/runTime.log"
    #assert (strong_df['offspring'].values == weak_df['offspring'].values).all()
    fig = plt.figure()
    plt.title("Population Scaling Comparison")
    labels = strong_df['offspring'].values
    y = strong_df['runtime (mean)'].values
    err = strong_df['runtime stddev'].values
    plt.plot(labels, y, color='blue', label="strong scaling")
    plt.fill_between(labels, y - err, y+ err, color='blue', alpha=.4)
    # revisit this line
    labels = weak_df['offspring'].values
    y = weak_df['runtime (mean)'].values
    err = weak_df['runtime stddev'].values
    plt.plot(labels, y, color='red', label="weak scaling")
    plt.fill_between(labels, y - err, y+ err, color='red', alpha=.4)
    
    plt.ylabel("time (s)")
    plt.xlabel("pop size")
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig(os.path.join("outputs", "scaling_time_compare"),  bbox_inches='tight')
    plt.close(fig)
    
    fig2 = plt.figure()
    plt.title("FOM comparison where FOM = pop size / nGPUs / runtime ")
    labels = strong_df['offspring'].values
    y = strong_df['cori fom'].values
    plt.plot(labels, y, color='blue', label="strong scaling")
    plt.ylim(bottom=0)

    # revisit this line
    labels = weak_df['offspring'].values
    y = weak_df['cori fom'].values
    err = weak_df['runtime stddev'].values
    plt.plot(labels, y, color='red', label="weak scaling")
    plt.ylabel("FOM")
    plt.xlabel("pop size")
    plt.legend()
    plt.savefig(os.path.join("outputs", "scaling_fom_compare"),  bbox_inches='tight')
    plt.close(fig)
    
    
def compare_stim_scaling(strong_df, weak_df):
    #f  = "runTimeLogs/runTime.log"
    strong_df = strong_df[strong_df['score functions'] == 20.0]
    strong_df = strong_df.sort_values(by='stims')
    fig = plt.figure()
    plt.title("Stim Scaling Comparison")
    labels = strong_df['stims'].values
    y = strong_df['runtime (mean)'].values
    err = strong_df['runtime stddev'].values
    plt.plot(labels, y, color='blue', label="strong scaling")
    plt.fill_between(labels, y - err, y+ err, color='blue', alpha=.4)
    
    
    # MONKEY PATCH
    weak_df =weak_df[~(weak_df['nodes'] >  weak_df['stims'])]
    
    y = weak_df['runtime (mean)'].values
    err = weak_df['runtime stddev'].values
    import pdb; pdb.set_trace()
    plt.plot(labels, y, color='red', label="weak scaling")
    plt.fill_between(labels, y - err, y+ err, color='red', alpha=.4)
    
    plt.ylabel("time (s)")
    plt.xlabel("number of stims")
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig(os.path.join("outputs", "stim_scaling_time_compare"),  bbox_inches='tight')
    plt.close(fig)

    
def compare_sf_scaling(strong_df, weak_df):
    #f  = "runTimeLogs/runTime.log"
    import pdb; pdb.set_trace()
    strong_df = strong_df[strong_df['stims'] == 1.0]
    strong_df = strong_df[strong_df['offspring'] == 500.0]
    strong_df = strong_df[strong_df['score functions'] < 71]
    weak_df = weak_df[~((weak_df['score functions'] == 20.0) & (weak_df['nodes'] == 1.0))]
    strong_df = strong_df.sort_values(by='score functions')
    fig = plt.figure()
    plt.title("Score Function Scaling Comparison")
    labels = strong_df['score functions'].values
    y = strong_df['runtime (mean)'].values
    err = strong_df['runtime stddev'].values
    plt.plot(labels, y, color='blue', label="strong scaling")
    plt.fill_between(labels, y - err, y+ err, color='blue', alpha=.4)
    
    y = weak_df['runtime (mean)'].values
    err = weak_df['runtime stddev'].values
    plt.plot(labels, y, color='red', label="weak scaling")
    plt.fill_between(labels, y - err, y+ err, color='red', alpha=.4)
    
    plt.ylabel("time (s)")
    plt.xlabel("# of score functions")
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig(os.path.join("outputs", "sf_scaling_time_compare"),  bbox_inches='tight')
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
        or (curr_pop not in constraints['offspring']) or (curr_sfs not in constraints['n_sfs'])):
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
        
    
def wrapProfileMaker(nCpus,nodes,pops, versions):
    for idx,(node,pop,nCpu,vers) in enumerate(zip(nodes,pops,nCpus,versions)):
        makeCustomProfile(node,nCpu,pop,vers)
          
def generate_result_table(nCpus,nodes,pops, sfs, stims, versions, how='vanilla'):
    df = pd.DataFrame(columns=['nodes','total cpu', 'total gpu',\
                               'offspring', 'stims', 'score functions', \
                               'runtime (mean)', 'runtime stddev', 'cori fom'])
    for idx,(node,pop,nCpu,stim, sf, vers) in enumerate(zip(nodes,pops,nCpus, stims, sfs, versions)):
        if 'vanilla' in how:
            f  = "outputs/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(node,nCpu, pop, default_stims, default_sfs, node,nCpu, pop, default_stims, default_sfs)
            label = "{}N_{}O".format(node, pop)
            figname = "pop_scale.tex"
            df_name = 'pop_scale.csv'
        elif 'stims' in how:
            f  = "outputs/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(node,nCpu, default_pop, stim, default_sfs, node,nCpu,default_pop, stim, default_sfs)
            label = "{}N_{}S".format(node, stim)
            figname = "stim_scale.tex"
            df_name = "stim_scale.csv"
        elif "sfs" in how:
            f  = "outputs/{}N_{}C_{}O_{}S_{}SF/{}N_{}C_{}O_{}S_{}SF.log".format(node,nCpu, default_pop, default_stims, sf, node,nCpu,default_pop, default_stims, sf)
            label = "{}N_{}SF".format(node, sf)
            figname = "sf_scale.tex"
            df_name = "sf_scale.csv"
        else:
            raise NotImplementedError

        logRes = processLog(f)
        mean_runtime, std_runtime = np.mean(logRes['runtimes']), np.std(logRes['runtimes'])
        mean_eval, std_eval = np.mean(logRes['evalTimes']), np.std(logRes['evalTimes'])
        mean_neuroGPU, std_neuroGPU =  np.mean(logRes['neuroGPUTimes']), np.std(logRes['neuroGPUTimes'])
        # TODO: ACTUALLY READ NGPUS FROM LOG -- HERE WE ARE ASSUMING ITS 6
        if np.isnan(mean_runtime):
            continue
        df.loc[idx] = [int(node),int(nCpu)*int(node), 6*int(node), \
                     int(pop),int(stim), int(sf), float(mean_runtime), float(std_runtime), int(pop)/(6*int(node))/float(mean_runtime)]
    df = df.sort_values('nodes', ascending=True)   
    formaters =  {"runtime (mean)": "{:0.2f}".format, "runtime stddev":  "{:0.2f}".format,   "cori fom" : "{:0.2f}".format }
    df.to_latex(os.path.join("outputs", figname), formatters=formaters, float_format="%.0f")
    df.to_csv(os.path.join("outputs", df_name))
        
    print("WARNING: assumed 6 gpus")
    return df

    
    
def write_all_files(dest, srcs):
    with open(dest, 'w') as outfile:
        for fname in srcs:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                
    
def merge_experiments(src, dest, version):
    src_path = os.path.join("outputs",src)
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
    shutil.copy(os.path.join(src_path,gpu_util_logname), os.path.join(dest,new_gpu_util_logname))
    # comebine regular log
    log_path = os.path.join(dest, "{}N_{}C_{}O_{}S_{}SF.log".format(curr_node, curr_core, curr_pop, curr_stims, curr_sfs, version))
    old_log = os.path.join(src_path, "{}N_{}C_{}O_{}S_{}SF_{}.log".format(curr_node, curr_core, curr_pop, curr_stims, curr_sfs, version))

    if os.path.isfile(log_path):
        write_all_files(log_path, [log_path, old_log])
    else:
        #assert int(version) == 1, "no master log and version is not 1?"
        write_all_files(log_path, [old_log])

    
def collapse_exps(exp_names):
    nodes = []
    pops = []
    nCpus = []
    stims = []
    sfs = []
    exp_names = sorted(exp_names)
    for exp_name in exp_names:
        try:
            curr_node, curr_core, curr_pop, curr_stims, curr_sfs, version = re.findall(r'\d+', exp_name) # TODO: use version appropriately
        except ValueError as e:
            print(exp_name, " doesn't confrom")
            continue
        
        nodes.append(curr_node), pops.append(curr_pop), nCpus.append(curr_core)
        sfs.append(curr_sfs), stims.append(curr_stims)
        agg_exp_path =  os.path.join("outputs","{}N_{}C_{}O_{}S_{}SF".format(curr_node, curr_core, curr_pop, curr_stims, curr_sfs))

        if int(version) == 0 or not os.path.isdir(agg_exp_path):
            if os.path.isdir(agg_exp_path):
                shutil.rmtree(agg_exp_path)
            shutil.copytree(os.path.join("outputs",exp_name), agg_exp_path)
            log_name = [file for file in os.listdir(agg_exp_path) if ".log" in file and "gpu" not in file]
            os.rename(os.path.join(agg_exp_path,log_name[0]), os.path.join(agg_exp_path,log_name[0][:-6] + log_name[0][-4:] ))
            print(log_name)
            
        else:
            merge_experiments(exp_name, agg_exp_path, version )
       
    return 
        
    
def sf_plot_strategy(exp_names, args, collapse=False):
    if collapse:
        collapse_exps(exp_names)
    nodes, pops,  nCpus, sfs, stims, versions = read_exps(exp_names, condition='sfs', args=args)
    plt.title("Population Size Scaling w. Nodes")
    # step 1
    plotScaling(nCpus,nodes,pops, sfs, stims, versions, how='sfs')
    # step 2
    #wrapProfileMaker(nCpus, nodes, pops, versions)
    
    # step 3
    plot_CPUGPU_bottleneck(nCpus,nodes,pops, sfs, stims, versions, how='sfs')

    # step 4
    generate_result_table(nCpus,nodes,pops, sfs, stims, versions, how='sfs')
    
def stim_plot_strategy(exp_names, args, collapse=False):
    
    if collapse:
        collapse_exps(exp_names)
    nodes, pops,  nCpus, sfs, stims, versions = read_exps(exp_names, condition='stims', args=args)
    plt.title("Population Size Scaling w. Nodes")
    # step 1
    how ='vanilla'
    if args.constraint_file:
        how = 'full'
    plotScaling(nCpus,nodes,pops, sfs, stims, versions, how=how)
    # step 2
    #wrapProfileMaker(nCpus, nodes, pops, versions)
    
    # step 3
    plot_CPUGPU_bottleneck(nCpus,nodes,pops, sfs, stims, versions, how=how)

    # step 4
    generate_result_table(nCpus,nodes,pops, sfs, stims, versions, how=how)
    
def vanilla_plot_strategy(exp_names, args, collapse=False):
    
#     if collapse:
#         collapse_exps(exp_names)
    nodes, pops,  nCpus, sfs, stims, versions = read_exps(exp_names, args=args)
    import pdb; pdb.set_trace()
    plt.title("Population Size Scaling w. Nodes")
    # step 1
    how ='vanilla'
    title = None
    if args.constraint_file:
        how = 'full'
        title =  args.constraint_file
    plotScaling(nCpus,nodes,pops, sfs, stims, versions, how=how, title=title)
    # step 2
    #wrapProfileMaker(nCpus, nodes, pops, versions)
    
    # step 3
    plot_CPUGPU_bottleneck(nCpus,nodes,pops, sfs, stims, versions, how='vanilla')

    # step 4
    generate_result_table(nCpus,nodes,pops, sfs, stims, versions, how='vanilla')
    
def strong_plot_strategy(exp_names, args, collapse=False):
    
    if collapse:
        collapse_exps(exp_names)
        
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
    strong_df = generate_result_table(nCpus,nodes,pops, sfs, stims, versions, how=how)
    weak_df = pd.read_csv(os.path.join("weak_outputs",weak_name))
    if not args.stims and not args.sfs:
        compare_scaling(strong_df, weak_df)
    elif args.stims:
        compare_stim_scaling(strong_df, weak_df)
    elif args.sfs:
        compare_sf_scaling(strong_df, weak_df)
        
    

    
if __name__ == "__main__":
    
    args = parser.parse_args()
    exp_names = [dirname for dirname in os.listdir("outputs") if "_" in dirname and "ipynb" not in dirname] # make this more strict later --> should match coresnodes_POPSIZE_iteration
    if args.stims:
        stim_plot_strategy(exp_names, args, collapse=True)
    elif args.sfs:
        sf_plot_strategy(exp_names, args, collapse=True)
    else:
        vanilla_plot_strategy(exp_names, args, collapse=True)
        
    if args.strong:
        strong_plot_strategy(exp_names, args, collapse=True)
    
        
    
    
    