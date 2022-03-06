import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cycler
import argparse
import matplotlib.ticker as mticker
from matplotlib.ticker import AutoLocator
from matplotlib.ticker import FuncFormatter
from matplotlib import colors
from collections import OrderedDict
import re
import os

save_mode = 'png'
# save_mode = 'pdf'

def scientific(x, pos):
    # x:  tick value - ie. what you currently see in yticks
    # pos: a position - ie. the index of the tick (from 0 to 9 in this example)
    return '%.0f' % x

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
    # plt.rc('axes', facecolor='white', edgecolor='black',
    #     axisbelow=True, grid=True)
    # plt.rc('grid', color='lightgrey', linestyle='solid')
    # plt.rc('patch', edgecolor='grey')
    plt.rc('lines', linewidth=2, color='black')
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams.update({'font.size': 18})


# XANDER TODO:
#     1. Figure out where missing summit scores went
#     2. Add num trials for CPU stuff
#     3. Make big tables using curr df and plot functions

def plot_compute_scales_problem_fixed(args):
    extension = "Compute Scales and Problem Fixed_scale.csv"
    plot_package = []
    all_labels = []
    all_runtimes = []
    all_names = []
    for exp_name in args.exp_names:   
        curr_df, color = find_experiment_df(exp_name, extension)
        runtimes  = curr_df['Runtime'].values
        stds = curr_df['Runtime Stddev'].values
        labels =  curr_df['Nodes'].values
        expected = runtimes[0] / labels/labels[0]
        # TODO: missing some trials for 2 ... fill these out
        # if 2 not in labels:
        #     print("WARNING : NEED TO FILL OUT TRIALS FOR EACH EXPERIMENT : ", exp_name)
        #     labels = np.insert(labels,1,2)
        #     stds = np.insert(stds,1,stds[0])
        #     runtimes = np.insert(runtimes,1,runtimes[0]/2)
        #     expected = np.insert(expected, 1, expected[0] / 2)

        all_labels.append(labels)
        [all_runtimes.append(runtime) for runtime in runtimes]
        all_names.append(exp_name)
        
        plot_package.append({'name': exp_name, 'runtimes': runtimes, 'stds': stds, 'labels': labels, \
                             'expected': expected, 'color': color})
 
    labels = np.unique(np.concatenate(all_labels))

    bench_name = 'Ideal'
    
    fig, axs = plot_runtime_v_expected(args, plot_package, labels, bench_name )
    plt.xlabel("Nodes")
    tcks = axs[0].get_yticks()
    # tcks[-1] = max(all_runtimes)
    tcks = tcks[2:-1]
    tcks[-1] = min(tcks[-1] , round(max(all_runtimes), -3) + 1000)
    axs[0].set_yticks(tcks)
    scientific_formatter = FuncFormatter(scientific)
    axs[0].yaxis.set_major_formatter(scientific_formatter)
    plt.savefig(f'figures/{os.path.basename(args.constraint_file)}_{all_names[0]}_{all_names[1]}.{save_mode}',  bbox_inches='tight')
    plt.close()

    
def plot_compute_fixed_problem_scales(args):
    extension = "Compute Fixed and Problem Scales_scale.csv"
    plot_package = []
    all_labels = []
    all_runtimes = []
    all_names = []
    for exp_name in args.exp_names:   
        curr_df, color = find_experiment_df(exp_name, extension)
        curr_df = curr_df[curr_df['Num Trials'] > 2]
        runtimes  = curr_df['Runtime'].values
        stds = curr_df['Runtime Stddev'].values
        labels =  curr_df['Offspring'].values
        

        # the mean gen size / offspring size to adjust for populations
        # that are not full size. This value is always VERY close to .8
        # hence the try / except
        try:
            expected = runtimes[0] * (labels/labels[0] * np.mean(curr_df['Mean Gen Size'] / curr_df['Offspring'].values))
            print(np.mean(curr_df['Mean Gen Size'] / curr_df['Offspring'].values))
        except:
            expected = runtimes[0] * (labels/labels[0] * .8)
        
        # if 'CPU' in exp_name:
        #     runtimes = np.insert(runtimes,3,expected[-4:] * .8)
        if not 'CPU' in exp_name:  
            all_labels.append(labels)
        [all_runtimes.append(runtime) for runtime in runtimes]
        all_names.append(exp_name)
        
        plot_package.append({'name': exp_name, 'runtimes': runtimes, 'stds': stds, 'labels': labels, \
                             'expected': expected, 'color':color})
 
    
    
        
    labels = np.unique(np.concatenate(all_labels))    # LABELS 1    
    bench_name = 'Exponential'
    
    fig, axs = plot_runtime_v_expected(args, plot_package, labels, bench_name )
    
    tcks = axs[0].get_yticks()
    #import pdb; pdb.set_trace()
    tcks = tcks[1:-1]
    tcks[-1] = min(tcks[-1],round(max(all_runtimes), -3) + 1000)
    axs[0].set_yticks(tcks)
    plt.xticks(ticks=np.arange(len(labels))[::2], labels=labels[::2].astype(int), rotation=45)
    scientific_formatter = FuncFormatter(scientific)
    axs[0].yaxis.set_major_formatter(scientific_formatter)
    plt.xlabel("Population")
    plt.savefig(f'figures/{os.path.basename(args.constraint_file)}_{all_names[0]}_{all_names[1]}.{save_mode}',  bbox_inches='tight')
    plt.close()
    
    
    fig, axs = plot_runtime_ratios(args, plot_package, labels, bench_name )    
    tcks = axs[0].get_yticks()
#     tcks[-1] = max(all_runtimes)
    axs[0].set_yticks(tcks)
    scientific_formatter = FuncFormatter(scientific)
    axs[0].yaxis.set_major_formatter(scientific_formatter)
    plt.xlabel("Population")
    plt.savefig(f'figures/{os.path.basename(args.constraint_file)}_{all_names[0]}_{all_names[1]}_comparision.{save_mode}',  bbox_inches='tight')
    plt.close()

def plot_compute_scales_problem_scales(args):
    extension = "Compute Scales and Problem Scales_scale.csv"
    
    plot_package = []
    all_labels = []
    all_runtimes = []
    all_names = []
    for exp_name in args.exp_names:   
        curr_df, color = find_experiment_df(exp_name, extension)
        runtimes  = curr_df['Runtime'].values
        stds = curr_df['Runtime Stddev'].values
        labels =  curr_df['Nodes'].values
        expected = np.repeat(runtimes[0],len(runtimes)) 
        
        all_labels.append(labels)
        [all_runtimes.append(runtime) for runtime in runtimes]
        all_names.append(exp_name)
        
        plot_package.append({'name': exp_name, 'runtimes': runtimes, 'stds': stds, 'labels': labels, \
                             'expected': expected, 'color': color})
 
    
    
        
    labels = np.unique(np.concatenate(all_labels))    # LABELS 1    

    bench_name = 'Ideal'
   
    
    fig, axs = plot_runtime_v_expected(args, plot_package, labels, bench_name )
    
    plt.ylim(bottom=10)
    plt.xlabel("Nodes")

#     plt.title('Compute Scales and Problem Scales')
#     [ ax.yaxis.set_minor_formatter(mticker.ScalarFormatter()) for ax in axs]
    tcks = axs[0].get_yticks()
    if "eFEL" in args.exp_names:
        tcks = [10,12,15,18,20]   
    elif "Summit" in args.exp_names:
        tcks = [10,20,30,50,60]
    else:
         tcks = tcks[1:-1]
        
        
    axs[0].set_yticks(tcks)
    
    scientific_formatter = FuncFormatter(scientific)
    axs[0].yaxis.set_major_formatter(scientific_formatter)
    if "eFEL" in args.exp_names or "Summit" in args.exp_names:
        if "eFEL" in args.exp_names:
            axs[0].set_ylim(10,20)
        axs[0].minorticks_off()
        

    plt.savefig(f'figures/{os.path.basename(args.constraint_file)}_{all_names[0]}_{all_names[1]}.{save_mode}',  bbox_inches='tight')
    plt.close()
    
    
def find_experiment_df(exp, extension):
    if 'NeuroGPU-EA' in exp and '8 SF' not in exp:
        df = pd.read_csv(f'GPU_genetic_alg/python/regular_outputs/{extension}')  
        color = 'green'
    elif 'CPU-EA' in exp:
        df = pd.read_csv(f'neuron_genetic_alg/outputs/{extension}')
        color = 'blue'
    elif 'Summit' in exp:
        df = pd.read_csv(f'GPU_genetic_alg/python/summit_outputs/{extension}') 
        color = 'darkorange'
    elif 'IPFX' in exp:
        df = pd.read_csv(f'GPU_genetic_alg/python/ipfx_outputs/{extension}')
        color = 'brown'
    elif '6 GPU' in exp:
        df = pd.read_csv(f'GPU_genetic_alg/python/6GPU_outputs/{extension}')
        color = 'green'
    elif 'eFEL' in exp:
        df = pd.read_csv(f'GPU_genetic_alg/python/8SF_outputs/{extension}')
        color = 'green'
    elif 'CoreNeuronGPU-EA' in exp:
        df = pd.read_csv(f'core_neuron_genetic_alg/python/outputs/{extension}')
        color = 'purple'
    if not 'CPU' in exp:
        try:
            redo_df = df[df['Num Trials'] < 48]
        except:
            import pdb; pdb.set_trace()
        # if len(redo_df) > 0 and not 'Summit' in exp:
        #     print(f" {exp} / {extension} redo nodes: {redo_df['Nodes'].values} offspring {redo_df['Offspring'].values}")
    df = df.drop_duplicates(subset=['Nodes','Runtime']) # TODO: add more keys
    df = df.rename({'Trials':'Num Trials'}, axis=1)
    if len(df[df['Num Trials'] < 49]) > 0 and exp != 'Summit':
        print("not enough trials for : ", exp, "@", extension)
        print(df[df['Num Trials'] < 49][['Num Trials', 'Offspring', 'Nodes']])
    return df, color


def assemble_experiment_tables():
    experiment_extensions = ["Compute Scales and Problem Scales_scale.csv", "Compute Fixed and Problem Scales_scale.csv", "Compute Scales and Problem Fixed_scale.csv"]
    path = 'figures/tables'
    for extension in experiment_extensions:
        gpu_df = pd.read_csv(f'GPU_genetic_alg/python/regular_outputs/{extension}')   
        neuron_df = pd.read_csv(f'neuron_genetic_alg/outputs/{extension}')
        summit_df = pd.read_csv(f'GPU_genetic_alg/python/summit_outputs/{extension}')   
        ipfx_df = pd.read_csv(f'GPU_genetic_alg/python/ipfx_outputs/{extension}')
        gpu_df_6 = pd.read_csv(f'GPU_genetic_alg/python/6GPU_outputs/{extension}')
        gpu_df_8 = pd.read_csv(f'GPU_genetic_alg/python/8SF_outputs/{extension}')
        
        # main_df = gpu_df.join(neuron_df,  lsuffix=' GPU Experiment', rsuffix=' CPU Experiment')

        
        main_df = gpu_df.drop('Unnamed: 0', axis=1)
        main_df = main_df.iloc[: , :6]
#         main_df = main_df.rename({'Total Cpus': 'Total CPUS GPU exp.' ,  'Offspring': 'Offspring GPU exp.'})
#         main_df['Total CPU CPU Exp.'] = [8,8,8,8,8]
#         main_df['Stimuli Scaling Exp.'] = [1,6,12,18, '-']
#         main_df['Score Function Exp.'] = [8,8,8,8,8]
        col_fmt = "|".join(np.repeat('c', len(main_df.columns)))
        col_fmt = "|" + col_fmt + "|"
        figname = extension.replace('.csv','')
        main_df.to_latex(os.path.join(path, figname), float_format="%.0f", index=False, column_format=col_fmt)
                                           
                                 
        
#         core_neuron_df = pd.read_csv(f'core_neuron_genetic_alg/python/outputs/{extension}')
        

    
    
def parse_df(df, id_col):
    runtimes  = df['Runtime'].values
    stds = df['Runtime Stddev'].values
    labels = df[id_col].values
    return runtimes, stds, labels
    
    

    
def plot_runtime_v_expected(args, plot_package, labels, bench_name, yscale=True):
    fig = plt.figure()  
    main_colors = ['red','blue','green','black']
    labels = [int(label) for label in labels]
    alt_colors = ['darkgoldenrod' ,'purple', 'darkcyan', 'grey']
    for exp_idx, exp_info in enumerate(plot_package):
        

        curr_name, curr_runtimes, curr_std = exp_info['name'], exp_info['runtimes'], exp_info['stds']
        if '6' not in curr_name: # if we aren't comparing against summit, remove Cori from name
            curr_name = re.sub('Cori','',curr_name)
        curr_expected, curr_labels = exp_info['expected'], exp_info['labels']
        plt.scatter(np.arange(len(curr_runtimes)), curr_runtimes, color=exp_info['color'], s=15)
        plt.plot(np.arange(len(curr_runtimes)), curr_runtimes,  color=exp_info['color'], label=f"{curr_name} Observed")
        plt.fill_between(np.arange(len(curr_runtimes)), curr_runtimes-curr_std, curr_runtimes+curr_std, alpha=.25, color=exp_info['color'])

        plt.plot(np.arange(len(curr_expected)), curr_expected,  color=exp_info['color'], linestyle='dashed', marker='o', label=f'{curr_name}'  + ' Expected', alpha=.6)

        plt.scatter(np.arange(len(curr_expected)), curr_expected, color=exp_info['color'])#label=f'{curr_name} ' + bench_name, s=15)
    


    axs = fig.axes
    # log scale y ticks
    if yscale:
        plt.yscale("log") 
#         plt.xscale('log')
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45)
    if  'Compute Scales and Problem Scales' in args.constraint_file:
        leg = plt.legend(fontsize=16, loc=(1.05,.5))
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=16, loc=(1.05,.5))

    return fig, axs


def plot_runtime_ratios(args, plot_package, labels, bench_name, yscale=False):
    fig = plt.figure()  
    main_colors = ['red','blue','green','black']
    labels = [int(label) for label in labels]
    alt_colors = ['darkgoldenrod' ,'purple', 'darkcyan', 'grey']
    seen = []
    for exp_idx, exp_info in enumerate(plot_package):
        for exp_idx_compare, exp_info_compare in enumerate(plot_package):
            
            min_len = min(len(exp_info['runtimes']), len(exp_info_compare['runtimes']))
            curr_name, curr_runtimes, curr_std = exp_info['name'], exp_info['runtimes'][:min_len], exp_info['stds'][:min_len]
            curr_expected, curr_labels = exp_info['expected'][:min_len], exp_info['labels'][:min_len]
            
            curr_name2, curr_runtimes2, curr_std2 = exp_info_compare['name'], exp_info_compare['runtimes'][:min_len], exp_info_compare['stds'][:min_len]
            curr_expected2, curr_labels2 = exp_info_compare['expected'][:min_len], exp_info_compare['labels'][:min_len]
            
            # skip common experiments or ones where rataio is < 1
            if curr_name == curr_name2 or curr_runtimes2[0] < curr_runtimes[0]:
                continue

            color = [(c1 + c2) / 2 for c1, c2 in zip(colors.to_rgba(main_colors[exp_idx]) , colors.to_rgba(main_colors[exp_idx_compare]))]
            
            if color in seen:
                continue
            else:
                seen.append(color)
            
            plt.scatter(np.arange(len(curr_runtimes)), curr_runtimes2 / curr_runtimes, color=color, label=f"{curr_name}/{curr_name2} Observed", s=15)
            plt.plot(np.arange(len(curr_runtimes)),  curr_runtimes2 / curr_runtimes,  color=color)
#             plt.fill_between(np.arange(len(curr_runtimes)), curr_runtimes-curr_std, curr_runtimes+curr_std, alpha=.25, color=main_colors[exp_idx])

#             plt.plot(np.arange(len(curr_expected)), curr_expected,  color=alt_colors[exp_idx], linestyle='dashed', marker='o')

#             plt.scatter(np.arange(len(curr_expected)), curr_expected, color=alt_colors[exp_idx], label=f'{curr_name} ' + bench_name, s=15)


    axs = fig.axes
    # log scale y ticks
    if yscale:
        plt.yscale("log") 
#         plt.xscale('log')
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45)
    plt.legend(fontsize=16, loc=(1.05,.5))

    return fig, axs





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Benchmarking viz2')
    parser.add_argument('--constraint_file', type=str, required=True, default=None)
    parser.add_argument('-n', '--exp_names', nargs='+', default=[])



    args = parser.parse_args()
    set_custom_params_plt()

    if 'Compute Scales and Problem Fixed' in args.constraint_file :
        plot_compute_scales_problem_fixed(args)
    elif  'Compute Fixed and Problem Scales' in  args.constraint_file:
        plot_compute_fixed_problem_scales(args)
    elif 'Compute Scales and Problem Scales' in args.constraint_file:
        plot_compute_scales_problem_scales(args)
    elif 'make_tables' in args.constraint_file:
        assemble_experiment_tables()
    else:
        assert False, "Arg for constraint file not recognized"