import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cycler
import argparse
import matplotlib.ticker as mticker
from matplotlib.ticker import AutoLocator
from matplotlib.ticker import FuncFormatter

import os

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
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
        axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('patch', edgecolor='#E6E6E6')
    plt.rc('lines', linewidth=2)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']



def plot_compute_scales_problem_fixed(args):
    extension = "Compute Scales and Problem Fixed_scale.csv"
    df1 = find_experiment_df(args.exp1, extension)
    df2 = find_experiment_df(args.exp2, extension)
    
    runtimes1  = df1['Runtime'].values
    runtimes2 = df2['Runtime'].values
    stds1 = df1['Runtime Stddev'].values
    stds2 = df2['Runtime Stddev'].values
    labels1 = df1['Nodes'].values
    labels2 =  df2['Nodes'].values
    
    bench_name = 'Ideal'
    expected2 = runtimes2[0] * labels2/labels2[0]
    expected1 = runtimes1[0] * labels1/labels1[0]
    
    labels = np.unique(np.concatenate([labels1, labels2]))
    # LABELS 1
    labels = labels1
    #NOTE: here is where I calculate expected line
#     other_labels  = np.insert(other_labels,1,2)
#     other_labels  = np.insert(other_labels,3,8)
#     cori_labels = np.insert(cori_labels,1,2)
#     labels = np.unique(np.concatenate([cori_labels, other_labels]))
#     labels = cori_labels

    expected2 = runtimes2[0] / (labels2 /labels2[0])
    expected1 = runtimes1[0] / (labels1/labels1[0])#[cori_runtimes[0] * (i+1) for i in 

#     other_stds = np.insert(other_stds,2,np.nan)

    bench_name = 'Ideal'
    
    fig, axs = plot_runtime_v_expected(args, runtimes1, expected1, stds1, expected2, runtimes2, stds2, bench_name, labels, other=args.use_other, first_name=args.exp1, other_name=args.exp2)
    plt.xlabel("Nodes")

    tcks = axs[0].get_yticks()
    tcks[-1] = max(max(runtimes1), max(runtimes2))
    axs[0].set_yticks(tcks)
    scientific_formatter = FuncFormatter(scientific)
    axs[0].yaxis.set_major_formatter(scientific_formatter)
    plt.savefig(f'figures/{os.path.basename(args.constraint_file)}_{args.exp1}_{args.exp2}.png',  bbox_inches='tight')
    plt.close()

    
def plot_compute_fixed_problem_scales(args):
    
    extension = "Compute Fixed and Problem Scales_scale.csv"
    df1 = find_experiment_df(args.exp1, extension)
    df2 = find_experiment_df(args.exp2, extension)
    
    runtimes1  = df1['Runtime'].values
    runtimes2 = df2['Runtime'].values
    stds1 = df1['Runtime Stddev'].values
    stds2 = df2['Runtime Stddev'].values
    labels1 = df1['Offspring'].values
    labels2 =  df2['Offspring'].values
    
    
    expected2 = runtimes2[0] * labels2/labels2[0]
    expected1 = runtimes1[0] * labels1/labels1[0]
    
    labels = np.unique(np.concatenate([labels1, labels2]))
    # LABELS 1
    labels = labels1
    
    bench_name = 'Exponential'
    
    fig, axs = plot_runtime_v_expected(args, runtimes1, expected1, stds1, expected2, runtimes2, stds2, bench_name, labels, other=args.use_other, first_name=args.exp1, other_name=args.exp2,yscale=True)
    
    tcks = axs[0].get_yticks()
    tcks[-1] = max(max(runtimes1), max(runtimes2))
    axs[0].set_yticks(tcks)
    scientific_formatter = FuncFormatter(scientific)
    axs[0].yaxis.set_major_formatter(scientific_formatter)
    
    plt.xlabel("Population")
    plt.savefig(f'figures/{os.path.basename(args.constraint_file)}_{args.exp1}_{args.exp2}.png',  bbox_inches='tight')
    plt.close()

    
    
def find_experiment_df(exp, extension):
    if 'Cori GPU' in exp:
        df = pd.read_csv(f'GPU_genetic_alg/python/regular_outputs/{extension}')   
    elif 'CPU' in exp:
        df = pd.read_csv(f'neuron_genetic_alg/outputs/{extension}')
    elif 'Summit' in exp:
        df = pd.read_csv(f'GPU_genetic_alg/python/summit_outputs/{extension}')   
    elif 'IPFX' in exp:
        df = pd.read_csv(f'GPU_genetic_alg/python/ipfx_outputs/{extension}')
    elif '6 GPU' in exp:
        df = pd.read_csv(f'GPU_genetic_alg/python/6GPU_outputs/{extension}')
    elif 'Cori GPU 8 SF' in exp:
        df = pd.read_csv(f'GPU_genetic_alg/python/8SF_outputs/{extension}')
    elif 'CoreNeuron' in exp:
        df = pd.read_csv(f'core_neuron_genetic_alg/python/outputs/{extension}')
    
    return df


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
    
    
def plot_compute_scales_problem_scales(args):
    extension = "Compute Scales and Problem Scales_scale.csv"
    df1 = find_experiment_df(args.exp1, extension)
    df2 = find_experiment_df(args.exp2, extension)

    runtimes1, stds1, labels1 = parse_df(df1, 'Nodes')
    runtimes2, stds2, labels2 = parse_df(df2, 'Nodes')
    
    expected2 = np.repeat(runtimes2[0],len(runtimes2)) 
    expected1 = np.repeat(runtimes1[0],len(runtimes1)) 

    bench_name = 'Ideal'
   
  
    labels = np.unique(np.concatenate([labels1, labels2]))
    # ?????
    labels = labels1
    
    fig, axs = plot_runtime_v_expected(args, runtimes1, expected1, stds1, expected2, runtimes2, stds2, bench_name, labels, other=args.use_other, first_name=args.exp1, other_name=args.exp2)
    
    plt.ylim(bottom=10)
    plt.xlabel("Nodes")

    plt.title('Compute Scales and Problem Scales')
    [ ax.yaxis.set_minor_formatter(mticker.ScalarFormatter()) for ax in axs]

    plt.savefig(f'figures/{os.path.basename(args.constraint_file)}_{args.exp1}_{args.exp2}.png',  bbox_inches='tight')
    plt.close()
    
def plot_runtime_v_expected(args, cori_runtimes, cori_expected, cori_stds, other_expected, other_runtimes, other_stds, bench_name, labels, other=False, yscale=True, first_name='first name', other_name='other'):
    fig = plt.figure()  
    # plot cori stuff
    plt.scatter(np.arange(len(cori_runtimes)), cori_runtimes, color='red', label=f"{first_name} Observed", s=15)
    plt.plot(np.arange(len(cori_runtimes)), cori_runtimes,  color='red')
    plt.fill_between(np.arange(len(cori_runtimes)), cori_runtimes-cori_stds, cori_runtimes+cori_stds, alpha=.25, color='red')
    
    plt.plot(np.arange(len(cori_expected)), cori_expected,  color='darkgoldenrod', linestyle='dashed', marker='o')
    
    plt.scatter(np.arange(len(cori_expected)), cori_expected, color='darkgoldenrod', label=f'{first_name} ' + bench_name, s=15)
    
    # plot other stuff
    if other:
        plt.scatter(np.arange(len(other_runtimes)), other_runtimes, color='blue', label=f"{other_name} Observed", s=15)
        plt.plot(np.arange(len(other_runtimes)), other_runtimes,  color='blue')
        plt.fill_between(np.arange(len(other_runtimes)), other_runtimes-other_stds, other_runtimes+other_stds, alpha=.25, color='blue')
        
        plt.plot(np.arange(len(other_expected)), other_expected,  color='purple', linestyle='dashed', marker='o',alpha=.5)
        
        plt.scatter(np.arange(len(other_expected)), other_expected, color='purple', label=f'{other_name} ' + bench_name, s=15, alpha=.8)
            
         
        # add scaling factor annotations
        if other_name == 'Cori CPU' and "Fixed" not in args.constraint_file:
            x = .1
            y = other_runtimes[0] - 10
            plt.annotate("Scaling Factor: 64 Indv. / Node",(x,y), fontsize=12)
            
            x = .1
            y = cori_runtimes[0] + 3
            plt.annotate("Scaling Factor: 250 Indv. / Node",(x,y), fontsize=12)

    axs = fig.axes
    # log scale y ticks
    if yscale:
        plt.yscale("log") 
#         plt.xscale('log')
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45)
    plt.ylabel("Total Runtime (s)")
    # note that command line arg is used as name
#     plt.title(args.constraint_file)

    plt.legend()
    return fig, axs





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Benchmarking viz2')
    parser.add_argument('--constraint_file', type=str, required=True, default=None)
    parser.add_argument('--exp1', type=str, required=False, default='Cori CPU')
    parser.add_argument('--exp2', type=str, required=False, default='Cori CPU')

    parser.add_argument('--use_other', type=bool, required=False, default=True)


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