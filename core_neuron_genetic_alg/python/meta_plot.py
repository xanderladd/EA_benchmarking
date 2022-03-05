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



def plot_compute_scales_problem_fixed(args):
    cori_df = pd.read_csv("outputs/Compute Scales and Problem Fixed_scale.csv")
    summit_df = pd.read_csv('summit_outputs/Compute Scales and Problem Fixed_scale.csv')
    cori_runtimes  = cori_df['Runtime'].values
    summit_runtimes = summit_df['Runtime'].values
    cori_labels = cori_df['Nodes'].values
    summit_labels =  summit_df['Nodes'].values
    #NOTE: here is where I calculate expected line
#     summit_labels  = np.insert(summit_labels,1,2)
#     summit_labels  = np.insert(summit_labels,3,8)
#     cori_labels = np.insert(cori_labels,1,2)
    labels = np.unique(np.concatenate([cori_labels, summit_labels]))
    
    labels = cori_labels
    
    summit_expected = summit_runtimes[0] / (summit_labels /summit_labels[0])#[summit_runtimes[0] * (i+1) for i in range(len(summit_runtimes))]
    cori_expected = cori_runtimes[0] / (cori_labels/cori_labels[0])#[cori_runtimes[0] * (i+1) for i in 


    cori_stds = cori_df['Runtime Stddev'].values
    
    


    summit_stds = summit_df['Runtime Stddev'].values
#     summit_stds = np.insert(summit_stds,2,np.nan)

    bench_name = 'Ideal'
    
    fig, axs = plot_runtime_v_expected(args, cori_runtimes, cori_expected, cori_stds, summit_expected, summit_runtimes, summit_stds, bench_name, labels)
    
    plt.xlabel("Nodes")
    
#     ax.yaxis.set_major_locator(AutoLocator())  # solution
#     ax.set_yticks([0,30,60,90,120])
#     ax.set_ylim(0,120)
    """
    https://stackoverflow.com/questions/46498157/overlapping-axis-tick-labels-in-logarithmic-plots
    """

    for ax in axs:
        ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
#         ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())

#         ax.yaxis.set_visible('False')
#         ax.minorticks_off() 
        for label in ax.yaxis.get_ticklabels()[::2]:#[::2]:
            label.set_visible(True)   
    plt.savefig(f'{args.constraint_file}.png',  bbox_inches='tight')
    

    
def plot_compute_fixed_problem_scales(args):
    cori_df = pd.read_csv("outputs/Compute Fixed and Problem Scales_scale.csv")
    cori_df = cori_df[cori_df['Offspring'] < 5000]
    summit_df = pd.read_csv('summit_outputs/Compute Fixed and Problem Scales_scale.csv')
    cori_runtimes  = cori_df['Runtime'].values
    summit_runtimes = summit_df['Runtime'].values
    cori_labels = cori_df['Offspring'].values
    summit_labels =  summit_df['Offspring'].values

    #NOTE: here is where I calculate expected line
    summit_expected = summit_runtimes[0] * summit_labels/summit_labels[0]#[summit_runtimes[0] * (i+1) for i in range(len(summit_runtimes))]
    cori_expected = cori_runtimes[0] * cori_labels/cori_labels[0]#[cori_runtimes[0] * (i+1) for i in range(len(cori_runtimes))]
    cori_stds = cori_df['Runtime Stddev'].values
    summit_stds = summit_df['Runtime Stddev'].values
    
    labels = np.unique(np.concatenate([cori_labels, summit_labels]))
    
    labels = cori_labels
    
    bench_name = 'Exponential'
    
    fig, axs = plot_runtime_v_expected(args, cori_runtimes, cori_expected, cori_stds, summit_expected, summit_runtimes, summit_stds, bench_name, labels)
    [ ax.yaxis.set_minor_formatter(mticker.ScalarFormatter()) for ax in axs]
    plt.xlabel("Population")
    
    plt.savefig(f'{args.constraint_file}.png',  bbox_inches='tight')

    
def plot_compute_scales_problem_scales(args):

    cori_df = pd.read_csv("outputs/Compute Scales and Problem Scales_scale.csv")
    summit_df = pd.read_csv('summit_outputs/Compute Scales and Problem Scales_scale.csv')
    cori_runtimes  = cori_df['Runtime'].values
    summit_runtimes = summit_df['Runtime'].values
    cori_stds = cori_df['Runtime Stddev'].values
    summit_stds = summit_df['Runtime Stddev'].values
    bench_name = 'Ideal'
    summit_expected = np.repeat(summit_runtimes[0],len(summit_runtimes)) 
    cori_expected = np.repeat(cori_runtimes[0],len(cori_runtimes)) 
    cori_labels = cori_df['Nodes'].values
    summit_labels =  summit_df['Nodes'].values
    
    labels = np.unique(np.concatenate([cori_labels, summit_labels]))
    
    labels = cori_labels
    
    
    fig, axs = plot_runtime_v_expected(args, cori_runtimes, cori_expected, cori_stds, summit_expected, summit_runtimes, summit_stds, bench_name, labels)
    
    plt.ylim(bottom=10)
    plt.xlabel("Nodes")
    [ ax.yaxis.set_minor_formatter(mticker.ScalarFormatter()) for ax in axs]
    [ ax.xaxis.set_minor_formatter(mticker.ScalarFormatter()) for ax in axs]

    plt.ylim(top=25)
    plt.title('Compute / Problem Ratio Scales at 250 Population / Node')
    plt.savefig(f'{args.constraint_file}.png',  bbox_inches='tight')
    
def plot_runtime_v_expected(args, cori_runtimes, cori_expected, cori_stds, summit_expected, summit_runtimes, summit_stds, bench_name, labels, summit=False, yscale=True):
    fig = plt.figure()  
    # plot cori stuff
    plt.scatter(np.arange(len(cori_runtimes)), cori_runtimes, color='red', label="Cori Observed", s=15)
    plt.plot(np.arange(len(cori_expected)), cori_expected,  color='orange')
    plt.fill_between(np.arange(len(cori_runtimes)), cori_runtimes-cori_stds, cori_runtimes+cori_stds, alpha=.35, color='red')
    plt.scatter(np.arange(len(cori_expected)), cori_expected, color='orange', label='Cori ' + bench_name, s=15)
    plt.plot(np.arange(len(cori_runtimes)), cori_runtimes,  color='red')
    # plot summit stuff
    if summit:
        plt.scatter(np.arange(len(summit_runtimes)), summit_runtimes, color='blue', label="Summit Observed", s=15)
        plt.plot(np.arange(len(summit_expected)), summit_expected,  color='purple')
        plt.fill_between(np.arange(len(summit_runtimes)), summit_runtimes-summit_stds, summit_runtimes+summit_stds, alpha=.35, color='blue')
        plt.scatter(np.arange(len(summit_expected)), summit_expected, color='purple', label='Summit ' + bench_name, s=15)
        plt.plot(np.arange(len(summit_runtimes)), summit_runtimes,  color='blue')
    axs = fig.axes
    # log scale y ticks
    if yscale:
        plt.yscale("log") 
#         plt.xscale('log')
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45)
    plt.ylabel("Total Runtime (s)")
    # note that command line arg is used as name
    plt.title(args.constraint_file)
    plt.legend()
    return fig, axs












def plot_compute_scales_problem_scales_GPUVS(args):

    cori_df = pd.read_csv("outputs/Compute Scales and Problem Scales_scale8gpu.csv")
    summit_df = pd.read_csv('outputs/Compute Scales and Problem Scales_scale.csv')
    cori_runtimes  = cori_df['Runtime'].values
    summit_runtimes = summit_df['Runtime Stddev'].values
    cori_stds = cori_df['Runtime Stddev'].values
    summit_stds = summit_df['Runtime Stddev'].values
    bench_name = 'Ideal'
    summit_expected = np.repeat(summit_runtimes[0],len(summit_runtimes)) 
    cori_expected = np.repeat(cori_runtimes[0],len(cori_runtimes)) 
    cori_labels = cori_df['Total Gpus'].values
    summit_labels =  summit_df['Total Gpus'].values
    cori_labels = cori_df['Nodes'].values
    summit_labels =  summit_df['Nodes'].values
    
    labels = np.unique(np.concatenate([cori_labels, summit_labels]))
    fig, axs = plot_runtime_v_expected_gpu(args, cori_runtimes, cori_expected, cori_stds, summit_expected, summit_runtimes, summit_stds, bench_name, summit_labels,  cori_labels)
    
    plt.ylim(bottom=10)
    plt.xlabel("Total # GPUs")
    [ ax.yaxis.set_minor_formatter(mticker.ScalarFormatter()) for ax in axs]
    plt.ylim(top=100)
    plt.savefig(f'{args.constraint_file}.png',  bbox_inches='tight')   
    
    
def plot_compute_scales_problem_fixed_GPUVS(args):
    cori_df = pd.read_csv("outputs/Compute Scales and Problem Fixed_scale8gpu.csv")
    summit_df = pd.read_csv('outputs/Compute Scales and Problem Fixed_scale.csv')
    cori_runtimes  = cori_df['Runtime'].values
    summit_runtimes = summit_df['Runtime'].values
#     cori_labels = cori_df['Total Gpus'].values
#     summit_labels =  summit_df['Total Gpus'].values
    cori_labels = cori_df['Nodes'].values
    summit_labels =  summit_df['Nodes'].values
#     cori_labels = cori_df['Total Gpus'].values
#     summit_labels =  summit_df['Total Gpus'].values
#     import pdb; pdb.set_trace()
    #NOTE: here is where I calculate expected line
#     summit_labels  = np.insert(summit_labels,1,2)
#     summit_labels  = np.insert(summit_labels,3,8)
#     cori_labels = np.insert(cori_labels,1,2)
    labels = np.unique(np.concatenate([cori_labels, summit_labels]))
    
    summit_expected = summit_runtimes[0] / (summit_labels /summit_labels[0])#[summit_runtimes[0] * (i+1) for i in range(len(summit_runtimes))]
    cori_expected = cori_runtimes[0] / (cori_labels/cori_labels[0])#[cori_runtimes[0] * (i+1) for i in 


    cori_stds = cori_df['Runtime Stddev'].values
    summit_stds = summit_df['Runtime Stddev'].values
#     summit_stds = np.insert(summit_stds,2,np.nan)

    bench_name = 'Ideal'
    
    fig, axs = plot_runtime_v_expected_gpu(args, cori_runtimes, cori_expected, cori_stds, summit_expected, summit_runtimes, summit_stds, bench_name, summit_labels,  cori_labels)
    
#     plt.xlabel("Total # of GPUs")
    plt.xlabel("Total # of Nodes")
    
#     ax.yaxis.set_major_locator(AutoLocator())  # solution
#     ax.set_yticks([0,30,60,90,120])
#     ax.set_ylim(0,120)
    """
    https://stackoverflow.com/questions/46498157/overlapping-axis-tick-labels-in-logarithmic-plots
    """

    for ax in axs:
        ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())

#         ax.yaxis.set_visible('False')
        #ax.minorticks_off() 
        for label in ax.yaxis.get_ticklabels()[::2]:#[::2]:
            label.set_visible(True)   
    plt.savefig(f'{args.constraint_file}.png',  bbox_inches='tight')


def plot_runtime_v_expected_gpu(args, cori_runtimes, cori_expected, cori_stds, summit_expected, summit_runtimes, summit_stds, bench_name, summit_labels,  cori_labels, yscale=True):
    fig = plt.figure()  
    # plot cori stuff
    plt.scatter(cori_labels, cori_runtimes, color='red', label="Cori 8 GPU / Node", s=15)
    plt.plot(cori_labels, cori_runtimes,  color='red')
    plt.fill_between(cori_labels, cori_runtimes-cori_stds, cori_runtimes+cori_stds, alpha=.20, color='red')
    
    
    plt.scatter(summit_labels, summit_expected, color='brown', label='Cori ' + bench_name, s=15, zorder=10)
    plt.plot(summit_labels, summit_expected,  color='brown')

    # plot summit stuff
    plt.scatter(summit_labels, summit_runtimes, color='blue', label="Cori 6 GPU / Node", s=15)
    plt.fill_between(summit_labels, summit_runtimes-summit_stds, summit_runtimes+summit_stds, alpha=.20, color='blue')
#     plt.scatter(np.arange(len(summit_expected)), summit_expected, color='purple', label='Summit ' + bench_name, s=15)
    plt.plot(summit_labels, summit_runtimes,  color='blue')

    axs = fig.axes
    # log scale y ticks
    if yscale:
        plt.yscale("log")
        plt.xscale("log")  
#     plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45)
    plt.ylabel("Total Runtime (s)")
    # note that command line arg is used as name
    plt.title(args.constraint_file)
    plt.legend()
    return fig, axs



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Benchmarking viz2')
    parser.add_argument('--constraint_file', type=str, required=True, default=None)
    parser.add_argument('--gpu', type=str, required=True, default=None)

    args = parser.parse_args()
    set_custom_params_plt()
    if args.gpu == 'True':
        if args.constraint_file =='Compute Scales and Problem Fixed' :
            plot_compute_scales_problem_fixed_GPUVS(args)
        elif args.constraint_file == 'Compute Scales and Problem Scales':
            plot_compute_scales_problem_scales_GPUVS(args)
        else:
            assert False, "Arg for constraint file not recognized"
    else:
        if args.constraint_file =='Compute Scales and Problem Fixed' :
            plot_compute_scales_problem_fixed(args)
        elif args.constraint_file == 'Compute Fixed and Problem Scales':
            plot_compute_fixed_problem_scales(args)
        elif args.constraint_file == 'Compute Scales and Problem Scales':
            plot_compute_scales_problem_scales(args)
        else:
            assert False, "Arg for constraint file not recognized"