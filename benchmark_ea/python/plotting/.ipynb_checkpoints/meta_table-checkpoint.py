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
pd.set_option('display.max_rows', 500)

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
    # df[main] = df[main].round(decimals=2).astype(str) \
    # + " ± " + df[std].round(decimals=2).astype(str) 
    df[main] = df[main].astype(str) \
    + " ± " + df[std].astype(str) 
    df[main] = df[main].str.replace('0.0','0')
    for i in range(len(df[main])):
        if len(re.findall(r'\d+' , df.loc[i,main])[0]) >  2:
            df.loc[i,main] = df.loc[i,main].replace('.0','')
        
    # df[main].loc[lambda x: len(x) > 2] = df[main].loc[lambda x: len(x) > 2].str.replace('.0', '')

    # df[main] = df[main].str.replace(r'\d{3,}[0]$','', regex=True)
    if drop:
        df = df.drop(std,axis=1)
    return df
    

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
#     TODO FINISH MAKING BIG TABLE:
#         1.Make copy of benchmarking so u dont f things up
#         2.Start tracking refactor on git
#         3. Fix GPU utilization and time to 50
#         4. merge all the plot benchmarking stuff
#         5. refactor
#         6. pop out nice new table
#         7. cancel and re run compute fixed problem scale stuff
#         8. re run some coreneuron fixed
#         9. re run ipfx and efel
    
def mini_table(args):
    extension = os.path.basename(args.constraint_file) + "_scale.csv"
    all_dfs = []
    for exp_name in args.exp_names:   
        curr_df, _ = find_experiment_df(exp_name, extension)
        curr_df = curr_df.sort_values('Offspring')
        curr_df = curr_df.reset_index()
        runtimes  = curr_df['Runtime'].values
        labels =  curr_df['Nodes'].values
        # the mean gen size / offspring size to adjust for populations
        # that are not full size. This value is always VERY close to .8
        # hence the try / except
        # expected = runtimes[0] * (labels/labels[0] * np.mean(curr_df['Mean Gen Size'] / curr_df['Offspring'].values))
        # print(np.mean(curr_df['Mean Gen Size'] / curr_df['Offspring'].values))
        
        curr_df['Experiment'] = exp_name
        sigFigs = 2
        for col in curr_df.columns:
            if 'Runtime' in col:
                try:
                    curr_df[col] = curr_df[col].apply(lambda x: round(x, sigFigs - int(np.floor(np.log10(abs(x))))))
                except:
                    curr_df[col] = curr_df[col].apply(lambda x: round(x, sigFigs - int(np.floor(np.log10(abs(x + .00001))))))
        latex_df = plus_minus_cols(curr_df, main='Runtime',std='Runtime Stddev')
        all_dfs.append(curr_df)
    import pdb; pdb.set_trace()
    df = pd.concat(all_dfs)
    import pdb; pdb.set_trace()

    
    mini_df = df[['Nodes', 'Total Cpus', 'Total Gpus', 'Offspring',
       'Stimuli', 'Score Functions', 'Runtime', 'Experiment']]
    mini_df_run_time = mini_df.pivot(values='Runtime', columns='Experiment').reset_index()
    exp_cols = [name + " Runtime (s)" for name in args.exp_names]
    curr_df = curr_df[['Nodes', 'Total Cpus', 'Total Gpus', 'Offspring',
       'Stimuli', 'Score Functions']]
    for col, name in zip(exp_cols, args.exp_names):
        curr_df[col]  = mini_df_run_time[name]

    ## # OF CPU MONKEY PATCH
    curr_df = curr_df.drop("Total Cpus", axis=1)
    curr_df.insert(1, "Total CPUs (CPU Node)", curr_df['Nodes'].values * 64)
    curr_df.insert(1, "Total CPUs (GPU Node)", curr_df['Nodes'].values * 80)
    curr_df = drop_constant(curr_df) #preserve_list=['Offspring'] )
    
    if extension == 'Compute Scales and Problem Scales_scale.csv':
        columns_titles =['Nodes', 'Total CPUs (CPU Node)', 'Total CPUs (GPU Node)', 'Total Gpus',
       'Offspring', 'CPU-EA Runtime (s)', 'NeuroGPU-EA Runtime (s)',
       'CoreNeuronGPU-EA Runtime (s)']
        curr_df = curr_df.reindex(columns=columns_titles)
        curr_df = curr_df.rename({'Offspring':'Population'}, axis=1)
        
    if extension == 'Compute Scales and Problem Fixed_scale.csv':
        columns_titles =['Nodes', 'Total CPUs (CPU Node)', 'Total CPUs (GPU Node)', 'Total Gpus', 'CPU-EA Runtime (s)', 'NeuroGPU-EA Runtime (s)',
       'CoreNeuronGPU-EA Runtime (s)']
        curr_df = curr_df.reindex(columns=columns_titles)

        
    curr_df.to_csv(os.path.join('figures','mini_tables', extension))
    # curr_df.to_csv(os.path.join('figures','mini_tables', extension), float_format="%.2f")
    col_fmt = "|".join(np.repeat('c', len(df.columns)))
    col_fmt = "|" + col_fmt + "|"
    curr_df.to_latex(os.path.join('figures','mini_tables', os.path.basename(args.constraint_file) + '.tex'), float_format="%.0f", index=False, column_format=col_fmt)
    
def big_table(args):

    extension = os.path.basename(args.constraint_file) + "_scale.csv"
    all_dfs = []
    for exp_name in args.exp_names:   
        curr_df, _ = find_experiment_df(exp_name, extension)
        curr_df = curr_df.reset_index()
        runtimes  = curr_df['Runtime'].values
        labels =  curr_df['Nodes'].values
        # the mean gen size / offspring size to adjust for populations
        # that are not full size. This value is always VERY close to .8
        # hence the try / except
        try:
            expected = runtimes[0] * (labels/labels[0] * np.mean(curr_df['Mean Gen Size'] / curr_df['Offspring'].values))
            print(np.mean(curr_df['Mean Gen Size'] / curr_df['Offspring'].values))
        except:
            expected = runtimes[0] * (labels/labels[0] * .8)
        curr_df = curr_df[curr_df['Offspring'] > 200]
        curr_df['Expected'] =  expected
        # curr_df['Experiment'] = exp_name
        latex_df = plus_minus_cols(curr_df, main='Runtime',std='Runtime Stddev')
        latex_df = latex_df.drop(['FOM','FOM Std Dev', 'Unnamed: 0', 'index'],axis=1)
        if 'Mean Eval Time' in latex_df.columns:
            latex_df = plus_minus_cols(latex_df, main='Mean Eval Time',std='Std Eval Time')
            latex_df = latex_df.rename({'Mean Eval Time': 'Eval Time'}, axis=1)

        if 'Mean Sim Time' in latex_df.columns:
            latex_df = plus_minus_cols(latex_df, main='Mean Sim Time',std='Std Sim Time')
            latex_df = latex_df.rename({'Mean Sim Time': 'Sim Time'}, axis=1)
        
        if 'DEAP time' in latex_df.columns:
            latex_df = latex_df.drop(['DEAP time','DEAP Time Std Dev'],axis=1)
        
        if 'Mean Gen Size' in latex_df.columns:
            latex_df = plus_minus_cols(latex_df, main='Mean Gen Size',std='Std Gen Size')
            latex_df = latex_df.rename({'Mean Gen Size': 'Gen Size'}, axis=1)
        

           
        latex_df.to_latex(f'figures/big_tables/{exp_name}.tex', index=False)
        
    return
''' below is an attempt to combine all the DFs, not gonna work like that I don't think without having too many columns or too many rows, need to find a way to kind of combine three seperate tables in TEX'''


#     all_dfs.append(curr_df)
#     final_df = pd.concat(all_dfs)
#     df = final_df
#     mini_df = df[['Nodes', 'Total Cpus', 'Total Gpus', 'Offspring',
#        'Stimuli', 'Score Functions', 'Runtime', 'Experiment']]
#     mini_df_run_time = mini_df.pivot(values='Runtime', columns='Experiment').reset_index()
#     mini_df = mini_df.pivot(columns='Experiment').reset_index()
#     import pdb; pdb.set_trace()
#     # curr_df.loc[:,exp_cols] = mini_df_run_time.iloc[:,-len(exp_cols):]
#     final_df = final_df.drop(['FOM', 'Unnamed: 0'],axis=1)
#     latex_df = drop_constant(df, preserve_list=['Nodes','Offspring'] )
#     # latex_df = plus_minus_cols(latex_df, main='Runtime',std='Runtime Stddev')
#     latex_df['GPU Utilization'] = latex_df['GPU Utilization'].astype(str).apply(lambda x: x[:4])  + "%"

#     col_fmt = "|".join(np.repeat('c', len(df.columns)))
#     col_fmt = "|" + col_fmt + "|"
#     latex_df.to_csv(os.path.join('figures','tables', extension), float_format="%.2f")

#     latex_df.to_latex(os.path.join('figures','tables', os.path.basename(args.constraint_file) + '.tex'), float_format="%.0f", index=False, column_format=col_fmt)
    
                    

       
    

def find_experiment_df(exp, extension):
    if 'NeuroGPU-EA' in exp and '8 SF' not in exp:
        df = pd.read_csv(f'data/NeuroGPU_EA/experiments/regular_outputs/{extension}')  
        color = 'green'
    elif 'CPU-EA' in exp:
        df = pd.read_csv(f'data/Neuron_EA/experiments/outputs/{extension}')
        color = 'blue'
        
    elif 'Summit' in exp:
        df = pd.read_csv(f'data/NeuroGPU_EA/experiments/summit_outputs/{extension}') 
        color = 'darkorange'
    elif 'IPFX' in exp:
        df = pd.read_csv(f'data/NeuroGPU_EA/experiments/ipfx_outputs/{extension}')
        color = 'brown'
    elif '6 GPU' in exp:
        df = pd.read_csv(f'data/NeuroGPU_EA/experiments/6GPU_outputs/{extension}')
        color = 'green'
    elif 'eFEL' in exp:
        df = pd.read_csv(f'data/NeuroGPU_EA/experiments/8SF_outputs/{extension}')
        color = 'green'
    elif 'CoreNeuronGPU-EA' in exp:
        df = pd.read_csv(f'data/CoreNeuron_EA/experiments/outputs/{extension}')
        color = 'purple'
    
    
     
    df = df.drop_duplicates(subset=['Nodes','Runtime']) # TODO: add more keys
    df = df.rename({'Trials':'Num Trials'}, axis=1)
    if len(df[df['Num Trials'] < 49]) > 0 and exp != 'Summit':
        print("not enough trials for : ", exp, "@", extension)
        print(df[df['Num Trials'] < 49][['Num Trials', 'Offspring', 'Nodes']])
    # cut values that do not have more than 50 trials
    # df = df[df['Num Trials'].astype(int) > 48]

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
    
    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Benchmarking viz2')
    parser.add_argument('--constraint_file', type=str, required=True, default=None)
    parser.add_argument('-n', '--exp_names', nargs='+', default=[])



    args = parser.parse_args()
    mini_table(args)
    # big_table(args)
   
  