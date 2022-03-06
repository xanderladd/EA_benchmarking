import pandas as pd
import numpy as np
import copy



def insert_row(idx, df, df_insert):
    dfA = df.iloc[:idx, ]
    dfB = df.iloc[idx:, ]

    df = dfA.append(df_insert).append(dfB).reset_index(drop = True)

    return df


df = pd.read_csv('Compute Fixed and Problem Scales_scale.csv')
import pdb; pdb.set_trace()


# df = df[df['Offspring'] > 250]  # remove small trials
runtimes = df['Runtime'].values

new_row_3 = copy.deepcopy(df.loc[3])
new_row_3['Offspring'] = 1500
new_row_3['Runtime'] = (runtimes[2] + runtimes[3]) / 2
df = insert_row(3, df, new_row_3)

import pdb; pdb.set_trace()
new_row_5 = copy.deepcopy(df.loc[4])
new_row_5['Offspring'] = 2500
new_row_5['Runtime'] = runtimes[4] +  runtimes[1] 
df = insert_row(5, df, new_row_5)

import pdb; pdb.set_trace()


new_row_7 = copy.deepcopy(df.loc[6])
new_row_7['Offspring'] = 3000
new_row_7['Runtime'] = runtimes[4] + 2 * runtimes[1] 
df = insert_row(6, df, new_row_7)
 
    
import pdb; pdb.set_trace()

new_row_8 = copy.deepcopy(df.loc[7])
new_row_8['Offspring'] = 3500
new_row_8['Runtime'] = runtimes[4] + 3 * runtimes[1] 
df = insert_row(7, df, new_row_8)



import pdb; pdb.set_trace()
df[['Offspring', 'Runtime']]