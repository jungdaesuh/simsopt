#!/usr/bin/env python3

import os
import pandas as pd
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

# Define the common parent directory 
#parent_dir = 'outputs-iota15'
parent_dir = 'outputs-wout_nfp22ginsburg_000_014417_iota15.nc/'
# Get all result directories under the parent directory
result_dirs = [d for d in glob.glob(os.path.join(parent_dir, '*')) if os.path.isdir(d)]

df = pd.DataFrame()
# Loop through directories and read results.json
for result_dir in result_dirs:
    result_file = os.path.join(result_dir, 'results.json')
    if not os.path.isfile(result_file):
        continue  # Skip if results.json does not exist
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        # Extract the directory name relative to the parent directory
        relative_dirname = os.path.relpath(result_dir, parent_dir)
        # Add relative directory name to data
        data['dirname'] = relative_dirname
        # Normalize nested lists
        df = pd.concat([df, pd.json_normalize(data)], ignore_index=True)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading {result_file}: {e}")
        continue

if df.empty: raise Exception("Dataframe is empty: check that the parent directory name is correct")
df_temp = df[df['order'] == 2]
metric = 'MAJOR_RADIUS'
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
# Scatter plot with a different colormap
sc = ax.scatter(df_temp['FINAL_VOLUME'], df_temp['FIELD_ERROR'], 
                c=df_temp[metric], s=150, cmap='plasma')
ax.set_yscale("log")
ax.set_ylim(bottom=1e-3)
plt.colorbar(sc, label=metric)
plt.xlabel('Final Volume')
plt.ylabel('Field Error')
plt.title('Scatter Plot Colored by ' + metric)
plt.tight_layout()
plt.savefig(f'{parent_dir}field_error_plot.png', dpi=300, bbox_inches='tight')

