#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Name: extraction.py

Description: This file extracts data from the Cytek Aurora flow cytometry 
results CSV files and saves the data in a numpy array format.

Author: Xiaoming Lu, Jonah R. Huggins
Date: 2024-05-23
Version: 1.0.0
"""

#------------------------------package import--------------------------------# 
import pandas as pd
import sys
import ast
import numpy as np
import glob
import os
sys.path.append('src')
import get_str


#------------------------------function definition----------------------------#
def file_extraction(experiment_dir: str | os.PathLike, 
                    experiment_date: str) -> None:
    """
    Extracts data from the Cytek Aurora flow cytometry results CSV files and saves the data in a numpy array format.

    Input: 
    - experiment_dir: The directory containing the Cytek Aurora flow cytometry results CSV files
    - experiment_date: The date of the experiment in the format MMDDYY

    Output:
    - x_axis_channel.npy: The x-axis labels for the Cytek channels
    - oc_{experiment_name}.npy: The processed data for oligo complexes
    Files are saved in the experiment_name directory.
    """

    print(f'Extracting results from {experiment_dir}...')
    # x-axis labels correspond to Cytek channels (V)iolet, (B)lue, (Y)ellow-(G)reen, (R)ed
    x_axis = ['V1-A', 'V2-A', 'V3-A', 'V4-A', 'V5-A', 'V6-A', 'V7-A', 'V8-A', 'V9-A', 'V10-A', 'V11-A', 'V12-A', 'V13-A',
          'V14-A', 'V15-A', 'V16-A', 'B1-A', 'B2-A', 'B3-A', 'B4-A', 'B5-A', 'B6-A', 'B7-A', 'B8-A', 'B9-A', 'B10-A',
          'B11-A', 'B12-A', 'B13-A', 'B14-A', 'YG1-A', 'YG2-A', 'YG3-A', 'YG4-A', 'YG5-A', 'YG6-A', 'YG7-A', 'YG8-A',
          'YG9-A', 'YG10-A', 'R1-A', 'R2-A', 'R3-A', 'R4-A', 'R5-A', 'R6-A', 'R7-A', 'R8-A']


    # Dictionaries to store data for oligo complex (oc)
    oc = {}

    files = glob.glob(os.path.join(f'{experiment_dir}/csv_files', '*.csv'))

    # Iterate through each results CSV, separate the data into pc and oc, and store in the respective dictionaries  
    for file in sorted(files):

        # Extract the cytek results directory name as the last instance of the directory path
        front_str = 'csv_files/'
        back_str = '.csv'
        name = get_str.get_str(file, front_str, back_str)

        data = pd.read_csv(file)

        data_array = np.array([ast.literal_eval(row[0]) for row in data.values])

        data_array = np.round(data_array, decimals=2)
        
        if "unstained" in name:
          oc.update({name: data_array})
        else:
          oc.update({name: data_array})

    # Creating a sorted dictionary for the oligo complex data
    oc_final = {}
    for each_key in sorted(oc.keys()):
        oc_final[each_key] = oc[each_key]

    # Save the extracted data to numpy files
    np.save(f'{experiment_dir}/x_axis_channel.npy', x_axis)
    np.save(f'{experiment_dir}/OC_{experiment_date}.npy', oc_final, allow_pickle=True)


    print(f'Results extracted, located at {os.path.join(os.getcwd(), experiment_dir)}')


if __name__ == '__main__':
#------------------------------function call 02/07/2024-----------------------#
    experiment_dir = '../paper_exp020724'
    experiment_date = '020724'
    file_extraction(experiment_dir, experiment_date)

#------------------------------function call 04/17/2024-----------------------#
    experiment_dir = '../paper_exp041724'
    experiment_date = '041724'
    file_extraction(experiment_dir, experiment_date)

#------------------------------function call 04/22/2024-----------------------#
    experiment_dir = '../paper_exp042224'
    experiment_date = '042224'
    file_extraction(experiment_dir, experiment_date)

