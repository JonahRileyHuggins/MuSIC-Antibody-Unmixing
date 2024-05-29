#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Name: PC_histogram.py

Description: 

Author: Xiaoming Lu, Jonah R. Huggins
Date: 2024-05-23
Version: 1.0.0
"""

import numpy as np
import statistics
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import LogLocator, LogFormatter
import math

def pc_histogram(experiment_dir: str, 
                 experiment_date: str) -> None:

    PC = np.load(f'{experiment_dir}/PC_{experiment_date}.npy', allow_pickle=True).item()

    x_axis = np.load(f'{experiment_dir}/x_axis_channel.npy')
    
    channel_num = len(x_axis)

    # determine the noise
    PC_autoFI = PC.pop('0.unstained')

    noise = []
    
    for i in range(channel_num):
    
        col_data = PC_autoFI[:, i]
        # now we get the col_data which stands for all cell's FI per channel
        # get the median fluorescence intensity value of each channel
        MFI = statistics.median(sorted(col_data))
    
        noise.append(MFI)
    
    noise_array = np.array(noise)

    # noise reduction
    PC_data = {}
    
    PC_unstained_data = {}
    
    RF_result = {}
    
    for key, value in sorted(PC.items()):
        
        pure_FI = value - noise_array  # noise reduction
        
        PC_data[key] = pure_FI

        # provide us the negative threshold from the unstained cells
        pure_FI_unstained = PC_autoFI - noise  # noise reduction
        
        PC_unstained_data[key] = pure_FI_unstained

        MFI_list = []
        
        for each_col in range(pure_FI.shape[1]):
        
            col_data = value[:, each_col]
        
            mfi = statistics.median(sorted(col_data))
        
            MFI_list.append(mfi)

        # max_val and max_index of this spectrum derived from MFI_list
        max_spec_val = max(MFI_list)
        
        max_spec_index = np.argmax(MFI_list)

        # now we need to know how many peaks higher than 2/5 of the height of highest peak in the spectrum
        # this is to roughly separate the background noise and the true signal
        height_threshold = round(max_spec_val * 0.4, 1)
        
        peaks, _ = find_peaks(MFI_list, height=height_threshold)
        
        RF_result[key] = max_spec_index, max_spec_val, peaks

        # Visualize the noise reduced spectrum
        plt.figure(figsize=(8, 4))
        plt.xlabel('Channels', fontsize=9)
        plt.plot(x_axis, MFI_list, label=key)
        plt.tick_params('x', labelsize=8, rotation=90)
        plt.xlabel('Channels', fontsize=9)
        plt.ylabel('Emission Fluorescence Intensity', fontsize=9)
        # plt.legend()
        plt.axhline(y=height_threshold, color='orange', linestyle='--')
        plt.subplots_adjust(bottom=0.14)
        plt.grid(True)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()

        path = f'{experiment_dir}/2.spectrum/'
        filename = key + '.png'
        filepath = os.path.join(path, filename)
        plt.savefig(filepath)

    PC_mix_data = PC_data.pop('4.PC_MIX')

    PC_pos_cells_neg = {}  # based on the negative threshold from the unstained cells

    PC_RF_pos_cells = {}  # based on the histogram of the stained cells
    
    for key, value in PC_data.items():
    
        values = []
    
        total_cells = value.shape[0]
    
        print(key, total_cells)

        # to get the FI value of each cell at the channel of max_spec_index
        for each_row in value:

            peak_value = each_row[RF_result[key][0]]  # RF_result[key] = max_spec_index
            
            values.append(peak_value)

        values = np.array(values)

        # to ensure all values are greater than 0 for subsequent logarithmic operations.
        pure_value = values - np.min(values) + 0.1

        log_value_list = []
        
        for each in pure_value:

            log_value = math.log10(each)
        
            log_value_list.append(round(log_value, 2))

        log_value_array = np.array(log_value_list)

        bins_num = round((max(log_value_array) - min(log_value_array)) * 30)

        # bin_width is an array containing each bin's right edge value
        bin_width = np.logspace(np.log10(min(pure_value)), np.log10(max(pure_value)), num=bins_num)

        # get both parameters of hist, and bins, bins as the x, hist as the y
        # the original x which is the intensity should be bins[index] while the original y which is the frequency is hist
        hist, bins, _ = plt.hist(pure_value, bins=bin_width, color='navy', alpha=0.4, label=key)

        # find the index of the max value of bins which is the max_hist_ind
        max_hist_ind = np.argmax(hist)

        # find the negative threshold
        PC_unstained = PC_unstained_data[key][:, RF_result[key][0]]  # RF_result[key] = max_spec_index
        pure_value_auto = PC_unstained - np.min(PC_unstained) + 0.1

        unstained_median = np.median(pure_value_auto)
        neg_threshold = np.percentile(pure_value_auto, 99.7)
        # print('neg_threshold', neg_threshold)

        # find the bin_index of neg_threshold, np.digitize is a NumPy function used to bin data into discrete intervals.
        # It returns an array indicating the bin number each element belongs to.
        neg_bin_index = np.digitize(neg_threshold, bin_width)

        y = []

        x = []

        for h in range(0, len(hist)):

            # since its log scale, so we need to * 0.1 then * 0.5 to get the middle point, which is 0.05
            x.append(bins[h] + bin_width[h] * 0.05)

            y.append(hist[h])

        # find the peaks on the curve
        curve_peak, _ = find_peaks(y, prominence=max(hist) * 0.01, distance=3)

        peak_info = []

        for i in curve_peak:

            peak_info.append([i, bins[i], int(hist[i])])

        # from here we know there are how many peaks in the spectrum, the negative peak should be within the negative
        # threshold while for RF threshold determination should be the highest peak since the negative threshold
        if max_hist_ind > neg_bin_index and max_hist_ind != peak_info[0][0]:

            main_positive_peak_index = max_hist_ind

        else:
            exclude_neg = peak_info[1:]
            
            # get the index of positive peak in curve_peak excluding the negative main peak
            maxHist_pos_peak = max(exclude_neg, key=lambda x3: x3[2])
            
            main_positive_peak_index = maxHist_pos_peak[0]

        # find the peak prior to the main positive peak
        prior_main_posPeak = None

        for idx, each_info in enumerate(peak_info):
        
            if each_info[0] == main_positive_peak_index:
        
                if idx > 0:
        
                    prior_main_posPeak = peak_info[idx - 1][0]
        
                break

        threshold = []

        # determine the negative peak bin index to separate negative and positive cells
        if peak_info[0][0] < neg_bin_index:

            main_negative_peak_index = peak_info[0][0]
            
            threshold.append(main_negative_peak_index)
            
            threshold.append(neg_bin_index)
            
        else:
            
            main_negative_peak_index = peak_info[0][0]
            
            threshold.append(main_negative_peak_index)
            
            threshold.append(main_negative_peak_index)
            
        # determine the positive peak bin index for the RF positive cells which should be in between of the main
        # pos_peak and the peak prior to it.
        RF_range = []
        
        for n in range(len(x)):
        
            if prior_main_posPeak < n < main_positive_peak_index:
        
                if x[n] > neg_threshold:
        
                    RF_range.append([n, x[n], y[n]])

        valley = min(RF_range, key=lambda y3: y3[2])
        
        RF_threshold = valley[0]

        threshold.append(RF_threshold)
        
        threshold.append(main_positive_peak_index)

        total_pos_cells_neg = []
        
        for each_ind in range(len(pure_value)):
        
            if pure_value[each_ind] >= bins[threshold[1]]:
        
                total_pos_cells_neg.append(each_ind)
        
        PC_pos_cells_neg[key] = np.array(total_pos_cells_neg)

        RF_pos_cells = []
        
        for each_ind in range(len(pure_value)):
        
            if pure_value[each_ind] >= bins[threshold[2]] + bin_width[threshold[2]] * 0.05:
        
                RF_pos_cells.append(each_ind)
        
        print('total positive cells based on the RF threshold: ', len(RF_pos_cells))
        
        PC_RF_pos_cells[key] = np.array(RF_pos_cells)

        plt.xscale('log')
        plt.gca().xaxis.set_major_locator(LogLocator(base=10.0))
        formatter = LogFormatter(labelOnlyBase=False, minor_thresholds=(1, 0.1))
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.xlim(800)
        plt.ylim(0, 1000)
        plt.xlabel('Fluorescence intensity', fontsize=9)
        plt.ylabel('Frequency', fontsize=9)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.axvline(x=bins[threshold[3]] + bin_width[threshold[3]] * 0.05, color='red', linestyle='--',
                    label='Positive main peak')
        plt.axvline(x=bins[threshold[2]] + bin_width[threshold[2]] * 0.05, color='blue', linestyle='--',
                    label='Reference threshold')

        path = f'{experiment_dir}/2.histogram/'
        filename = key + '.png'
        filepath = os.path.join(path, filename)
        plt.savefig(filepath)

    np.save(f'{experiment_dir}/PC_pos_cells_neg_threshold.npy', PC_pos_cells_neg, allow_pickle=True)
    np.save(f'{experiment_dir}/PC_RF_pos_cells.npy', PC_RF_pos_cells, allow_pickle=True)
    np.save(f'{experiment_dir}/PC_NR_data.npy', PC_data, allow_pickle=True)
    np.save(f'{experiment_dir}/PC_NR_unstained_data.npy', PC_unstained_data, allow_pickle=True)
    np.save(f'{experiment_dir}/PC_NR_mix_data.npy', PC_mix_data)


if __name__ == '__main__':
#------------------------------function call 02/07/2024-----------------------#
    experiment_dir = '../paper_exp020724'
    experiment_date = '../020724'
    pc_histogram(experiment_dir, experiment_date)
#------------------------------function call 04/17/2024-----------------------#
    experiment_dir = '../paper_exp041724'
    experiment_date = '041724'
    pc_histogram(experiment_dir, experiment_date)
#------------------------------function call 04/22/2024-----------------------#
    experiment_dir = '../paper_exp042224'
    experiment_date = '042224'
    pc_histogram(experiment_dir, experiment_date)