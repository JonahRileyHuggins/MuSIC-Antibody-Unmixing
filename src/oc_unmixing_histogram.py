#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Name: oc_unmixing_histogram.py

Description: 

Author: Xiaoming Lu, Jonah R. Huggins
Date: 2024-05-23
Version: 1.0.0
"""

import numpy as np
import statistics
from scipy.optimize import nnls
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
from scipy.signal import find_peaks

def oc_unmixing_histogram(experimental_dir: str | os.PathLike, 
                          experimental_date: str) -> None:
    
    OC = np.load(f'{experimental_dir}/OC_{experimental_date}.npy', allow_pickle=True).item()
    OC_RF_pos_cells = np.load(f'{experimental_dir}/OC_RF_pos_cells.npy', allow_pickle=True).item()

    x_axis = np.load(f'{experimental_dir}/x_axis_channel.npy')
    channel_num = len(x_axis)

    OC_auto_FI = OC.pop('0.unstained')
    if '4.OC_MIX_1' in OC and '4.OC_MIX_2' in OC: 
        OC_mix_1 = OC.pop('4.OC_MIX_1')
        OC_mix_2 = OC.pop('4.OC_MIX_2')

    else:
        OC_mix = OC.pop('4.OC_MIX')

    OC_pos_cells = {}
    for key, value in OC.items():
        pos_cell_list = []
        for each_ind in OC_RF_pos_cells[key]:
            pos_cell_list.append(value[each_ind])
        OC_pos_cells[key] = np.array(pos_cell_list)

    # get the MFI of unstained cells as the autofluorescence
    auto_FI = []
    for i in range(channel_num):
        col_data = OC_auto_FI[:, i]  # now we get the col_data which stands for all cell's FI per channel
        MFI = statistics.median(sorted(col_data))
        auto_FI.append(MFI)
    RF_auto_FI = np.array(auto_FI)

    # get the pure signal of each reference
    OC_RF_MFI = {}
    for key, value in OC_pos_cells.items():
        pure_FI = value - auto_FI
        RF_MFI = []
        for i in range(channel_num):
            col_data = pure_FI[:, i]  # now we get the col_data which stands for all cell's FI per channel
            pure_MFI = statistics.median(sorted(col_data))
            RF_MFI.append(np.round(pure_MFI, decimals=1))
        OC_RF_MFI[key] = RF_MFI

    OC_RF_MFI['0.auto_FI'] = RF_auto_FI

    OC_RF0 = OC_RF_MFI['0.auto_FI']
    OC_RF1 = OC_RF_MFI['1.OC_ATTO488']
    OC_RF2 = OC_RF_MFI['2.OC_ATTO488_647_FRET']
    OC_RF3 = OC_RF_MFI['3.OC_ATTO647']

    OC_RF = [OC_RF0, OC_RF1, OC_RF2, OC_RF3]
    OC_RF = np.array(OC_RF).T

    scale_x_single_stained = {}
    count = 0
    OC_pos_gating = {}
    for key, val in sorted(OC.items()):
        print(key)
        scale_x = []
        count += 1
        for each_cell in val:
            x, _ = nnls(OC_RF, each_cell)
            x = np.round(x, decimals=4)
            scale_x.append(x)
        scale_x = np.array(scale_x)
        scale_x_single_stained[key] = scale_x

        total_cells = len(scale_x)
        print("total_cells", total_cells)

        for col in range(scale_x.shape[1]):
            RF_list = scale_x[:, col]
            bins_num = round((max(RF_list) - min(RF_list)) * 100) + 1

            posCell_list = []
            if col > 0:
                # get both parameters of hist, and bins, bins as the x, hist as the y, the original x which is the scale_x
                # should be bins[index] while the original y which is the frequency is hist
                plt.figure(figsize=(6, 4))
                hist, bins, _ = plt.hist(RF_list, bins=bins_num, color='navy', alpha=0.6, label='Reference' + str(col))
                # find the bins width
                bin_width = np.histogram_bin_edges(RF_list, bins=bins_num)[1] - np.histogram_bin_edges(RF_list,
                                                                                                    bins=bins_num)[0]
                plt.rcParams['font.family'] = 'DejaVu Sans'
                plt.xlabel('Unmixed Relative Abundance of ' + key, fontsize=9)
                plt.ylabel('Frequency', fontsize=9)
                plt.legend(loc='upper right', fontsize=9)
                plt.ylim(-0.1)
                plt.xlim(-0.1, 2.0)
                plt.xticks(fontsize=9)
                plt.yticks(fontsize=9)
                plt.tight_layout()
                path1 = f'{experimental_dir}/3.OC_unmixing_histogram/'
                filename = key + '_RF' + str(col) + '.png'
                filepath = os.path.join(path1, filename)
                plt.savefig(filepath)

                if col == count:
                    y = []
                    x = []
                    for h in range(0, len(hist)):
                        x.append(bins[h] + bin_width * 0.5)
                        y.append(hist[h])

                    max_height = np.max(y)
                    highest_peak_index = np.argmax(y)
                    neg_threshold = x[highest_peak_index]

                    # If there are too many noise peaks of peak[0] which is the negative peak, we need to use the
                    # smoothed data

                    sigma = 0.98
                    smoothed_data = gaussian_filter(y, sigma)

                    plt.figure(figsize=(6, 4))
                    hist, bins, _ = plt.hist(RF_list, bins=bins_num, color='green', alpha=0.25, label='Reference')
                    plt.plot(x, y, label='curve from histogram')
                    plt.plot(x, smoothed_data, label='Gaussian smoothed curve')

                    peak, _ = find_peaks(smoothed_data, prominence=max_height * 0.001)

                    if col == 1:

                        # find the highest positive peak and its index
                        max_peak_freq = -np.inf
                        max_peak_index = None
                        for each_peak in peak:
                            if col == count:
                                if x[each_peak] > 0.3 and smoothed_data[each_peak] > max_peak_freq:
                                    max_peak_freq = smoothed_data[each_peak]
                                    max_peak_index = each_peak

                            elif smoothed_data[each_peak] > max_peak_freq:
                                max_peak_freq = smoothed_data[each_peak]
                                max_peak_index = each_peak

                        # determine the positive peak bin index for the positive cells which should be in between of the
                        # main pos_peak and negative_threshold
                        RF_range = []
                        for n in range(len(x)):
                            if neg_threshold <= x[n] < x[max_peak_index]:
                                RF_range.append([n, x[n], smoothed_data[n]])

                        valley = min(RF_range, key=lambda y1: y1[2])
                        pos_threshold_bin = int(valley[0])
                        pos_threshold = x[pos_threshold_bin]

                    else:
                        smoothed_data[0] = 0
                        peak, _ = find_peaks(smoothed_data, prominence=max_height * 0.001)
                        dy_dx = np.gradient(smoothed_data, x)
                        pos_threshold_bin = np.where(np.diff(np.sign(dy_dx[peak[0]:])) > 0)[0][0] + peak[0]
                        pos_threshold = x[pos_threshold_bin]

                    for idx in range(len(RF_list)):
                        if RF_list[idx] >= pos_threshold:
                            posCell_list.append(idx)
                    print("the total number of positive cells gated by positive threshold by unmixing is " + str(
                        len(posCell_list)))

                    OC_pos_gating[key] = np.array(posCell_list)

                    plt.rcParams['font.family'] = 'DejaVu Sans'
                    plt.xlabel('Unmixed Relative Abundance of ' + key, fontsize=9)
                    plt.ylabel('Frequency', fontsize=9)
                    plt.axvline(x=pos_threshold, color='black', linestyle='--',
                                label='positive threshold')

                    plt.legend(loc='upper right', fontsize=9)
                    plt.xlim(-0.1, 2.0)
                    plt.ylim(-0.1, 500)
                    plt.tight_layout()

                    path2 = f'{experimental_dir}/3.OC_unmixing_histogram/gating/'
                    filename2 = key + '_RF' + str(col) + '.png'
                    filepath2 = os.path.join(path2, filename2)
                    plt.savefig(filepath2)

                    # get the figure with gating line for paper
                    plt.figure(figsize=(6, 4))
                    plt.rcParams['font.family'] = 'DejaVu Sans'
                    plt.hist(RF_list, bins=bins_num, color='navy', alpha=0.6, label='Reference' + str(col))
                    plt.xlabel('Unmixed Relative Abundance of ' + key, fontsize=9)
                    plt.ylabel('Frequency', fontsize=9)
                    plt.axvline(x=pos_threshold, color='black', linestyle='--',
                                label='positive threshold')

                    plt.xlim(-0.1, 2.0)
                    plt.ylim(-0.1, 500)
                    plt.xticks(fontsize=9)
                    plt.yticks(fontsize=9)
                    plt.tight_layout()

                    path3 = f'{experimental_dir}/3.OC_unmixing_histogram/final/'
                    filename3 = key + '_RF' + str(col) + '.png'
                    filepath3 = os.path.join(path3, filename3)
                    plt.savefig(filepath3)

    np.save(f'{experimental_dir}/OC_pos_gating_by_unmixing.npy', OC_pos_gating, allow_pickle=True)


if __name__ == '__main__':
#------------------------------function call 02/07/2024-----------------------#
    experiment_dir = '../paper_exp020724'
    experiment_date = '020724'
    oc_unmixing_histogram(experiment_dir, experiment_date)

#-----------------------------function call 04/17/2024------------------------#
    experiment_dir = '../paper_exp041724'
    experiment_date = '041724'
    oc_unmixing_histogram(experiment_dir, experiment_date)

#-----------------------------function call 04/22/2024------------------------#
    experiment_dir = '../paper_exp042224'
    experiment_date = '042224'
    oc_unmixing_histogram(experiment_dir, experiment_date)
