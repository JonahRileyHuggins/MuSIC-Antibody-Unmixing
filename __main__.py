#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Name: __main__.py

Description: This module is the main module of the project. It calls all the 
functions in the other modules to perform the data extraction, unmixing,
histogram generation, and unmixing histogram generation.

Author: Xiaoming Lu, Jonah R. Huggins
Date: 2024-05-23
Version: 1.0.0
"""

from src.extraction import file_extraction
from src.oc_histogram import oc_histogram
from src.pc_histogram import pc_histogram
from src.oc_unmixing_histogram import oc_unmixing_histogram
from src.pc_unmixing_histogram import pc_unmixing_histogram
from src.oc_mix_unmixing import oc_unmixing
from src.pc_mix_unmixing import pc_unmixing
from src.snr import snr
from src.OC_FI_vs_unmixing import ocfi_vs_unmixing
from src.fig6b_replication import replicate_fig6B
from src.fig6C_replication import fig6C_replication
from src.log_y_paper import log_y_paper

if __name__ == '__main__':
#------------------------------function call----------------------#
    experiment_dir = ['paper_exp020724', 'paper_exp041724', 'paper_exp042224']
    experiment_date = ['020724', '041724', '042224']

    for i in range(len(experiment_dir)):

        #### To extract data from Excel files and store them in different dictionaries. PC refers to Positive Control, which means the realative samples were prepared with the conventional kit. OC refers to the Oligo Complex, which means the relative samples were stained with the oligo complex labeled antibodies.###
        file_extraction(experiment_dir[i], experiment_date[i])

        ####Here, since Cytek aurora's 48 channels provide 48 fluorescence valies per cell, forming an array of shape as N x 48, N is the cell number for each stained cell group. By setting the MFI of unstained cells as autofluorescence(1 x 48) and  subtracting it from each cell's fluorescence array, we could obtain the cell's fluorescence signal. This allow us to plot the spectrum of the eantire population of each stained cell group and find the index of the peak channel. Fluoresence signals of all cells in each stained cell group at this channel were collected and plotted as a histogram. Identify the main positive cell peak and selectively gate it to obtain high signal to noise positive cells as the reference of that dye, which provide us the references for the further unmixing.####
        oc_histogram(experiment_dir[i], experiment_date[i])
        pc_histogram(experiment_dir[i], experiment_date[i])

        ### Comment Here: ###
        oc_unmixing_histogram(experiment_dir[i], experiment_date[i])
        pc_unmixing_histogram(experiment_dir[i], experiment_date[i])

        ### Comment Here: ###
        log_y_paper(experiment_dir[i], experiment_date[i])

        ### Comment Here: ###
        oc_unmixing(experiment_dir[i], experiment_date[i])
        pc_unmixing(experiment_dir[i], experiment_date[i])

        # The SNR script stands for "Signal to Noise Ratio" and the signal is 
        # the median fluorescence intensity of the unstained cells subtracted from
        # the median fluorescence intensity of the stained cells. The 
        # noise is the median fluorescence intensity of the unstained cells. 
        # The SNR is calculated by dividing the signal of the peak channel of the 
        # stained cells by the noise from the same channel.
        snr(experiment_dir[i], experiment_date[i])

        ### Comment Here: ###
        ocfi_vs_unmixing(experiment_dir[i], experiment_date[i])

        if experiment_dir[i] == 'paper_exp042224':

            # Only one instance of the replicate_fig6B function is needed to
            # replicate the results of figure 6B in the paper.
            replicate_fig6B(experiment_dir[i])

            # The fig6C_replication script replicates the results of figure 6C in 
            # the paper.
            fig6C_replication(experiment_dir[i])