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
from src.oc_unmixing_histogram import oc_unmixing_histogram
from src.oc_mix_unmixing import oc_unmixing
from src.OC_FI_vs_unmixing import ocfi_vs_unmixing
from src.fig6b_replication import replicate_fig6B
from src.fig6C_replication import fig6C_replication
from src.log_y_paper import log_y_paper

if __name__ == '__main__':
#------------------------------function call----------------------#
    experiment_dir = ['paper_exp020724', 'paper_exp041724', 'paper_exp042224']
    experiment_date = ['020724', '041724', '042224']

    for i in range(len(experiment_dir)):

        # Eextract data from Cytek results and store them in sorted dictionaries. 
        # PC refers to Positive Control, which means the relative samples were prepared with
        # the conventional kit. OC refers to the Oligo Complex, which means the relative 
        # samples were stained with the oligo complex labeled antibodies.###
        file_extraction(experiment_dir[i], experiment_date[i])

        # Since Cytek aurora's 48 channels provide 48 fluorescence valies per cell,
        # forming an array of shape as N x 48, N is the cell number for each 
        # stained cell group. By setting the MFI of unstained cells as autofluorescence(1 x 48)
        # and  subtracting it from each cell's fluorescence array, we could obtain the 
        # cell's fluorescence signal. This allow us to plot the spectrum of the entire
        # population of each stained cell group and find the index of the peak channel. 
        # Fluoresence signals of all cells in each stained cell group at this channel 
        # were collected and plotted as a histogram. Identify the main positive cell peak
        # and selectively gate it to obtain high signal to noise positive cells as the 
        # reference of that dye, which provide us the references for the further unmixing.####
        oc_histogram(experiment_dir[i], experiment_date[i])

        # Autofluorescence from unstained cells was also considered one of the referece 
        # together with the references for other three dyes, forming the reference list 
        # for unmixing as [autofluorescence, dye1, dye2, dye3]. The raw data of each cell
        # in each singly stained cell group were unmixied using non-negative least 
        # squares(scipy.optimize.nnls) to generate an unmixing weight array as N x 4, 
        # N is the cell numbers. Those unmixing weight values of each reference column 
        # in this result array were used to plot histograms to separate the positive and 
        # negative polulation.
        oc_unmixing_histogram(experiment_dir[i], experiment_date[i])

        # We applied logarithmic scaling to the y_axis to better visualize the distribution
        # of the histograms obtained above for publication.
        log_y_paper(experiment_dir[i], experiment_date[i])

        # The raw data of each cell in each triple-stained cell group were unmixed using
        # non-negative least squares (NNLS), as described above, to generate an unmixing
        # weight array.
        oc_unmixing(experiment_dir[i], experiment_date[i])

        # Here, the median fluorescence intensity (MFI) of positive cells in each group was 
        # used to generate the spectrum and identify the peak channel.  
        # Fluorescence value of each cell in each singly stained cell group at this peak channel 
        # was normalized by dividing it by the group’s MFI at the same channel. 
        # Similarly, the unmixed value of each cell in the same cell group was normalized by 
        # dividing it by its corresponding median value to generate the normalized unmixed relative 
        # abundance. The normalized fluorescence intensity in each cell is plotted versus its 
        # normalized unmixed relative abundance from PBMC singly stained with each antibody 
        # to generate fig.6B ###
        ocfi_vs_unmixing(experiment_dir[i], experiment_date[i])

        if experiment_dir[i] == 'paper_exp042224':

            # Figure 6B replication, only one instance of the 
            # replicate_fig6B function is needed.
            replicate_fig6B(experiment_dir[i])

            # The fig6C_replication script replicates the results of figure 6C in 
            # the paper.
            fig6C_replication(experiment_dir[i])
