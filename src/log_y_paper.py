import numpy as np
import statistics
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import LogLocator, LogFormatter


def log_y_paper(experiment_dir: str | os.PathLike, 
                experiment_date:str) -> None:
    OC = np.load(f'{experiment_dir}/OC_{experiment_date}.npy', allow_pickle=True).item()
    OC_RF_pos_cells = np.load(f'{experiment_dir}/OC_RF_pos_cells.npy', allow_pickle=True).item()

    x_axis = np.load(f'{experiment_dir}/x_axis_channel.npy')
    channel_num = len(x_axis)

    OC_auto_FI = OC.pop('0.unstained')
    if '4.OC_MIX_1' in OC and '4.OC_MIX_2' in OC: 
        OC_mix = OC.pop('4.OC_MIX_1')
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
            x, residuals = nnls(OC_RF, each_cell)
            x = np.round(x, decimals=4)
            scale_x.append(x)
        scale_x = np.array(scale_x)
        scale_x_single_stained[key] = scale_x

        total_cells = len(scale_x)
        print("total_cells", total_cells)

        for col in range(scale_x.shape[1]):
            RF_list = scale_x[:, col]
            bins_num = round((max(RF_list) - min(RF_list)) * 100) + 1
            # print('bins_num', bins_num)

            posCell_list = []
            if col > 0:
                # get both parameters of hist, and bins, bins as the x, hist as the y, the original x which is the scale_x
                # should be bins[index] while the original y which is the frequency is hist
                plt.figure(figsize=(6, 4))
                fig, ax = plt.subplots()
                ax.set_yscale('log')
                plt.rcParams['font.family'] = 'DejaVu Sans'
                plt.hist(RF_list, bins=bins_num, color='navy', alpha=0.6)
                plt.xlabel('Unmixed Relative Abundance', fontsize=18)
                plt.ylabel('Frequency', fontsize=18)
                ax.set_yscale('log')
                ax.set_ylim(1, 20000)
                major_locator = LogLocator(base=10.0, subs=(1.0,), numticks=10)
                ax.yaxis.set_major_locator(major_locator)

                formatter = LogFormatter(base=10, labelOnlyBase=True)
                ax.yaxis.set_major_formatter(formatter)

                plt.xlim(-0.1, 3)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.tight_layout()

                path4 = f'{experiment_dir}/3.OC_log_y/logy/'
                filename4 = key + '_RF' + str(col) + '.png'
                filepath4 = os.path.join(path4, filename4)
                plt.savefig(filepath4)

    np.save(f'{experiment_dir}/OC_scale_x_single_stained.npy', scale_x_single_stained, allow_pickle=True)


if __name__ == '__main__':
#------------------------------function call 02/07/2024-----------------------#
    experiment_dir = '../paper_exp020724'
    experiment_date = '020724'
    log_y_paper(experiment_dir, experiment_date)

#-----------------------------function call 04/17/2024------------------------#
    experiment_dir = '../paper_exp041724'
    experiment_date = '041724'
    log_y_paper(experiment_dir, experiment_date)

#-----------------------------function call 04/22/2024------------------------#
    experiment_dir = '../paper_exp042224'
    experiment_date = '042224'
    log_y_paper(experiment_dir, experiment_date)