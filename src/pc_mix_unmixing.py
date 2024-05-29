import numpy as np
import statistics
from scipy.optimize import nnls
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
from scipy.signal import find_peaks

def pc_unmixing(experiment_dir: str | os.PathLike, experiment_name: str) -> None:
    PC = np.load(f'{experiment_dir}/PC_{experiment_name}.npy', allow_pickle=True).item()
    PC_RF_pos_cells = np.load(f'{experiment_dir}/PC_RF_pos_cells.npy', allow_pickle=True).item()

    x_axis = np.load(f'{experiment_dir}/x_axis_channel.npy')
    channel_num = len(x_axis)

    PC_auto_FI = PC.pop('0.unstained')
    PC_mix = PC.pop('4.PC_MIX')

    noise = []
    for i in range(channel_num):
        col_data = PC_auto_FI[:, i]  # now we get the col_data which stands for all cell's FI per channel
        MFI = statistics.median(sorted(col_data))
        noise.append(MFI)

    PC_pos_cells = {}
    for key, value in PC.items():

        pos_cell_list = []
        for each_ind in PC_RF_pos_cells[key]:
            pos_cell_list.append(value[each_ind])
        PC_pos_cells[key] = np.array(pos_cell_list)

    PC_RF_MFI = {}
    for key, value in PC_pos_cells.items():

        pure_FI = value - noise  # noise reduction
        RF_MFI = []
        for i in range(channel_num):
            col_data = pure_FI[:, i]  # now we get the col_data which stands for all cell's FI per channel
            pure_MFI = statistics.median(sorted(col_data))
            RF_MFI.append(np.round(pure_MFI, decimals=1))
        PC_RF_MFI[key] = RF_MFI

    PC_RF_MFI['0.auto_FI'] = noise

    PC_RF0 = PC_RF_MFI['0.auto_FI']
    PC_RF1 = PC_RF_MFI['1.PC_CF488']
    PC_RF2 = PC_RF_MFI['2.PC_CF568']
    PC_RF3 = PC_RF_MFI['3.PC_CF647']

    PC_RF = [PC_RF0, PC_RF1, PC_RF2, PC_RF3]

    PC_RF = np.array(PC_RF).T

    scale_x = []
    for i in range(len(PC_mix)):
        each_cell = PC_mix[i]
        x, residuals = nnls(PC_RF, each_cell)
        x = np.round(x, decimals=4)
        scale_x.append(x)

    total_cells = len(scale_x)
    print("total_cells of 4.PC_mix", total_cells)

    scale_x = np.array(scale_x)

    PC_mix_pos_gating = {}
    for col in range(scale_x.shape[1]):
        RF_list = scale_x[:, col]
        print('RF' + str(col))

        bins_num = round((max(RF_list) - min(RF_list)) * 100)

        posCell_list = []
        if col > 0:
            # get both parameters of hist, and bins, bins as the x, hist as the y
            # the original x which is the scale_x should be bins[index] while the original y which is the frequency is hist
            plt.figure(figsize=(6, 4))
            plt.rcParams['font.family'] = 'DejaVu Sans'
            hist, bins, _ = plt.hist(RF_list, bins=bins_num, color='green', alpha=0.2, label='Reference')
            # find the bins width
            bin_width = np.histogram_bin_edges(RF_list, bins=bins_num)[1] - np.histogram_bin_edges(RF_list, bins=bins_num)[
                0]

            y = []
            x = []
            for h in range(0, len(hist)):
                x.append(bins[h] + bin_width * 0.5)
                y.append(hist[h])

            plt.plot(x, y, label='curve from histogram')

            max_height = np.max(y)

            if experiment_dir != 'paper_exp020724':
                sigma = 0.98
            else:
                sigma = 2

            smoothed_data = gaussian_filter(y, sigma)

            # since there are too many noise peak of peak[0] which is the negative peak, we need to use the smoothed data
            dy_dx = np.gradient(y, x)

            plt.plot(x, smoothed_data, label='Gaussian smoothed curve')
            
            if experiment_dir == 'paper_exp020724':
                peak, _ = find_peaks(smoothed_data, prominence=max_height * 0.005)
                if col == 1:
                    closest_peak = min(peak, key=lambda p: abs(x[p] - 0.25))
                    peak[0] = closest_peak
                    dy_dx = np.gradient(smoothed_data, x)
                    pos_threshold = np.where(np.diff(np.sign(dy_dx[peak[0]:])) > 0)[0][0] + peak[0]
                else:
                    if x[peak[0]] > 0.2:
                        y[0] = 0
                        peak, _ = find_peaks(y, prominence=max_height * 0.01)
                    dy_dx = np.gradient(y, x)
                    pos_threshold = np.where(np.diff(np.sign(dy_dx[peak[0]:])) > 0)[0][0] + peak[0]

            elif experiment_dir == 'paper_exp041724':

                peak, _ = find_peaks(smoothed_data, prominence=max_height * 0.05)
                
                if col != 2:
                    if len(peak) == 1:
                        if x[peak[0]] > 0.2:
                            y[0] = 0
                            peak, _ = find_peaks(y, prominence=max_height * 0.05)

                    RF_range = []
                    for n in range(len(x)):
                        if x[peak[0]] <= x[n] < x[peak[1]]:
                            RF_range.append([n, x[n], smoothed_data[n]])

                    valley = min(RF_range, key=lambda y1: y1[2])

                    pos_threshold_bin = int(valley[0])
                    pos_threshold = x[pos_threshold_bin]

                else:
                    pos_threshold_index = np.where(np.diff(np.sign(dy_dx[peak[0]:])) > 0)[0][0] + peak[0]
                    pos_threshold = x[pos_threshold_index]

                
            elif experiment_dir == 'paper_exp042224':
                
                peak, _ = find_peaks(y, prominence=max_height*0.01)

                if x[peak[0]] > 0.2:
                    y[0] = 0
                    peak, _ = find_peaks(y, prominence=max_height * 0.02)

                dy_dx = np.gradient(smoothed_data, x)
                pos_threshold_index = np.where(np.diff(np.sign(dy_dx[peak[0]:])) > 0)[0][0] + peak[0]
                pos_threshold = x[pos_threshold_index]

            for idx2 in range(len(RF_list)):
                if RF_list[idx2] >= pos_threshold:
                    posCell_list.append(idx2)
            print("the total number of positive cells gated by positive threshold by unmixing is " + str(
                len(posCell_list)))
            PC_mix_pos_gating[col] = np.array(posCell_list)

            plt.xlabel('Unmixed Relative Abundance of PC_mix', fontsize=9)
            plt.ylabel('Frequency', fontsize=9)
            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)
            plt.axvline(x=pos_threshold, color='black', linestyle='--', label='positive threshold')

            plt.xlim(-0.1, 2.0)
            plt.ylim(-0.1, 500)
            plt.tight_layout()

            path = f'{experiment_dir}/4.PC_mix_unmixing/'
            filename = 'PC_mix_RF' + str(col) + '.png'
            filepath = os.path.join(path, filename)
            plt.savefig(filepath)

            # get the figure with gating line for paper
            plt.figure(figsize=(6, 4))
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.hist(RF_list, bins=bins_num, color='navy', alpha=0.6, label='Reference' + str(col))
            plt.xlabel('Unmixed Relative Abundance of PC_mix', fontsize=9)
            plt.ylabel('Frequency', fontsize=9)
            plt.axvline(x=pos_threshold, color='black', linestyle='--', label='positive threshold')

            plt.legend(loc='upper right', fontsize=9)
            plt.xlim(-0.1, 2.0)
            plt.ylim(-0.1)
            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)
            plt.tight_layout()

            path2 = f'{experiment_dir}/4.PC_mix_unmixing/final/'
            filename2 = 'PC_mix_RF' + str(col) + '.png'
            filepath2 = os.path.join(path2, filename2)
            plt.savefig(filepath2)

    np.save(f'{experiment_dir}/PC_mix_pos_gating_{experiment_name}.npy', PC_mix_pos_gating, allow_pickle=True)


if __name__ == '__main__':
#------------------------------function call 02/07/2024-----------------------#
    experiment_dir = '../paper_exp020724'
    experiment_date = '020724'
    pc_unmixing(experiment_dir, experiment_date)

#-----------------------------function call 04/17/2024------------------------#
    experiment_dir = '../paper_exp041724'
    experiment_date = '041724'
    pc_unmixing(experiment_dir, experiment_date)

#-----------------------------function call 04/22/2024------------------------#
    experiment_dir = '../paper_exp042224'
    experiment_date = '042224'
    pc_unmixing(experiment_dir, experiment_date)