import os
import numpy as np
import statistics
from scipy.signal import find_peaks

def snr(experiment_dir: str | os.PathLike, experiment_name: str) -> None:

    PC = np.load(f'{experiment_dir}/PC_{experiment_name}.npy', allow_pickle=True).item()
    OC = np.load(f'{experiment_dir}/OC_{experiment_name}.npy', allow_pickle=True).item()
    x_axis = np.load(f'{experiment_dir}/x_axis_channel.npy')
    channel_num = len(x_axis)

    PC_pos_index = np.load(f'{experiment_dir}/PC_pos_gating_by_unmixing.npy', allow_pickle=True).item()
    OC_pos_index = np.load(f'{experiment_dir}/OC_pos_gating_by_unmixing.npy', allow_pickle=True).item()

    autoFI = PC.pop('0.unstained')

    # # determine the noise as the MFI of unstained cells
    unstained_MFI = []
    for i in range(channel_num):
        col_data = autoFI[:, i]  # now we get the col_data which stands for all cell's FI per channel
        MFI_auto = statistics.median((sorted(col_data)))
        unstained_MFI.append(MFI_auto)
    unstained_MFI = np.array(unstained_MFI)
    print(unstained_MFI)

    CD8_488_dict = {
        '1.PC_CF488': PC.pop('1.PC_CF488'),
        '1.OC_ATTO488': OC.pop('1.OC_ATTO488')
    }

    pos_cell_index = {
        '1.PC_CF488': PC_pos_index.pop('1.PC_CF488'),
        '1.OC_ATTO488': OC_pos_index.pop('1.OC_ATTO488')
    }

    # get the val of positive cells in each staining group
    pos_cells = {}
    for key, value in CD8_488_dict.items():
        pos_cell_list = []
        for each_ind in pos_cell_index[key]:
            pos_cell_list.append(value[each_ind])
        pos_cells[key] = np.array(pos_cell_list)

    # determine the peak of each stained cells in PC_pos_cells
    pos_spec_peak = {}
    for key, val in pos_cells.items():
        print(key)
        MFI_list = []
        for i in range(channel_num):
            col_stained = val[:, i]
            MFI = statistics.median(sorted(col_stained))
            MFI_list.append(MFI)
        MFI_array = np.array(MFI_list)
        pure_FI = MFI_array - unstained_MFI
        np.set_printoptions(suppress=True)

        max_spec_val = max(pure_FI)
        max_spec_index = np.argmax(pure_FI)
        print("max_peak_index: ", max_spec_index)
        height_threshold = round(max_spec_val * 0.6, 1)
        peak, _ = find_peaks(pure_FI, height=height_threshold)

        for p in peak:
            signal = pure_FI[p]
            noise = unstained_MFI[p]
            SNR = round(signal/noise, 4)
            print('for channel ' + str(p) + ', the SNR is ' + str(SNR))


#------------------------------function call----------------------#
if __name__ == '__main__':
    experiment_dir = ['../paper_exp020724', '../paper_exp041724', '../paper_exp042224']
    experiment_date = ['020724', '041724', '042224']

    for i in range(len(experiment_dir)):
        snr(experiment_dir[i], experiment_date[i])
