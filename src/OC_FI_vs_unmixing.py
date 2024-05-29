import numpy as np
import os
import statistics

def ocfi_vs_unmixing(experiment_dir: str | os.PathLike, experiment_date: str) -> None:

    OC_data = np.load(f'{experiment_dir}/OC_NR_data.npy', allow_pickle=True).item()
    OC_RF_pos_cells = np.load(f'{experiment_dir}/OC_RF_pos_cells.npy', allow_pickle=True).item()
    OC_scale_x_single_stained = np.load(f'{experiment_dir}/OC_scale_x_single_stained.npy', allow_pickle=True).item()

    x_axis = np.load(f'{experiment_dir}/x_axis_channel.npy')
    channel_num = len(x_axis)

    # get the positive cell population for reference
    OC_pos_cells = {}
    for key, value in OC_data.items():
        pos_cell_list = []
        for each_ind in OC_RF_pos_cells[key]:
            pos_cell_list.append(value[each_ind])
        OC_pos_cells[key] = np.array(pos_cell_list)

    # get the main pos_peak channel
    OC_peak = {}
    OC_RF_MFI = {}
    for key, value in OC_pos_cells.items():
        RF_MFI = []
        for i in range(channel_num):
            col_data = OC_data[key][:, i]  # now we get the col_data which stands for all cell's FI per channel
            pure_MFI = statistics.median(sorted(col_data))
            RF_MFI.append(np.round(pure_MFI, decimals=1))

        # max_val and max_index of this spectrum derived from MFI_list
        max_spec_val = max(RF_MFI)
        max_spec_index = np.argmax(RF_MFI)
        OC_peak[key] = max_spec_index

    normalized_FI = {}
    for key, val in OC_data.items():
        total_cells = val.shape[0]
        print(key, total_cells)
        FI_val = val[:, OC_peak[key]]
        Median_FI = statistics.median(sorted(FI_val))
        print('Median of fluorescence intensity', Median_FI)

        # normalization to MFI
        nFI_val = np.round(FI_val/Median_FI, decimals=4)
        normalized_FI[key] = nFI_val

    count = 0
    scale_x = {}
    for key, val in OC_scale_x_single_stained.items():
        total_cells = val.shape[0]
        print(key, total_cells)
        count += 1

        for col in range(val.shape[1]):
            if col == count:
                RF_arr = val[:, col]
                median_RF = statistics.median(RF_arr)
                nRF_arr = np.round(RF_arr/median_RF, decimals=4)
                scale_x[key] = nRF_arr

    exp042224_6B = {}
    for key in normalized_FI.keys():
        zipped = list(zip(scale_x[key], normalized_FI[key]))
        exp042224_6B[key] = zipped

    np.save(f'{experiment_dir}/exp{experiment_date}_6B.npy', exp042224_6B, allow_pickle=True)


#-----------------------------------function call---------------------------#
if __name__ == '__main__':
    experiment_dir = ['../paper_exp020724', '../paper_exp041724', '../paper_exp042224']
    experiment_date = ['020724', '041724', '042224']

    for i in range(len(experiment_dir)):
        ocfi_vs_unmixing(experiment_dir[i], experiment_date[i])









