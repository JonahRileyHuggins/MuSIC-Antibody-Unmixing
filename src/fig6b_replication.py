import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def replicate_fig6B(experiment_dir: str | os.PathLike):
    dict_0207 = np.load('paper_exp020724/exp020724_6B.npy', allow_pickle=True).item()
    dict_0417 = np.load('paper_exp041724/exp041724_6B.npy', allow_pickle=True).item()
    dict_0422 = np.load('paper_exp042224/exp042224_6B.npy', allow_pickle=True).item()

    '''
    for key in dict_0207.keys():
        print(key)
    '''

    new_dict = {}
    for key in dict_0207.keys():
        combined_data = []
        data1 = dict_0207[key]
        data2 = dict_0417[key]
        data3 = dict_0422[key]

        combined_data.extend(data1)
        combined_data.extend(data2)
        combined_data.extend(data3)

        new_dict[key] = combined_data

    final_dict = {}

    key_mapping = {'1.OC_ATTO488': 'Atto 488', '2.OC_ATTO488_647_FRET': 'Atto 488/647', '3.OC_ATTO647': "Atto 647"}

    for old_key, value in new_dict.items():
        new_key = key_mapping.get(old_key, old_key)
        final_dict[new_key] = value

    # print(final_dict.keys())

    colors = ['orange', 'gray', 'blue']
    markers = ['o', 's', '^']

    fig = plt.figure(figsize=(7, 6))
    epsilon = 1e-4

    sns.set(style="whitegrid")
    palette = sns.color_palette("Set1", 3)

    plt.figure(figsize=(7, 7))
    plt.rcParams['font.family'] = 'DejaVu Sans'
    epsilon = 1e-4

    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

    ax.tick_params(axis='both', width=2, colors='black')

    for (cell_type, combined_list), color, marker in zip(final_dict.items(), palette, markers):
        x_coords = [item[0] for item in combined_list]
        y_coords = [item[1] for item in combined_list]

        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        x_coords[x_coords <= 0] = epsilon
        y_coords[y_coords <= 0] = epsilon

        x_log = np.log10(x_coords)
        y_log = np.log10(y_coords)

        plt.scatter(x_log, y_log, color=color, label=cell_type, marker=marker, alpha=0.8, edgecolors='w', s=20, linewidth=0.25)

    plt.xlabel('Log10(Normalized Unmixed Relative Abundance)', fontsize=18)
    plt.ylabel('Log10(Normalized Fluorescence intensity)', fontsize=18)

    legend = plt.legend(fontsize=14)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_linewidth(2)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    path = f'{experiment_dir}/result/'
    filename2 = 'summary of fig6B_seaborn.png'
    filepath2 = os.path.join(path, filename2)
    plt.savefig(filepath2)
    plt.close(fig)


if __name__ == '__main__':
    replicate_fig6B()
