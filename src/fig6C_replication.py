import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter
import os

def fig6C_replication(experiment_dir: str | os.PathLike,) -> None:
    OC_scale_x_single_stained = np.load(f'{experiment_dir}/OC_scale_x_single_stained.npy', allow_pickle=True).item()

    row_labels = ['Atto488 Stain', 'Atto488/647 Stain', 'Atto647 Stain']
    col_labels = ['Atto488 Unmix', 'Atto488/647 Unmix', 'Atto647 Unmix']
    x_label = 'Unmixed Relative Abundance'
    y_label = 'Frequency'

    data_list = []
    for key, val in OC_scale_x_single_stained.items():
        total_cells = val.shape[0]
        # print(key, total_cells)
        for col in range(val.shape[1]):
            RF_list = val[:, col]
            if col > 0:
                data_list.append(RF_list)


    fig, axs = plt.subplots(3, 3, figsize=(8, 6))

    for i in range(3):
        for j in range(3):
            ax = axs[i, j]
            data = data_list[i * 3 + j]
            # print(data)

            bins_num = round((max(data) - min(data)) * 100) + 1

            ax.set_yscale('log')
            plt.rcParams['font.family'] = 'DejaVu Sans'
            ax.hist(data, bins=bins_num, color='navy', alpha=0.6)

            ax.set_yscale('log')
            ax.set_ylim(1, 20000)
            ax.set_xlim(-0.1, 3)

            major_locator = LogLocator(base=10.0, subs=(1.0,), numticks=10)
            ax.yaxis.set_major_locator(major_locator)
            formatter = ScalarFormatter()
            formatter.set_powerlimits((0, 4))
            ax.yaxis.set_major_formatter(formatter)

            ax.tick_params(axis='y', labelsize=8)
            ax.tick_params(axis='x', labelsize=8)

            if i == 2 and j == 1:
                ax.set_xlabel(x_label, fontsize=11)

            if i == 0:
                ax.set_title(col_labels[j], fontsize=13, pad=10)

            if j == 0:
                ax.annotate(row_labels[i], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 10, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points', fontsize=13, ha='right', va='center',
                            rotation=90)

            ax.yaxis.set_major_formatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$')
            ax.set_xticks(np.arange(-0.1, 3.0, 0.5))

    y_label_horizontal_pos = 0.05
    y_label_vertical_pos = 0.45
    fig.text(y_label_horizontal_pos, y_label_vertical_pos, y_label, ha='center', fontsize=11, rotation='vertical')
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.tight_layout()

    path = f'{experiment_dir}/3.OC_log_y/subplot/'
    filename = 'fig6C.png'
    filepath = os.path.join(path, filename)
    plt.savefig(filepath)
    plt.close(fig)

#------------------------------function call----------------------#
if __name__ == '__main__':
    experiment_dir = '../paper_exp042224'
    fig6C_replication(experiment_dir)
