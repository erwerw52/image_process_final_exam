import matplotlib.pyplot as plt
import numpy as np

def plot_grid(matrix, title, ax, highlight_coords=None):
    """繪製矩陣網格圖的輔助函式"""
    ax.imshow(matrix, cmap='Greys', vmin=0, vmax=1)
    ax.set_title(title)
    # 加上網格線
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)
    # 標註數值
    for (j, i), label in np.ndenumerate(matrix):
        color = 'white' if label == 0 else 'black' # 為了對比
        text_color = 'black' if label == 0 else 'white'
        ax.text(i, j, int(label), ha='center', va='center', color=text_color)
    
    if highlight_coords:
        for y, x in highlight_coords:
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
