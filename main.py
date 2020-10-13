# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.legend_handler import HandlerLine2D

def formaterY(x, pos):
    'The two args are the value and tick position'
    return '%3d' % (x*100)
def formaterX(x, pos):
    return '%3d' % (x*1e-3)

def read_file_csv(file_name):
    with open(file_name) as csv_file:
        effort = []
        avg = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                sum  = 0
                for col in range(len(row)):
                    if col == 0 or col == 1:
                        if col == 0:
                            effort.append(row[col])
                    else:
                        sum += float(row[col])
                avg.append(sum/32)
    return effort, avg

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.



def left_plot(ax):
    arr1 = ['rsel.csv', 'cel-rs.csv', '2cel-rs.csv', 'cel.csv', '2cel.csv']
    colors = ['blue', 'green', 'red', 'black', 'magenta']
    marks = ['o', 'v', 'D', 's', 'd']
    line_labels = ['1-Evol-RS', '1-Coev-RS', '2-Coev-RS', '1-Coev', '2-Coev']

    ax.grid(True, axis='both', color='grey', linestyle=(0, (5, 20)), linewidth=0.3, zorder=10000)
    ax.tick_params(axis='both', bottom=True, top=True, right=True, left=True, direction='in', length=6, labelsize=11)

    fig, ax = plt.subplots()
    formatterYY = FuncFormatter(formaterY)
    formaterXX = FuncFormatter(formaterX)
    ax.set_xticks([x for x in range(0, 500000, 100000)])

    ax2 = ax.twiny()
    ax2.set_xticks([x for x in range(0, 240, 40)])
    ax2.tick_params(axis='both', bottom=False, top=True, right=False, left=False, direction='in', labelsize=12)
    ax2.set_xlabel('Pokolenie', size=12)

    ax.set_xlabel('Rozegranych gier (×1000)', size=12)
    ax.set_ylabel('Odsetek wygranych gier [%]', size=12)
    lines = []
    ax.set_xlim(0, 500000)
    for i in range(len(arr1)):
        arrX, arrY = read_file_csv(arr1[i])
        ax.yaxis.set_major_formatter(formatterYY)
        ax.xaxis.set_major_formatter(formaterXX)
        line, = plt.plot(arrX, arrY,
                         marks[i],
                         ls='-',
                         color=colors[i],
                         alpha=0.8,
                         zorder=0,
                         markevery=25,
                         markeredgecolor='black',
                         label=line_labels[i]
                         )
        plt.legend()
        lines.append(line)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fig, ax_global = plt.subplots(1, 2, figsize=(10, 8))

    left_plot(ax_global[0])
    print_hi('PyCharm')
    plt.show()
    plt.close()

