import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from node import get_node_data
from results_helper_functions import find_similar_pairs, get_pair_df
import pandas as pd
import numpy as np
import seaborn as sns

kernel_colors = {"linear" : "r", "rbf" : "b"}
balanced_colors = {True : "k", False : "c"}
# colors = ["bo", "go", "ro", "co", "mo", "yo", "ko"]
direction_markers = [9, 8]
method_markers = ["o", "."]
marker_meaning = {9 : "forward", 8 : "backwards"}
normalized = {True : "_std", False: ""}
alpha_norm = {True : 1/2, False: 1}
std = ["original", "standardised"]

def plot_gnfuv_r2_discrpenacy(df):
    for experiment in range(1,4):
        directory = f'results/GNFUV/figures/experiment_{experiment}{normalized[df["std"].values[0]]}.pdf' 
        print(f"Experiment {experiment}")
        exp = df.loc[df.experiment==experiment]
        experiment_stats = pd.DataFrame(columns = exp.discrepancy.describe().index)
        plot_r2_discrepancy(directory, exp, experiment_stats)
        
def plot_banking_r2_discrepancy(df):
    directory = f'results/bank-marketing/summary.pdf' 
    stats = pd.DataFrame(columns = df.discrepancy.describe().index)
    df["std"] = False
    plot_r2_discrepancy(directory, df, stats)

def plot_r2_discrepancy(directory, df, stats):
    with PdfPages(directory) as pdf:
        similar_pairs = find_similar_pairs(df)
        pair_dfs = [get_pair_df(pair, df) for pair in similar_pairs[::2]]

        n_pairs = len(pair_dfs)

        xlabel = "Discrepancy"
        ylabel = "Coefficient of Determination"

        for pair_index in range(n_pairs):
            forward_pair, backward_pair = pair_dfs[pair_index].keys()
            forward_df, backward_df = pair_dfs[pair_index].values()
            pair_df = pd.concat([forward_df, backward_df])

            fig, axs = plt.subplots(nrows = 1, ncols = 1, sharey="row", sharex= "row", figsize= (6, 6))
            fig.suptitle(forward_pair, fontsize =13)
            fig.tight_layout(pad=6)

            stats.loc[forward_pair] = forward_df.discrepancy.describe().values.T
            stats.loc[backward_pair] = backward_df.discrepancy.describe().values.T

            for row in pair_df.itertuples(index=False):
                if "kernel" in str(row):
                    color = kernel_colors[row.kernel]
                else:
                    color = balanced_colors[row.balanced]
                x, y = round(row.discrepancy,2), round(row.model_r2,2)

                if row.model_node < row.test_node:
                    direction = 0
                else:
                    direction = 1 

                axs.plot(x, y, color, alpha = alpha_norm[row.std], marker = method_markers[0])
                axs.plot(x, y, color, alpha = alpha_norm[row.std], marker = direction_markers[direction])
                axs.plot(x, y, "w", alpha = 0.8, marker = method_markers[1])
                axs.plot([0,x],[y-x,y], color, alpha = 0.3, linestyle = (0, (1, 1)))

            axs.set_ylabel(ylabel)
            axs.set_xlabel(xlabel)
            axs.set_ylim([0,1.1])
            x, y = [x], [y]

            usability_line = axs.plot([0,1.1],[0,1.1],c = "g", linestyle= "--", alpha=0.5)
            if "kernel" in pair_df.columns:
                legend_color = kernel_colors["rbf"]
                model_type_lines = [Line2D(x, y, c= kernel_colors[k], alpha = 1, linewidth=5) for k in ["linear","rbf"]]
                std_lines = [Line2D(x, y, c= "b", alpha = alpha_norm[True], linewidth=5)]
                lines = std_lines + model_type_lines + usability_line
                text = ["std", "linear", "non-linear", "equilibrium"]
            else:
                legend_color = balanced_colors[True]
                model_type_lines = [Line2D(x, y, c= balanced_colors[b], alpha = 1, linewidth=5) for b in [True,False]]
                lines = model_type_lines + usability_line
                text = ["balanced", "unbalanced", "equilibrium"]


            threshold_lines = []
            forward_direction_line = Line2D(x, y, c= legend_color, alpha = alpha_norm[row.std], marker = 9, linewidth = 0)
            backward_direction_line = Line2D(x, y, c= legend_color, alpha = alpha_norm[row.std], marker = 8, linewidth = 0)
            inner_point = Line2D(x, y, c= "w", alpha = 0.8, marker = method_markers[1], linewidth = 0)

            threshold_lines.append((Line2D(x, y, c= legend_color, alpha = alpha_norm[row.std], marker = method_markers[0], linewidth = 0),
                                    forward_direction_line, inner_point))
            threshold_lines.append((Line2D(x, y, c= legend_color, alpha = alpha_norm[row.std], marker = method_markers[0], linewidth = 0),
                                    backward_direction_line, inner_point))

            lines += threshold_lines


            mean, std, minimum, maximum = stats.loc[forward_pair][["mean", "std", "min", "max"]]
            forward_statistics_text =f"\nmin={round(minimum,2)} max={round(maximum,2)} \nmean={round(mean,2)} std={round(std,2)}"

            mean, std, minimum, maximum = stats.loc[backward_pair][["mean","std", "min", "max"]]
            backward_statistics_text =f"\nmin={round(minimum,2)} max={round(maximum,2)} \nmean={round(mean,2)} std={round(std,2)}"

            mmd_threshold = pair_df.mmd_score.values[0]
            ocsvm_thresholds = forward_df.ocsvm_score.values[0], backward_df.ocsvm_score.values[0]

            text.append(f"forward pair \nMMD({mmd_threshold}) OCSVM({ocsvm_thresholds[0]})" + 
                        forward_statistics_text)
            text.append(f"backward pair \nMMD({mmd_threshold}) OCSVM({ocsvm_thresholds[1]})"+ 
                        backward_statistics_text)

            l1 = plt.legend(lines, text[:-2])
            l2 = plt.legend(threshold_lines, text[-2:], bbox_to_anchor=(0.5, 0.5, 0, 1.5),
                            loc="center", borderaxespad=0)
            axs.add_artist(l1)
            axs.add_artist(l2)
            pdf.savefig(fig)
            plt.show()
            
def pairplot_experiment(node_data):
    if "std" in node_data[0].columns:
        label = "pi_std"
        alpha = ".5"
    else:
        label = "pi"
        alpha = ".9"
    df = pd.concat(node_data)[["humidity", "temperature", label]]

    g = sns.PairGrid(df, hue=label, corner = True, height = 4)
    g.map_diag(sns.histplot, multiple="stack", element="step", color= alpha)
    g.map_offdiag(sns.scatterplot)
    experiment = node_data[0].experiment.values[0]
    g.add_legend(title="Experiment "+str(experiment), fontsize = 12,
                 adjust_subtitles=True, bbox_to_anchor=(0.75,0.7))
    
    if "std" in node_data[0].columns:
        directory = f'results/GNFUV/figures/experiment_{experiment}_pairplot_std.png' 
    else:
        directory = f'results/GNFUV/figures/experiment_{experiment}_pairplot.png' 
    plt.savefig(directory)
    plt.show()
    
def visualise_experiment(node_data):
    plt.rcParams["figure.figsize"] = (8,5)
    
    a,b,c,d = node_data
    plt.matshow(corr)
    plt.scatter(a.humidity, a.temperature, marker= ".", label="pi2", alpha =0.5)
    plt.scatter(b.humidity, b.temperature, marker = "v", label="pi3", alpha=0.5)
    plt.scatter(c.humidity, c.temperature, marker = "<", label="pi4", alpha=0.5)
    plt.scatter(d.humidity, d.temperature, marker = "*", label="pi5", alpha=0.5)
    
    plt.legend()
    plt.title(label="Experiment "+str(a.experiment.values[0])+ " Data")
    plt.xlabel(xlabel="Humidity")
    plt.ylabel(ylabel="Temperature")
    plt.show()

def visualise_experiments(data):
    for i in range(1,4):
        node_data = get_node_data(data, experiment = i, filtered=False)
        visualise_experiment(node_data)