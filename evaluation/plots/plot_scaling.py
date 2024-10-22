import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


set_data = {
    "set_procs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    "observed_tput": [
        70.76981133,
        82.05666667,
        149.46,
        229.2966667,
        289.39,
        351.71,
        418.7233333,
        478.2433333,
        535.5066667,
        601.7,
        634.5366667,
        681.65,
        729.01,
        757.992,
        791.775,
        820.474,
        862.17,
    ],
}

target_data = {
    "target_tput": [
        40,
        100,
        200,
        400,
        600,
    ],
    "observed_procs": [
        0,
        2,
        3,
        6,
        10,
    ],
}

f = plt.figure(figsize=(3.33, 1.8), dpi=600)

# Line plot for the set data
ax = sns.lineplot(
    x="set_procs",
    y="observed_tput",
    data=pd.DataFrame(set_data),
    color="blue",
    label="observed throughput",
    linewidth=1,
)
ax.set_ylabel("Throughput (samples/s)", fontsize=6, labelpad=2)
ax.set_xlabel("Distributed Processes", fontsize=6, labelpad=2)
ax.tick_params(axis="x", direction="out", length=2, color="black")
ax.tick_params(axis="y", direction="out", length=2, color="black")
# set tick labels to small
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)

# Add x and y axis grid lines
ax.yaxis.grid(color="lightgray", linestyle="--", linewidth=0.5)
ax.xaxis.grid(color="lightgray", linestyle="--", linewidth=0.5)

ax.set_xlim(-1, 16)
ax.set_ylim(-50, 900)

# Draw horizontal line for target throughput
for i in range(len(target_data["target_tput"])):
    # Make the line go from the left to where the corresponding observed_procs is
    ax.axhline(
        target_data["target_tput"][i],
        xmin=0,
        xmax=(target_data["observed_procs"][i] + 1) / 17,
        color="red",
        linewidth=0.4,
        linestyle="-",
        label="target throughput",
    )
    # Draw the vertial line for the observed_procs
    ax.axvline(
        target_data["observed_procs"][i],
        ymin=0,
        ymax=(target_data["target_tput"][i] + 50) / 950,
        color="red",
        linewidth=0.4,
        linestyle="-",
    )


# Change legend to say "target throughput"
handles, labels = ax.get_legend_handles_labels()
# Don't show the legend for the set data
ax.legend(
    handles=handles,
    labels=["Observed Throughput", "Target Throughput and Tuned Scale"],
    fontsize=6,
    title_fontsize="6",
)

# Reduce pad between axis and labels
ax.tick_params(axis="both", which="major", pad=2)




plt.tight_layout()
# ax.legend(fontsize=6, title_fontsize='6')
f.savefig("scaling.png", bbox_inches="tight")
