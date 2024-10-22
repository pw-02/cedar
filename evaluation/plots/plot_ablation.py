import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = "~/cedar/evaluation/plots/ablation.csv"
data = pd.read_csv(file_path)

rename_dict = {
    "Baseline": "Baseline",
    "plus parallelism": "+P",
    "plus reorder": "+PR",
    "plus offload": "+PRO",
    "plus fusion": "+PROF",
}
data["Setup"] = data["Setup"].map(rename_dict)

# Convert execution time to throughput
data["Runtime"] = 1 / data["Runtime"]

# Normalize the 'Average' for 'cedar-remote' in each 'Pipeline' group
normalization_factors = data[data["Setup"] == "Baseline"].set_index(
    "Pipeline"
)["Runtime"]
data["Normalized Runtime"] = data.apply(
    lambda row: row["Runtime"] / normalization_factors.get(row["Pipeline"], 1),
    axis=1,
)
print(data)

# Create the plot with normalized values
f = plt.figure(figsize=(3.33, 1.8), dpi=600)
# sns.set_style("whitegrid")
ax = sns.barplot(
    x="Pipeline",
    y="Normalized Runtime",
    hue="Setup",
    data=data,
    linewidth=0,
    hue_order=["Baseline", "+P", "+PR", "+PRO", "+PROF"],
)

# Add hatches
for i, bar in enumerate(ax.patches):
    if i in range(0, 8):
        bar.set_hatch("//")
    if i in range(8, 16):
        bar.set_hatch("\\\\")
    if i in range(16, 24):
        bar.set_hatch("--")
    if i in range(24, 32):
        bar.set_hatch("..")
    if i in range(32, 40):
        bar.set_hatch("oo")
ax.patches[40].set_hatch("////")
ax.patches[41].set_hatch("\\\\\\")
ax.patches[42].set_hatch("----")
ax.patches[43].set_hatch("..")
ax.patches[44].set_hatch("oo")

# Adding vertical lines and red "X" for missing values
pipeline_labels = data["Pipeline"].unique()  # Get unique pipeline labels

# Set x-ticks
# Adding vertical lines to mark ranges of each x category
for i in range(len(pipeline_labels) - 1):
    ax.axvline(
        x=i + 0.5, color="grey", linestyle="-", linewidth=0.5
    )  # End of group

plt.xticks(rotation=30, ha="right", fontsize=6)
plt.yticks(fontsize=6)
# plt.yticks((0, 0.5, 1), fontsize=6)
# ax.set_ylim((0, 50))
ax.tick_params(axis="both", which="major", pad=0)
# Set y ticks to small font
ax.set_ylabel("Normalized Throughput", fontsize=6)
ax.set_xlabel("")
ax.tick_params(axis="x", direction="out", length=3, color="black")
# ax.set_yscale("log")

ax.set_ylim((0, 26))

# Write the throughput for the ASR PROF setup above the bar
ax.text(0.93, 1.02, "43.83", fontsize=5, transform=ax.transAxes)


ax.legend(fontsize=5, title_fontsize="6", ncol=5)

# Display the plot
plt.tight_layout()
# ax.legend(fontsize=6, title_fontsize='6')
f.savefig("ablation.png", bbox_inches="tight")
