import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = "~/cedar/evaluation/plots/aggregate_data.csv"
data = pd.read_csv(file_path)

# Do some renaming
data["System"] = data["System"].replace("ember-local", "(l) cedar")
data["System"] = data["System"].replace("ember-remote", "(r) cedar")
data["System"] = data["System"].replace("torch", "torch")
data["System"] = data["System"].replace("tf", "(l) tf")
data["System"] = data["System"].replace("plumber", "plumber")
data["System"] = data["System"].replace("ray-local", "(l) ray")
data["System"] = data["System"].replace("tfdata-service", "(r) tf")
data["System"] = data["System"].replace("fastflow", "fastflow")
data["System"] = data["System"].replace("ray-remote", "(r) ray")

gpu_throughputs = {
    "simclr": 895.49,
    "gpt2": 739.884,
    "lstm": 3383.13,
    "rnnt": 338.060,
    "ssd": 48.618,
}


# Make the order of the pipelines consistent
# Assign a consistent color to each system
# Use seaborn default pallette
color_map = {
    "torch": sns.color_palette()[0],
    "(l) tf": sns.color_palette()[1],
    "plumber": sns.color_palette()[2],
    "(l) ray": sns.color_palette()[3],
    "(l) cedar": sns.color_palette()[4],
    "(r) tf": sns.color_palette()[5],
    "fastflow": sns.color_palette()[6],
    "(r) ray": sns.color_palette()[7],
    "(r) cedar": sns.color_palette()[8],
}

hatch_map = {
    "torch": "//",
    "(l) tf": "\\\\",
    "plumber": "||",
    "(l) ray": "--",
    "(l) cedar": "++",
    "(r) tf": "xx",
    "fastflow": "oo",
    "(r) ray": "..",
    "(r) cedar": "**",
}

ordering = [
    "torch",
    "(l) tf",
    "plumber",
    "(l) ray",
    "(l) cedar",
    "(r) tf",
    "fastflow",
    "(r) ray",
    "(r) cedar",
]
# For each system, create a new column in the data that assigns the system the appropriate index
data["Ordering"] = data["System"].apply(lambda x: ordering.index(x))

# Create subplot with 2 rows and 4 columns
f, axes = plt.subplots(2, 4, figsize=(7, 2.5), dpi=600)


def plot_subplot(df, ax, title, colors, hatches):
    # Matplotlib bar plot
    ax = df.plot(
        kind="bar",
        ax=ax,
        x="System",
        y="Throughput",
        color=colors,
        hatch=hatches,
    )

    # Don't show the legend
    ax.get_legend().remove()

    ax.set_xticks([])
    ax.set_ylabel("")

    # ax.ticklabel_format(axis='y', scilimits=[0, 0])

    ax.set_xlabel(title, fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.tick_params(axis="both", which="minor", labelsize=8)

    # Add light gray y-axis grid lines
    ax.yaxis.grid(color="gray", linestyle="--", linewidth=0.5)



# Filter out each pipeline
df = data[data["Pipeline"] == "CV-torch"]
# Place in specific order
df = df.sort_values("Ordering").reset_index(drop=True)
colors = [color_map[x] for x in df["System"]]
hatches = [hatch_map[x] for x in df["System"]]
plot_subplot(df, axes[0, 0], "CV-torch", colors, hatches)
ax = axes[0, 0]
ax.axvline(2.5, color="black", linewidth=1)
ax.text(0.3, 1.05, "Local", transform=ax.transAxes, ha="center", fontsize=6)
ax.text(0.8, 1.05, "Remote", transform=ax.transAxes, ha="center", fontsize=6)
ax.axhline(gpu_throughputs["simclr"], color="red", linewidth=1, linestyle="--")


df = data[data["Pipeline"] == "CV-tf"]
df = df.sort_values("Ordering").reset_index(drop=True)
colors = [color_map[x] for x in df["System"]]
hatches = [hatch_map[x] for x in df["System"]]
plot_subplot(df, axes[0, 1], "CV-tf", colors, hatches)
ax = axes[0, 1]
ax.axvline(3.5, color="black", linewidth=1)
ax.text(0.25, 1.05, "Local", transform=ax.transAxes, ha="center", fontsize=6)
ax.text(0.75, 1.05, "Remote", transform=ax.transAxes, ha="center", fontsize=6)
ax.axhline(gpu_throughputs["simclr"], color="red", linewidth=1, linestyle="--")

df = data[data["Pipeline"] == "NLP-torch"]
df = df.sort_values("Ordering").reset_index(drop=True)
colors = [color_map[x] for x in df["System"]]
hatches = [hatch_map[x] for x in df["System"]]
ax = axes[1, 0]
plot_subplot(df, ax, "NLP-torch", colors, hatches)
ax.axvline(2.5, color="black", linewidth=1)
ax.text(0.3, 1.05, "Local", transform=ax.transAxes, ha="center", fontsize=6)
ax.text(0.8, 1.05, "Remote", transform=ax.transAxes, ha="center", fontsize=6)
ax.axhline(gpu_throughputs["lstm"], color="red", linewidth=1, linestyle="--")

df = data[data["Pipeline"] == "NLP-hf-tf"]
df = df.sort_values("Ordering").reset_index(drop=True)
colors = [color_map[x] for x in df["System"]]
hatches = [hatch_map[x] for x in df["System"]]
ax = axes[1, 1]
plot_subplot(df, ax, "NLP-hf-tf", colors, hatches)
ax.axvline(2.5, color="black", linewidth=1)
ax.text(0.3, 1.05, "Local", transform=ax.transAxes, ha="center", fontsize=6)
ax.text(0.8, 1.05, "Remote", transform=ax.transAxes, ha="center", fontsize=6)
ax.axhline(gpu_throughputs["lstm"], color="red", linewidth=1, linestyle="--")

df = data[data["Pipeline"] == "NLP-tf"]
df = df.sort_values("Ordering").reset_index(drop=True)
colors = [color_map[x] for x in df["System"]]
hatches = [hatch_map[x] for x in df["System"]]
ax = axes[1, 2]
plot_subplot(df, ax, "NLP-tf", colors, hatches)
ax.axvline(2.5, color="black", linewidth=1)
ax.text(0.225, 1.05, "Local", transform=ax.transAxes, ha="center", fontsize=6)
ax.text(0.7, 1.05, "Remote", transform=ax.transAxes, ha="center", fontsize=6)
ax.axhline(gpu_throughputs["lstm"], color="red", linewidth=1, linestyle="--")

df = data[data["Pipeline"] == "ASR"]
df = df.sort_values("Ordering").reset_index(drop=True)
colors = [color_map[x] for x in df["System"]]
hatches = [hatch_map[x] for x in df["System"]]
ax = axes[1, 3]
plot_subplot(df, ax, "ASR", colors, hatches)
ax.axvline(3.5, color="black", linewidth=1)
ax.text(0.35, 1.05, "Local", transform=ax.transAxes, ha="center", fontsize=6)
ax.text(0.85, 1.05, "Remote", transform=ax.transAxes, ha="center", fontsize=6)
ax.axhline(gpu_throughputs["rnnt"], color="red", linewidth=1, linestyle="--")

df = data[data["Pipeline"] == "SSD-torch"]
df = df.sort_values("Ordering").reset_index(drop=True)
colors = [color_map[x] for x in df["System"]]
hatches = [hatch_map[x] for x in df["System"]]
ax = axes[0, 2]
plot_subplot(df, ax, "SSD-torch", colors, hatches)
ax.axvline(2.5, color="black", linewidth=1)
ax.text(0.3, 1.05, "Local", transform=ax.transAxes, ha="center", fontsize=6)
ax.text(0.8, 1.05, "Remote", transform=ax.transAxes, ha="center", fontsize=6)
ax.axhline(gpu_throughputs["ssd"], color="red", linewidth=1, linestyle="--")

df = data[data["Pipeline"] == "SSD-tf"]
df = df.sort_values("Ordering").reset_index(drop=True)
colors = [color_map[x] for x in df["System"]]
hatches = [hatch_map[x] for x in df["System"]]
ax = axes[0, 3]
plot_subplot(df, ax, "SSD-tf", colors, hatches)
ax.axvline(3.5, color="black", linewidth=1)
ax.text(0.25, 1.05, "Local", transform=ax.transAxes, ha="center", fontsize=6)
ax.text(0.75, 1.05, "Remote", transform=ax.transAxes, ha="center", fontsize=6)
ax.axhline(gpu_throughputs["ssd"], color="red", linewidth=1, linestyle="--")

# Add a common legend with all colors
# handles = [plt.Rectangle((0, 0), 1, 1, fc=color_map[x]) for x in ordering]
handles = [
    plt.Rectangle((0, 0), 1, 1, fc=color_map[x], hatch=hatch_map[x])
    for x in ordering
]
f.legend(
    handles,
    ordering,
    bbox_to_anchor=(0.5, 1.0),
    loc="upper center",
    ncol=9,
    fontsize=7,
)

f.tight_layout(rect=[0.02, 0, 1, 0.95])

f.text(
    0.01,
    0.5,
    "Throughput (samples/s)",
    va="center",
    rotation="vertical",
    fontsize=8,
)

# Draw a horizontal like in ax[0, 0]
# dashed line
# axes[0, 2].axhline(gpu_throughputs["gpt2"], color="red", linewidth=1, linestyle="--")
# axes[0, 3].axhline(gpu_throughputs["gpt2"], color="red", linewidth=1, linestyle="--")
# axes[1, 0].axhline(gpu_throughputs["gpt2"], color="red", linewidth=1, linestyle="--")
# axes[1, 0].axhline(gpu_throughputs["lstm"], color="red", linewidth=1, linestyle="--")
# axes[1, 1].axhline(gpu_throughputs["rnnt"], color="red", linewidth=1, linestyle="--")
# axes[1, 2].axhline(gpu_throughputs["ssd"], color="red", linewidth=1, linestyle="--")
# axes[1, 3].axhline(gpu_throughputs["ssd"], color="red", linewidth=1, linestyle="--")

# Save the image
plt.savefig("aggregate_data.png", bbox_inches="tight")
