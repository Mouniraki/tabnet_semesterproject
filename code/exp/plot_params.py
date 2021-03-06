import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="sans-serif")
plt.rc("pdf", fonttype=42)
plt.rc("ps", fonttype=42)

matplotlib.rcParams["text.latex.preamble"] = "\n".join(
    [
        #       r'\usepackage{helvet}',
        r"\usepackage{siunitx}",
        r"\sisetup{detect-all}",
        r"\usepackage{sansmath}",
        r"\sansmath",
    ]
)

# Sizes
SMALL_SIZE = 30
MEDIUM_SIZE = SMALL_SIZE + 1
BIGGER_SIZE = MEDIUM_SIZE + 1
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsi ze of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
