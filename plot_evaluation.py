import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("tkagg")


def plot_ratios(ratios, labels, scales, output_dir="outputs/evaluation/",
                title="Comparisson of target change ratio", filename="ratio_comparison.png"):
    plt.figure(figsize=(7,7))

    for ratio, label in zip(ratios, labels):
        plt.plot(ratio, scales, label=label, marker='.')

    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.xlabel("Target Change Ratio", fontsize=12)
    plt.ylabel("Scaling Factor", fontsize=12)
    plt.xlim(0,1)
    plt.ylim(scales[0], scales[-1])
    plt.savefig(output_dir + filename)
