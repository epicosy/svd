from pylab import *
import numpy as np
from scipy import stats
from scipy.stats import norm
from statistics import median

fig_width = 20
fig_height = 10
fig_size = (fig_width, fig_height)


def zipf_log(labels, values):
    # sort values in descending order
    idx_sort = np.argsort(values)[::-1]

    # rearrange data
    tokens = np.array(labels)[idx_sort]
    counts = np.array(values)[idx_sort]

    # source https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-3-zipfs-law-data-visualisation-fc9eadda71e7
    ranks = arange(1, len(counts) + 1)
    idxs = argsort(-counts)
    frequencies = counts[idxs]
    # fig, ax = plt.figure()
    fig, ax = plt.subplots(figsize=fig_size)  # create figure and axes
    # Find intersection point within the range
    x, y = find_point(ranks, frequencies, (1000, 10000), (100, 1000))
    plt.vlines(x, 0, 10 ** 6, colors='r', linestyles='solid', label="intersection", linewidth=3)
    ax.annotate(x, xy=(x, 1), xytext=(0, 25), textcoords='offset points', rotation=0, va='bottom', ha='center',
                annotation_clip=False, fontsize=20)

    # plt.xlabel("Frequency rank of token", fontsize=26)
    # plt.ylabel("Absolute frequency of token", fontsize=26)
    plt.ylim(1, 10 ** 6)
    plt.xlim(1, 10 ** 6)
    loglog(ranks, frequencies, marker=".")
    plt.plot([1, frequencies[0]], [frequencies[0], 1], color='r')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    grid(True)

    for n in list(logspace(-0.5, log10(len(counts) - 2), 25).astype(int)):
        dummy = text(ranks[n], frequencies[n], " " + tokens[idxs[n]],
                     verticalalignment="bottom", fontdict={'size': 20},
                     horizontalalignment="left")
    plt.show()


def find_point(ranks, freqs, x_range, y_range):
    rank_freq = [(r, f) for r, f in zip(ranks, freqs) if x_range[0] < r < x_range[1] and y_range[0] < f < y_range[1]]
    position = round(median(value[0] for value in rank_freq)), round(median(value[1] for value in rank_freq))
    print(position)
    return position


def token_counts(tkn_list):
    tokens_mapping = {}

    for sentence in tkn_list:
        for token in sentence:
            if token in tokens_mapping:
                tokens_mapping[token] += 1
            else:
                tokens_mapping[token] = 1

    tokens, values = zip(*tokens_mapping.items())
    size = len(tokens_mapping)
    print(f"Unique tokens: {size}")

    # tokens, counts, size
    return tokens, values


def histogram(data, interval, bins_size: int = 100):
    bins = np.linspace(interval[0], interval[1], bins_size)
    fig, ax = plt.subplots(figsize=fig_size)  # create figure and axes
    n, x, _ = plt.hist(data, bins=bins, alpha=0.6, density=True, color=['green'])
    density = stats.gaussian_kde(data)
    ppf = norm(loc=np.average(data), scale=np.std(data)).ppf(0.95)
    y = density(x)
    plt.plot(x, y, color='red', label="kde", linewidth=3)
    plt.vlines(ppf, 0, 0.005, colors='blue', linestyles='solid', label='ppf')
    ax.annotate(round(ppf), xy=(ppf, min(y)), xytext=(0, 25), textcoords='offset points', rotation=0, va='bottom',
                ha='center', annotation_clip=False, fontsize=20)
    # plt.xlabel("Number of tokens",fontsize=26)
    # plt.ylabel("Frequency", fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='upper right', prop={'size': 20})
    plt.show()
