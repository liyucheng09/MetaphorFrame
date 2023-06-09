from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from typing import List

MARKERS = ['.', 'o', 'v', '^', '<', '>', 's', '*', '+', 'x']
COLORS = ['darkviolet', 'blue', 'saddlebrown', 'gold', 'green', 'cyan', 'hotpink', 'mediumseagreen', 'red', 'teal', 'tan', 'yellow', 'wheat', 'ivory',  'aqua', 'azure', 'beige', 'coral',]

def plotDimensionReduction(X, labels: List[str], figure_name, \
        plot_type = 'PCA', n_components = 2, legend_loc = 6,
        bbox_to_anchor = (1, 0.5), borderaxespad=0., no_legend = False, **kwargs):
    def pca(X, **kwargs):
        pca = PCA(n_components=n_components, **kwargs)
        X = pca.fit_transform(X)
        return X
    def tSNE(X, **kwargs):
        tsne = TSNE(n_components=n_components, **kwargs)
        X = tsne.fit_transform(X)
        return X
    
    plot_func = {
        'PCA': pca,
        'tSNE': tSNE
    }

    X = plot_func[plot_type](X, **kwargs)

    label_type = []
    for label in labels:
        if label not in label_type: label_type.append(label)

    label_type.sort()
    labels = np.array(labels)
    num_labels = len(label_type)
    if num_labels > len(COLORS):
        print(f"Colors not enough, have {len(COLORS)} colors, but got {num_labels} of labels.")
        return
    fig, ax = plt.subplots()
    markers, colors = MARKERS[:num_labels], COLORS[: num_labels]

    names = []
    for label in label_type:
        index = (labels == label)
        ax.scatter(X[index, 0], X[index, 1], c=colors[label_type.index(label)])
    if not no_legend:
        ax.legend(label_type, bbox_to_anchor=bbox_to_anchor, loc=legend_loc, borderaxespad=borderaxespad)
    # plt.show()
    plt.savefig(figure_name, bbox_inches = "tight", dpi=300.)
    print(f'Saved to {figure_name}!')
    plt.close()
    return X