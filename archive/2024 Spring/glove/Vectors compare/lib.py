from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

    
def plot_dendrogram(lnk, labels):
    plt.figure(figsize=(25, 300))
    plt.vlines(0.2, 0, 10000)
    plt.vlines(0.3, 0, 10000)
    plt.vlines(0.4, 0, 10000)
    
    dendrogram(
        lnk,
        leaf_font_size=20., 
        orientation='left',
        leaf_label_func=lambda v: labels[v]
    )
    plt.show()
    
def tree_depth(vectors):
    leafs = vectors.shape[0]
    root_to_depth = {}
    root_to_level = {}
    def get_depth(index):
        return 1 if index < leafs else root_to_depth[index]
    def get_level(index):
        return 0 if index < leafs else root_to_level[index]
    max_depth = 1
    cnt = leafs
    for link in linkage(vectors, metric='cosine', optimal_ordering=False):
        depth0 = get_depth(link[0])
        depth1 = get_depth(link[1])
        if depth0 > depth1:
            root_to_depth[cnt] = depth0 + (link[2] > get_level(link[0]))
        else:
            root_to_depth[cnt] = depth1 + (link[2] > get_level(link[1]))
        root_to_level[cnt] = link[2]
        max_depth = max(max_depth, root_to_depth[cnt])
        cnt += 1
    return max_depth
    
def leaf_metrics(vectors):
    leaf_depths = linkage(vectors, metric='cosine', optimal_ordering=True)[:, 2]
    print('leaf depth min', np.mean(leaf_depths))
    print('leaf depth median', np.median(leaf_depths))
    print('max leaf depth', max(leaf_depths))
    leaf_diff = np.absolute(np.diff(leaf_depths))
    print()
    print('leaf difference mean', np.mean(leaf_diff))
    print('leaf difference median', np.median(leaf_diff))
    print('max leaf difference', max(leaf_diff))
    print()
    print('tree depth', tree_depth(vectors))
    
def load_vectors(path):
    vdf = pd.read_csv(path, sep=' ',header=None).set_index(0)
    vdf = vdf.reset_index(drop=True).iloc[:-1]
    vdf.columns=list(range(vdf.shape[1]))
    vdf.index.name='index'
    return vdf.drop(columns=[(vdf.shape[1] / 2 - 1), vdf.shape[1] - 1])
