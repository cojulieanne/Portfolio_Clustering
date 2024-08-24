import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS, DBSCAN, cluster_optics_dbscan
from sklearn_extra.cluster import KMedoids
from sklearn.base import clone
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from sklearn.base import clone

def pooled_within_ssd(X, y, centroids, dist):
    """Compute pooled within-cluster sum of squares around the cluster mean

    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    centroids : array
        Cluster centroids
    dist : callable
        Distance between two points. It should accept two arrays, each
        corresponding to the coordinates of each point

    Returns
    -------
    float
        Pooled within-cluster sum of squares around the cluster mean
    """
    # 1. Loop through centroids (as cluster mean)
    # 2. Get all points tagged to the centroids
    # 3. Calculate SSD per point vs cluster mean: (1/2n)*dist(x_i - centroid)^2
    # 4. Sum all SSD's for each cluster
    # 5. Sum all SSD's across clusters
    ssd_list = []
    for i, centroid in enumerate(centroids):
        cluster_ssd = 0
        cluster_points = X[y==i]
        n = len(cluster_points)
        # print(f'i: {i}, centroid: {centroid},\n cluster_points:\n {cluster_points}')
        for point in cluster_points:
            cluster_ssd += (dist(point, centroid)**2)/(2*n)
        # print(cluster_ssd)
        ssd_list.append(cluster_ssd)
    pooled_ssd = np.sum(ssd_list)
    # print(pooled_ssd)
    return pooled_ssd

def gen_realizations(X, b, random_state=None):
    """Generate b random realizations of X

    The realizations are drawn from a uniform distribution over the range of
    observed values for that feature.

    Parameters
    ---------
    X : array
        Design matrix with each row corresponding to a point
    b : int
        Number of realizations for the reference distribution
    random_state : int, default=None
        Determines random number generation for realizations

    Returns
    -------
    X_realizations : array
        random realizations with shape (b, X.shape[0], X.shape[1])
    """
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    rng = np.random.default_rng(random_state)
    nrows, ncols = X.shape
    return rng.uniform(
        np.tile(mins, (b, nrows, 1)),
        np.tile(maxs, (b, nrows, 1)),
        size=(b, nrows, ncols),
    )

def gap_statistic(X, y, centroids, dist, b, clusterer, random_state=None):
    """Compute the gap statistic

    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a data point
    y : array
        Class label of each point
    centroids : array
        Cluster centroids
    dist : callable
        Distance between two points. It should accept two arrays, each
        corresponding to the coordinates of each point
    b : int
        Number of realizations for the reference distribution
    clusterer : KMeans
        Clusterer object that will be used for clustering the reference
        realizations
    random_state : int, default=None
        Determines random number generation for realizations

    Returns
    -------
    gs : float
        Gap statistic
    gs_std : float
        Standard deviation of gap statistic
    """
    # Formula: gap_stat = (1/b)*sum( SSD(realizations) - SSD(original) )

    # 1. Get Realizations (using given gen_realizations function)
    X_refs = gen_realizations(X, b, random_state)

    # 2. Calculate Wki_ssd: SSD of original data (known y labels and centroids)
    Wk_ssd = pooled_within_ssd(X, y, centroids, dist)

    gaps_list = [None]*b
    # loop thru realizations
    for i, X_realization in enumerate(X_refs):
        # 3. Use Clusterer to predict y_realization values and centroids_realization
        # kmeans = clusterer(n_clusters=3, random_state=random_state, n_init="auto").fit(X_realization)
        kmeans = clusterer.fit(X_realization)
        y_realization = kmeans.labels_
        centroids_realization = kmeans.cluster_centers_
        # 4. Calculate Wki_ssd: SSD of realization
        Wki_ssd = pooled_within_ssd(X_realization, y_realization, centroids_realization, dist)

        # 5. Calculate gap for each realization, then store into gaps_list
        gap = np.log(Wki_ssd) - np.log(Wk_ssd)
        # print(gap)
        gaps_list[i] = gap

    # 6. Calculate Gap Statistic and Gap stat standard deviation
    gs = np.sum(gaps_list)/b
    # print(gap_stat)
    gs_std = np.std(gaps_list)
    # print(gap_stat_stdev)

    return gs, gs_std

def cluster_range(X, clusterer, k_start, k_stop):
    """Cluster X for different values of k

    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a data point
    clusterer : sklearn.base.ClusterMixin
        Perform clustering for different value of `k` using this model. It
        should have been initialized with the desired parameters
    k_start : int
        Perform k-means starting from this value of `k`
    k_stop : int
        Perform k-means up to this value of `k` (inclusive)

    Returns
    -------
    dict
        The value of each key is a list with elements corresponding to the
        value at `k`. The keys are:
            * `ys`: cluster labels
            * `centers`: cluster centroids
            * `inertias`: sum of squared distances to the cluster centroid
            * `chs`: Calinski-Harabasz indices
            * `scs`: silhouette coefficients
            * `dbs`: Davis-Bouldin indices
            * `gss`: gap statistics
            * `gssds`: standard deviations of gap statistics
    """
    random_state=1337
    ys = []
    centers = []
    inertias = []
    chs = []
    scs = []
    dbs = []
    gss = []
    gssds = []

    # perform for the range of k (start and stop) input parameters
    for k in range(k_start, k_stop+1):
        clusterer_k = clone(clusterer)

        # Change n_clusters parameter to k, then performing clustering
        clusterer_k.set_params(n_clusters=k, random_state=random_state)

        clusterer_k.fit_transform(X)

        # store clustering values for specific k
        y_k = clusterer_k.labels_                # labels of each point
        centers_k = clusterer_k.cluster_centers_  # centroids coordinates
        inertias_k = clusterer_k.inertia_         # sum of squared distances
        # print(f'ys: {y_k},\ncenters: {centers_k},\ninertias: {inertias_k}\n')

        # append data into lists
        ys.append(y_k)
        centers.append(centers_k)
        inertias.append(inertias_k)
        chs.append(calinski_harabasz_score(X, y_k))
        scs.append(silhouette_score(X, y_k))
        dbs.append(davies_bouldin_score(X, y_k))

        gs = gap_statistic(
            X,
            y_k,
            clusterer_k.cluster_centers_,
            euclidean,
            5,
            clone(clusterer).set_params(n_clusters=k),
            random_state=1234,
        )
        gss.append(gs[0])
        gssds.append(gs[1])

    dict = {
        'ys': ys,
        'centers': centers,
        'inertias': inertias,
        'chs': chs,
        'scs': scs,
        'dbs': dbs,
        'gss': gss,
        'gssds': gssds
    }

    return dict

def plot_internal(ax, inertias, chs, scs, dbs, gss, gssds):
    """Plot internal validation values"""
    ks = np.arange(2, len(inertias) + 2)
    ax.plot(ks, inertias, "-o", label="SSE")
    ax.plot(ks, chs, "-ro", label="CH")
    ax.set_xlabel("$k$")
    ax.set_ylabel("SSE/CH")
    lines, labels = ax.get_legend_handles_labels()
    ax2 = ax.twinx()
    ax2.errorbar(ks, gss, gssds, fmt="-go", label="Gap statistic")
    ax2.plot(ks, scs, "-ko", label="Silhouette coefficient")
    ax2.plot(ks, dbs, "-gs", label="DB")
    ax2.set_ylabel("Gap statistic/Silhouette/DB")
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right')
    return ax

def plot_internal_zoom_range(k_start, k_stop, results_range):
    """Plot internal validation values, but with individual y-axis legend values
       to zoom into fluctuations of specific metrics
    """
    # Plot Internal Validation Metrics in One Plot
    # Create the x-axis values starting from k_start
    k_values = list(range(k_start, k_stop + 1))

    # Create a figure
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot each data set on the same plot with different y-axes and colors
    ax1.plot(k_values, results_range["chs"], label='CH: Higher Better', color='red')
    ax1.set_ylabel('CH Score', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, which='both', axis='x')

    # Create a second y-axis for Silhouette score
    ax2 = ax1.twinx()
    ax2.plot(k_values, results_range["scs"], label='Silhouette: Higher Better', color='black')
    ax2.set_ylabel('Silhouette Score', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Create a third y-axis for Gap Statistic
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(k_values, results_range["gss"], label='Gap Statistic: Higher Better', color='purple')
    ax3.set_ylabel('Gap Statistic', color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')

    # Create a fourth y-axis for SSE
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 120))
    ax4.plot(k_values, results_range["inertias"], label='SSE: Lower Better', color='blue')
    ax4.set_ylabel('SSE', color='blue')
    ax4.tick_params(axis='y', labelcolor='blue')

    # Create a fifth y-axis for DB
    ax5 = ax1.twinx()
    ax5.spines['right'].set_position(('outward', 180))
    ax5.plot(k_values, results_range["dbs"], label='DB: Lower Better', color='green')
    ax5.set_ylabel('DB Score', color='green')
    ax5.tick_params(axis='y', labelcolor='green')

    # Add vertical grid lines for specific values
    for tick in range(5, k_stop, 5):
        ax1.axvline(x=tick, color='gray', linewidth=2.5, linestyle='--')

    # Set x-axis label and show all x-axis values
    ax1.set_xlabel('Clusters (k)')
    ax1.set_xticks(k_values)
    ax1.tick_params(axis='x', labelsize=8)

    # Set title
    plt.title('Cluster Validation Metrics')

    # Add legends for each y-axis
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    lines5, labels5 = ax5.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3 + lines4 + lines5, labels1 + labels2 + labels3 + labels4 + labels5, loc='center right')

    # Adjust layout
    plt.tight_layout()
    plt.show()

def calcCorr(X):
    col = X.columns
    cor1 = np.corrcoef(X, rowvar = 0)
    cor = pd.DataFrame(cor1, columns = col, index = col)
    return cor

def correlDist(corr):
    dist = 1 - corr
    return dist


def generate_date_combinations(date_index, train_period, test_period):
    
    date_index = pd.DatetimeIndex(date_index)
    combinations = []

    for start in date_index:
        end = start + pd.DateOffset(years=train_period)
        if end not in date_index:
            pos = date_index.searchsorted(end, side='right')
            if pos < len(date_index):
                end = date_index[pos]
            else:
                continue  # No valid end date available

        buy = end + pd.DateOffset(days=1)
        
        if buy not in date_index:
            pos = date_index.searchsorted(buy, side='right')
            if pos < len(date_index):
                buy = date_index[pos]
            else:
                continue  # No valid buy date available

        sell = buy + pd.DateOffset(years=test_period)
        
        if sell not in date_index:
            pos = date_index.searchsorted(sell, side='right')
            if pos < len(date_index):
                sell = date_index[pos]
            else:
                continue  # No valid sell date available

        combinations.append((start, end, buy, sell))

    combinations_df = pd.DataFrame(combinations, columns=['Start', 'End', 'Buy', 'Sell'])

    return combinations_df.astype(str)
