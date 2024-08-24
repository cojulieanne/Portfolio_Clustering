from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS, DBSCAN, cluster_optics_dbscan
from sklearn_extra.cluster import KMedoids
import random
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from sklearn.base import clone
import yfinance as yf
from .utils import Portfolio



def kmeans_cluster(X, X_sharpe, port_test, k = 30, top = 1, scale = False, verbose = True, cluster_metrics = False, compare = None):

    X_train = X.T.copy()
    
    if not isinstance(X_train, np.ndarray):
        X_train = X_train.to_numpy()
        
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
    
    kmc = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_clusters = kmc.fit_predict(X_train)
    
    df = pd.DataFrame(kmeans_clusters, index=X.columns, columns=['Cluster'])
    df = pd.concat([df, X_sharpe], axis=1)
    portfolio = df.sort_values(by=['Cluster', 'Sharpe'], ascending=[True, False])
    portfolio = list(portfolio.groupby('Cluster').head(top).index)

    
    # Compute Sharpe ratio of the portfolio on the test period
    sharpe_kmeans = port_test.get_portf_sharpe(tick = portfolio)
    sortino_kmeans = port_test.get_portf_sortino(tick = portfolio)
    return_kmeans = port_test.get_portf_return(tick = portfolio)
    # print internal metrics

    sil = silhouette_score(X_train, kmeans_clusters)
    db = davies_bouldin_score(X_train, kmeans_clusters)
    ch = calinski_harabasz_score(X_train, kmeans_clusters)
    
    if verbose:
        print(f'Clustering Type        : KMeans\n'
              f'Clusters (preset)      : {k}\n'
              f'Portfolio Sharpe Ratio : {sharpe_kmeans:.4f}\n'
              f'Silhouette Score       : {sil:.4f}\n'
              f'Davies-Bouldin         : {db:.4f}\n'
              f'Calinski Harabasz      : {ch:.4f}'
             )

    output = {'portfolio': portfolio,
             'sharpe': sharpe_kmeans,
             'sortino': sortino_kmeans,
             'return': return_kmeans}
    
    if cluster_metrics:
        output['silhouette'] = sil
        output['db'] = db
        output['ch'] = ch

    if compare:
        attributes = ['min_mult', 'p25_mult', 'p50_mult', 'p75_mult', 'max_mult', 'mean_mult',
                      'min_preturn', 'p25_preturn', 'p50_preturn', 'p75_preturn', 'max_preturn', 'mean_preturn']
        comparisons = port_test.compare_portf_returns(compare)

        for i, attr in enumerate(attributes):
            output[attr] = comparisons[i]
    return output



def kmedoids_cluster(X, X_sharpe, port_test, k = 30, top = 1, scale = False, verbose = True, cluster_metrics = False, compare = None):

    X_train = X.T.copy()

    if not isinstance(X_train, np.ndarray):
        X_train = X_train.to_numpy()

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)


    kmedoids = KMedoids(n_clusters=k, method="pam", random_state=42)
    kmedoids_clusters = kmedoids.fit_predict(X_train)

    df = pd.DataFrame(kmedoids_clusters, index=X.columns, columns=['Cluster'])
    df = pd.concat([df, X_sharpe], axis=1)
    portfolio = df.sort_values(by=['Cluster', 'Sharpe'], ascending=[True, False])
    portfolio = list(portfolio.groupby('Cluster').head(top).index)

    
    # Compute Sharpe ratio of the portfolio
    sharpe_kmedoids = port_test.get_portf_sharpe(tick = portfolio)
    sortino_kmedoids = port_test.get_portf_sortino(tick = portfolio)
    return_kmedoids = port_test.get_portf_return(tick = portfolio)

    sil = silhouette_score(X_train, kmedoids_clusters)
    db = davies_bouldin_score(X_train, kmedoids_clusters)
    ch = calinski_harabasz_score(X_train, kmedoids_clusters)
    
    # print internal metrics
    if verbose:
        print(f'Clustering Type        : KMedoids\n'
              f'Clusters (preset)      : {k}\n'
              f'Portfolio Sharpe Ratio : {sharpe_kmedoids:.4f}\n'
              f'Silhouette Score       : {sil:.4f}\n'
              f'Davies-Bouldin         : {db:.4f}\n'
              f'Calinski Harabasz      : {ch:.4f}'
             )

    output = {'portfolio': portfolio,
             'sharpe': sharpe_kmedoids,
             'sortino': sortino_kmedoids,
             'return': return_kmedoids}
    
    if cluster_metrics:
        output['silhouette'] = sil
        output['db'] = db
        output['ch'] = ch

    if compare:
        attributes = ['min_mult', 'p25_mult', 'p50_mult', 'p75_mult', 'max_mult', 'mean_mult',
                      'min_preturn', 'p25_preturn', 'p50_preturn', 'p75_preturn', 'max_preturn', 'mean_preturn']
        comparisons = port_test.compare_portf_returns(compare)

        for i, attr in enumerate(attributes):
            output[attr] = comparisons[i]
    return output



def agglomerative_cluster(X, X_sharpe, port_test, method, k = 30, top = 1, scale = False, metric = 'euclidean', verbose = True, cluster_metrics = False, compare = None):

    X_train = X.T.copy()

    if not isinstance(X_train, np.ndarray):
        X_train = X_train.to_numpy()

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    agg = AgglomerativeClustering(n_clusters=k, linkage = method, metric = metric)
    groups = agg.fit_predict(X_train)

    df = pd.DataFrame(groups, index=X.columns, columns = ['Cluster'])
    df = pd.concat([df, X_sharpe], axis=1)
    portfolio = df.sort_values(by=['Cluster', 'Sharpe'], ascending=[True, False])
    portfolio = list(portfolio.groupby('Cluster').head(top).index)
    
    sharpe_agglo = port_test.get_portf_sharpe(tick = portfolio)
    sortino_agglo = port_test.get_portf_sortino(tick = portfolio)
    return_agglo = port_test.get_portf_return(tick = portfolio)
    
    sil = silhouette_score(X_train, groups)
    db = davies_bouldin_score(X_train, groups)
    ch = calinski_harabasz_score(X_train, groups)
    
    # Print output
    if verbose:
        print(f"Linkage Method            : {method}\n"
              f"Preset Clusters           : {k}\n"
              f"Sharpe Ratio of Portfolio : {sharpe_agglo:.4f}\n"
              f'Silhouette Score       : {sil:.4f}\n'
              f'Davies-Bouldin         : {db:.4f}\n'
              f'Calinski Harabasz      : {ch:.4f}'
        
        )

    output = {'portfolio': portfolio,
             'sharpe': sharpe_agglo,
             'sortino': sharpe_agglo,
             'return': sharpe_agglo}
    
    if cluster_metrics:
        output['silhouette'] = sil
        output['db'] = db
        output['ch'] = ch

    if compare:
        attributes = ['min_mult', 'p25_mult', 'p50_mult', 'p75_mult', 'max_mult', 'mean_mult',
                      'min_preturn', 'p25_preturn', 'p50_preturn', 'p75_preturn', 'max_preturn', 'mean_preturn']
        comparisons = port_test.compare_portf_returns(compare)

        for i, attr in enumerate(attributes):
            output[attr] = comparisons[i]
    return output

