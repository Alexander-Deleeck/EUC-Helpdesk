# UMAP and HDBSCAN

## UMAP

Hyperparameters control how it performs dimensionality reduction, but two of the most important are n_neighbors and n_components

### Parameters

- `n_components`: controls the dimensionality of the final embedded data after performing dimensionality reduction on the input data. E.g. if n_components is 2, the data will be embedded in 2D.
- `n_neighbors`: controls how UMAP balances local versus global structure in the data. This parameter controls the size of the neighborhood UMAP looks to learn the manifold structure, and so lower values of n_neighbors will focus more on the very local structure.
- `min_dist`: The minimum distance between points in the UMAP embedding.
- `metric`: The metric to use for the UMAP algorithm.



## HDBSCAN

### Parameters

- `min_cluster_size`: This controls the smallest grouping you want to consider as a cluster.
- `min_samples`: Controls how conservative the clustering is. The larger it is, the more points are discarded as noise/outliers (defaults to being equal to min_cluster_size if unspecified).
- `metric`: The metric to use for the HDBSCAN algorithm.


