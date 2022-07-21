import leidenalg as la

def leiden(G, cells,
           resolution=1, seed=0, n_iteration=2,
           **kwargs,
          ):
    """cells are in order of the Graph node order -- whatever it is
    """
    partition = la.find_partition(G, 
                                  la.RBConfigurationVertexPartition, # modularity with resolution
                                  resolution_parameter=resolution, seed=seed, n_iterations=n_iteration, **kwargs)
    # get cluster labels from partition
    labels = [0]*(len(cells)) 
    for i, cluster in enumerate(partition):
        for element in cluster:
            labels[element] = i+1
    return labels


