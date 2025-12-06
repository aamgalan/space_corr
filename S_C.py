def get_space_corr(
    x: np.ndarray, y: np.ndarray, 
    merge_order: List[List[int]], 
    rescaled_scalar: bool = True
) -> Union[float, List[float]]:
    """Compute spatial correlation statistic.
    
    This is the novel extension of SkienaA (for spatial autocorrelation) that tracks cross-products instead of squared deviations.
    
    Parameters
    ----------
    x : np.ndarray
        Array of N real numbers (first variable)
    y : np.ndarray
        Array of N real numbers (second variable)
    merge_order : list
        Legitimate merge order defined on the N vertices
    rescaled_scalar : bool, optional
        Whether to return single value in [-1, 1] (default: True)
        
    Returns
    -------
    float or list
        spatial correlation statistic value(s)
    """
    len_V = len(x)
    # print(x.shape, y.shape)
    
    if len(y) != len_V:
        raise ValueError(f"x and y must have same length. Got {len_V} and {len(y)}")
    
    cross_sum = 0.0
    var_sum_x = 0.0
    var_sum_y = 0.0
    cross_running = [0.0]
    correlation_running = [0.0]
    
    ds = DisjointSet()
    
    # Initialize cluster information for all necessary variables
    cluster_info = []
    for i in range(len_V):
        cluster_info.append({
            "size": 1,
            "mean_x": x[i],
            "mean_y": y[i],
            # "sum_of_sq_x": 0.0,
            # "sum_of_sq_y": 0.0,
            # "cross_product": 0.0, # this is really the sum of cross products of x and y 
            # "cross_product_normalized": 0.0
        })

    # Process merge order provided line by line
    for vertex_1, vertex_2 in merge_order:
        # TODO verify that this cluster ID fetching logic is working correctly. 
        cluster_1 = ds.find(vertex_1)
        cluster_2 = ds.find(vertex_2)
        ds.union(cluster_1, cluster_2)

        cluster_info_1 = cluster_info[cluster_1]
        cluster_info_2 = cluster_info[cluster_2]

        # Compute merged cluster means for both variables
        n1 = cluster_info_1["size"]
        n2 = cluster_info_2["size"]
        n12 = n1 + n2

        # combined cluster means for X and Y 
        mean_x_12 = (n1 * cluster_info_1["mean_x"] + n2 * cluster_info_2["mean_x"]) / n12
        mean_y_12 = (n1 * cluster_info_1["mean_y"] + n2 * cluster_info_2["mean_y"]) / n12

        # changes in the means of clusters for X and Y 
        delta_x_1 = mean_x_12 - cluster_info_1["mean_x"]
        delta_x_2 = mean_x_12 - cluster_info_2["mean_x"]
        delta_y_1 = mean_y_12 - cluster_info_1["mean_y"]
        delta_y_2 = mean_y_12 - cluster_info_2["mean_y"]

        # Update cross-products (OUR MAIN INNOVATION)
        # This measures how X and Y co-vary spatially within merged clusters

        # cluster_info_1["sum_of_sq_x"] += cluster_SS_x_change_1 + cluster_SS_x_change_2
        # cluster_info_1["sum_of_sq_y"] += cluster_SS_y_change_1 + cluster_SS_y_change_2
        
        # cluster_info_1["cross_product"] += delta_cluster_cross_1
        # cluster_info_2["cross_product"] += delta_cluster_cross_2
        cross_sum += n1 * delta_x_1 * delta_y_1 + n2 * delta_x_2 * delta_y_2
        var_sum_x += n1 * delta_x_1**2 + n2 * delta_x_2**2
        var_sum_y += n1 * delta_y_1**2 + n2 * delta_y_2**2

        # Update cluster info
        cluster_info_1["size"] = n12
        cluster_info_2["size"] = n12
        cluster_info_1["mean_x"] = mean_x_12
        cluster_info_2["mean_x"] = mean_x_12
        cluster_info_1["mean_y"] = mean_y_12
        cluster_info_2["mean_y"] = mean_y_12

        cross_running.append(cross_sum)
        correlation_running.append(cross_sum/math.sqrt(var_sum_x*var_sum_y))
    """
    if rescaled_scalar:
        # Compute total cross-product for normalization
        mean_x = np.mean(z1)
        mean_y = np.mean(z2)
        total_cross = np.sum((z1 - mean_x) * (z2 - mean_y))
        
        if abs(total_cross) < 1e-10:
            warnings.warn("Total cross-product near zero, returning 0")
            return 0.0
        
        space_corr_mean = np.mean(cross_running) / total_cross
        return space_corr_mean
        # return 2.0 * (1.0 - space_corr_mean) - 1.0
    """
    return correlation_running


def compute_space_corr_matrix(
    variables: np.ndarray, 
    merge_order: List[List[int]], 
    verbose: bool = False
) -> np.ndarray:
    """Compute pairwise spatial correlation matrix for multiple variables.
    
    Parameters
    ----------
    variables : np.ndarray
        Array of shape (n_points, n_variables) containing variable values
    merge_order : list
        Merge order computed once and reused for all variable pairs
    verbose : bool, optional
        Whether to print progress (default: False)
        
    Returns
    -------
    np.ndarray
        space_corr matrix of shape (n_variables, n_variables)
    """
    n_vars = variables.shape[1]
    n_points = variables.shape[0]
    space_corr_matrix = np.zeros((n_vars, n_vars, n_points))

    # SKIPPING THE DIAGONAL ENTRIES FOR NOW
    """
    # Diagonal is autocorrelation (SA)
    for i in range(n_vars):
        space_corr_matrix[i, i] = 0.
        # space_corr_matrix[i, i] = SkienaA(variables[:, i], merge_order)
        if verbose and (i + 1) % 10 == 0:
            print(f"Computed autocorrelation for {i+1}/{n_vars} variables")
    """ 
    # Off-diagonal is spatial correlation (space_corr)
    total_pairs = n_vars * (n_vars - 1) // 2
    pair_count = 0
    
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            space_corr = get_space_corr(variables[:, i], variables[:, j], merge_order)
            # TODO change to keeping and reporting the entire list of correlations
            space_corr_matrix[i, j] = space_corr # np.array([scc[-1], np.mean(scc), np.mean(scc)/scc[-1]]) # scc
            space_corr_matrix[j, i] = space_corr # np.array([scc[-1], np.mean(scc), np.mean(scc)/scc[-1]]) # scc
            pair_count += 1
            
            if verbose and pair_count % 100 == 0:
                print(f"Computed {pair_count}/{total_pairs} spatial correlations "
                      f"({100*pair_count/total_pairs:.1f}%)")
    
    return space_corr_matrix
