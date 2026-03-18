import numpy as np
import matplotlib.pyplot as plt


def plot_multipanel_timeseries(
    data, time_points=None, figsize=None, 
    var_names=None, trace_names=None, suptitle=None,
    ylabel='Value', xlabel='Time',
    share_y=False, tight_layout=True):
    """
    Plot a multipanel figure showing time traces for N x N variable pairs.
    Supports multiple traces per panel.
    
    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (N, N, M) for single trace, or (N, N, M, K) for K traces
        containing statistics for N x N pairs with M time points for each pair.
    
    time_points : array-like, optional
        Array of length M containing time values for x-axis.
        If None, uses indices 0 to M-1.
    
    figsize : tuple, optional
        Figure size as (width, height). If None, automatically scales
        based on N (default: (2*N, 2*N)).
    
    var_names : list of str, optional
        List of N variable names for labeling rows and columns.
        If None, uses 'Var 0', 'Var 1', etc.
    
    trace_names : list of str, optional
        List of K trace names for legend (only used if K > 1).
        If None, uses 'Trace 0', 'Trace 1', etc.
    
    suptitle : str, optional
        Overall figure title.
    
    ylabel : str, optional
        Label for y-axes (default: 'Value').
    
    xlabel : str, optional
        Label for x-axes (default: 'Time').
    
    share_y : bool or str, optional
        Whether to share y-axis scales. Options:
        - False: each panel has independent y-scale (default)
        - 'row': panels in same row share y-scale
        - 'col': panels in same column share y-scale
        - True or 'all': all panels share same y-scale
    
    tight_layout : bool, optional
        Whether to use tight_layout (default: True).
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    
    axes : numpy.ndarray
        Array of shape (N, N) containing the axes objects.
    
    Examples
    --------
    >>> # Single trace: 4x4 pairs with 100 time points each
    >>> N, M = 4, 100
    >>> data = np.random.randn(N, N, M).cumsum(axis=2)
    >>> time = np.linspace(0, 10, M)
    >>> fig, axes = plot_multipanel_timeseries(data, time_points=time,
    ...                                        var_names=['A', 'B', 'C', 'D'],
    ...                                        suptitle='Time Series Analysis')
    >>> plt.show()
    
    >>> # Multiple traces: 4x4 pairs with 100 time points and 3 traces
    >>> N, M, K = 4, 100, 3
    >>> data = np.random.randn(N, N, M, K).cumsum(axis=2)
    >>> fig, axes = plot_multipanel_timeseries(data, time_points=time,
    ...                                        var_names=['A', 'B', 'C', 'D'],
    ...                                        trace_names=['MST', 'Single', 'Complete'],
    ...                                        suptitle='Comparison')
    >>> plt.show()
    """
    
    # Validate input and determine if multiple traces
    if len(data.shape) == 3:
        # Single trace: (N, N, M)
        N1, N2, M = data.shape
        K = 1
        data = data[:, :, :, np.newaxis]  # Add trace dimension
    elif len(data.shape) == 4:
        # Multiple traces: (N, N, M, K)
        N1, N2, M, K = data.shape
    else:
        raise ValueError(f"Data must be 3D (N, N, M) or 4D (N, N, M, K) array, got shape {data.shape}")
    
    if N1 != N2:
        raise ValueError(f"First two dimensions must be equal, got {N1} x {N2}")
    
    N = N1
    
    # Set up time points
    if time_points is None:
        time_points = np.arange(M)
    elif len(time_points) != M:
        raise ValueError(f"time_points length ({len(time_points)}) must match "
                        f"data's last dimension ({M})")
    
    # Set up variable names
    if var_names is None:
        var_names = [f'Var {i}' for i in range(N)]
    elif len(var_names) != N:
        raise ValueError(f"var_names length ({len(var_names)}) must match N ({N})")
    
    # Set up trace names
    if trace_names is None:
        trace_names = [f'Trace {k}' for k in range(K)]
    elif len(trace_names) != K:
        raise ValueError(f"trace_names length ({len(trace_names)}) must match K ({K})")
    
    # Set up figure size
    if figsize is None:
        figsize = (2.5 * N, 2.5 * N)
    
    # Determine y-axis sharing
    sharey_param = False
    if share_y == 'row':
        sharey_param = 'row'
    elif share_y == 'col':
        sharey_param = 'col'
    elif share_y in [True, 'all']:
        sharey_param = 'all'
    
    # Create figure and subplots
    fig, axes = plt.subplots(N, N, figsize=figsize, sharex=True, sharey=sharey_param)
    
    # Handle case where N=1 (axes is not an array)
    if N == 1:
        axes = np.array([[axes]])
    
    # Plot data in each panel
    for i in range(N):
        for j in range(N):
            ax = axes[i, j]
            
            # Plot each trace
            for k in range(K):
                # Plot the time series
                line = ax.plot(time_points, data[i, j, :, k], linewidth=1.5, label=trace_names[k] if K > 1 else None)
                trace_color = line[0].get_color()  # Get the color of the plotted line
                
                # Calculate and plot the mean value of the trace
                mean_value = np.mean(data[i, j, :, k])
                ax.axhline(y=mean_value, color=trace_color, linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
                
                # Plot the final value (Pearson correlation endpoint)
                final_value = data[i, j, -1, k]
                final_time = time_points[-1]
                ax.plot(final_time, final_value, marker='D', markersize=8, 
                        markerfacecolor='none', markeredgecolor=trace_color, 
                        markeredgewidth=2, zorder=3)
            
            # Add horizontal line at y=0 (zero correlation baseline)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7, zorder=1)

            ax.grid(True, alpha=0.3)
            
            # Add column labels on top row
            ax.set_title(f'{var_names[i]} \n {var_names[j]}', fontsize=10, fontweight='bold')
            
            # Add legend if multiple traces (only on first panel to avoid clutter)
            if K > 1 and i == 0 and j == 0:
                ax.legend(fontsize=8, loc='best', framealpha=0.9)
            
            # Format ticks
            ax.tick_params(labelsize=8)
    
    # Add overall title if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.995)
    
    # Adjust layout
    if tight_layout:
        plt.tight_layout()
    
    return fig, axes