# -*- encoding: utf8 -*-

# Author: Nobuhito Manome
# License: BSD 3 clause

import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

def savefig_line2D(data, xlim=None, ylim=None, xscale=None, yscale=None, markevery=1000, labels=None, xlabel=None, ylabel=None, title=None, path='output/', filename='figure', extension='png', is_show=True):
    '''
    Save a 2D line plot with customizable options.

    ----------
    Parameters
    ----------
    data : array-like, shape = [n_models, steps]
        The data to be plotted.
        n_models ≤ 20, representing the number of lines in the plot.
    xlim : list of float, optional (default=None)
        The limits for the x-axis. If None, the axis will auto-scale.
    ylim : list of float, optional (default=None)
        The limits for the y-axis. If None, the axis will auto-scale.
    xscale : str, optional (default=None)
        The scale for the x-axis. Possible values: 'linear', 'log', etc.
    yscale : str, optional (default=None)
        The scale for the y-axis. Possible values: 'linear', 'log', etc.
    markevery : int, optional (default=1000)
        Specifies the interval at which markers will appear on the plot.
    labels : list of str, optional (default=None)
        The labels for the plotted lines. If None, no legend is displayed.
    xlabel : str, optional (default=None)
        Label for the x-axis.
    ylabel : str, optional (default=None)
        Label for the y-axis.
    title : str, optional (default=None)
        Title of the plot.
    path : str, optional (default='output/')
        The directory path where the plot will be saved.
    filename : str, optional (default='figure')
        The name of the file to save the plot as.
    extension : str, optional (default='png')
        The file extension to save the plot as (e.g., 'png', 'jpg').
    is_show : bool, optional (default=True)
        If True, the plot will be displayed after saving.
    '''
    # Ensure the directory exists for saving the figure
    os.makedirs(path, exist_ok=True)

    # Create the figure with a specified size
    plt.figure(figsize=(10, 5.5))
    plt.rcParams['font.size'] = 20  # Set global font size for the plot
    plt.grid(color='gray', linestyle='--')  # Add a grid to the plot

    # Define style options for the plot lines
    linewidth = 4
    linestyle = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    markersize = 12
    marker = ['o', '^', 'v', '>', '<', 's', 'D', '*', '2', '1', '3', '4', 'p', 'h', 'H', 'd', '$◎$', 'x', '$∴$', '+']
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#393B79', '#6B6ECF', '#637939', '#B5CF6B', '#8C6D31', '#E7BA52', '#843C39', '#D6616B', '#7B4173', '#CE6DBD']

    # Set x and y axis limits if specified
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)

    # Set x and y axis scales if specified
    if xscale is not None: plt.xscale(xscale)
    if yscale is not None: plt.yscale(yscale)

    # Plot each data series with specific styling options
    for idx, elem in enumerate(data):
        plt.plot(
            elem,
            linewidth=linewidth,
            linestyle=linestyle[idx % len(linestyle)],
            markersize=markersize,
            markevery=int(markevery),
            marker=marker[idx % len(marker)],
            color=color[idx % len(color)]
        )

    # Set the plot's title and axis labels if provided
    if title is not None: plt.title(title, color='#000000')
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)

    # Add a legend if labels are provided
    if labels is not None: plt.legend(labels, loc='upper left', bbox_to_anchor=(1.02, 1.02))

    # Save the plot to the specified path with the given filename and extension
    plt.savefig(f'{path}{filename}.{extension}', bbox_inches='tight')

    # Display the plot if 'is_show' is True
    if is_show: plt.show()

    # Close the plot to release resources
    plt.close()

    print(f'Saved 2D line plot to {path}{filename}.{extension}')


def plot_bayesian_optimization_progress(
    z_vals,
    comparative_value,
    graph_title,
    graph_z_label,
    comparative_gwaucb1_alpha,
    comparative_gwaucb1_m,
    save_path
):
    '''
    Plot the progress of Bayesian optimization and save the plot.

    ----------
    Parameters
    ----------
    z_vals : list or np.array
        Values to plot (y-axis) representing the progress of the optimization.
    comparative_value : float
        The value to compare against (represented by a horizontal line).
    graph_title : str
        The title of the plot.
    graph_z_label : str
        The label for the y-axis of the plot.
    comparative_gwaucb1_alpha : float
        The α value for the GWA-UCB1 model.
    comparative_gwaucb1_m : float
        The m value for the GWA-UCB1 model.
    save_path : str
        The file path where the generated plot will be saved.

    '''
    # Initialize the plot with a specific figure size
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.size"] = 17  # Set global font size for the plot

    # Plot the optimization progress with markers and lines
    plt.plot(z_vals, marker='o', markersize=8, linestyle='-', alpha=1.0)

    # Set the plot's title and axis labels
    plt.title(graph_title)
    plt.xlabel('Iterations')
    plt.ylabel(graph_z_label)
    plt.grid(True)

    # Draw a horizontal line to represent the comparative value
    plt.axhline(y=comparative_value, color='black', linewidth=2, linestyle='--')

    # Annotate the horizontal line with the GWA-UCB1 model parameters
    annotation_text = f'GWA-UCB1\nα={comparative_gwaucb1_alpha:.2f}, m={comparative_gwaucb1_m:.2f}'
    plt.text(
        x=0, y=comparative_value,
        s=annotation_text,
        color='black', ha='center', va='center', fontsize=10
    )

    # Save the plot to the specified file path
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'Bayesian optimization progress saved to {save_path}')


def generate_3d_bayesian_optimization_plot(
    result,
    x_iters,
    z_vals,
    graph_title,
    graph_x_label,
    graph_y_label,
    graph_z_label,
    save_path,
    x_space,
    y_space
):
    '''
    Generate a 3D plot for Bayesian optimization results and save it.

    ----------
    Parameters
    ----------
    result : OptimizeResult
        The result object containing the best parameters found during the optimization.
    x_iters : list
        A list of iteration points (x, y) from the optimization process.
    z_vals : list
        The objective function values at the corresponding (x, y) points.
    graph_title : str
        The title of the plot.
    graph_x_label : str
        The label for the x-axis.
    graph_y_label : str
        The label for the y-axis.
    graph_z_label : str
        The label for the z-axis.
    save_path : str
        The file path where the generated plot will be saved.
    x_space : tuple
        The range for the x-axis (min, max).
    y_space : tuple
        The range for the y-axis (min, max).
    '''
    # Extract x and y values from the optimization iteration data
    x_vals = np.array([point[0] for point in x_iters])
    y_vals = np.array([point[1] for point in x_iters])

    # Create a meshgrid for interpolation in the x and y ranges
    x_range = np.linspace(x_space[0], x_space[1], 100)
    y_range = np.linspace(y_space[0], y_space[1], 100)
    X, Y = np.meshgrid(x_range, y_range)

    # Interpolate Z values using the griddata method based on the (x, y) points
    Z = griddata((x_vals, y_vals), z_vals, (X, Y), method='linear')

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    plt.rcParams["font.size"] = 12  # Set global font size for the plot
    ax = fig.add_subplot(111, projection='3d')

    # Set the initial view angle of the plot
    ax.view_init(elev=25, azim=135)

    # Plot the surface with a color map and transparency
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7, zorder=300)

    # Plot the optimization iterations (excluding the best parameters)
    ax.scatter(x_vals, y_vals, z_vals, color='#000000', s=50,
               label='Optimization iterations', alpha=0.7, zorder=200, marker='o')

    # Plot dashed lines for the best parameters found in the optimization
    ax.plot([result.x[0], result.x[0]], [y_space[0], y_space[1]], [0, 0], color='red', linestyle='--')
    ax.plot([x_space[0], x_space[1]], [result.x[1], result.x[1]], [0, 0], color='red', linestyle='--', label='Best parameters')

    # Replot the surface to ensure it stays on top
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.75)

    # Set plot labels, title, and legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 0.95), fontsize=11)
    ax.set_title(graph_title)
    ax.set_xlabel(graph_x_label)
    ax.set_ylabel(graph_y_label)
    ax.set_zlabel(graph_z_label)

    # Adjust the layout and save the figure
    fig.subplots_adjust(right=0.95)
    ax.set_zlim(0, None)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'3D Bayesian optimization plot saved to {save_path}')
