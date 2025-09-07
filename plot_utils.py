"""
Plotting utilities for MMS Magnetosphere Region Classification

This module contains functions for visualizing MMS data and classification results.
It provides specialized plots for displaying magnetosphere region classifications
with appropriate color coding and legends.

Functions:
    plot_region_classifications: Visualizes classified regions with color coding
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


def plot_region_classifications(epoch, label, figsize=(12, 2), show=True):
    """
    Creates a color-coded plot of magnetosphere region classifications.
    
    Parameters:
    -----------
    epoch : array-like
        Time values for the x-axis
    label : array-like
        Classification labels (0-3) corresponding to each epoch
    figsize : tuple, optional
        Figure size as (width, height) in inches, default (12, 2)
    show : bool, optional
        If True, calls plt.show() to display the plot, default True
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    ax : matplotlib.axes.Axes
        The axes object containing the plot
        
    Notes:
    ------
    Color coding for regions:
    - Blue (0): Solar Wind (SW)
    - Black (1): Ion Foreshock (1/F)
    - Yellow (2): Magnetosheath (MSH)
    - Red (3): Magnetosphere (MSP)
    """
    # Define color mapping: 0=Blue, 1=Black, 2=Yellow, 3=Red
    class_colors = ['blue', 'black', 'yellow', 'red']
    cmap = mcolors.ListedColormap(class_colors)
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=figsize)
    
    # Show as an image: shape (1, N)
    im = ax.imshow(label[np.newaxis, :], aspect='auto', cmap=cmap, norm=norm)
    ax.set_yticks([])
    ax.set_xlabel('Epoch')
    ax.set_title('CNN Class Output (Blue: SW, Black: 1/F, Yellow: MSH, Red: MSP)')

    # Custom legend
    legend_elements = [
        Patch(facecolor='blue', edgecolor='k', label='0/SW'),
        Patch(facecolor='black', edgecolor='k', label='1/F'),
        Patch(facecolor='yellow', edgecolor='k', label='2/MSH'),
        Patch(facecolor='red', edgecolor='k', label='3/MSP')
    ]
    ax.legend(handles=legend_elements, loc='upper right', ncol=4, bbox_to_anchor=(1, 1.5))
    
    plt.tight_layout()
    
    if show:
        plt.show()
        
    return fig, ax


def plot_combined_spectrograms(epoch, label, energy_spectrogram=None, energy_values=None, figsize=(12, 6)):
    """
    Creates a combined plot with energy spectrogram and region classifications.
    
    Parameters:
    -----------
    epoch : array-like
        Time values for the x-axis
    label : array-like
        Classification labels (0-3) corresponding to each epoch
    energy_spectrogram : array-like, optional
        Energy spectrogram data, shape (n_times, n_energies)
    energy_values : array-like, optional
        Energy bin values in eV
    figsize : tuple, optional
        Figure size as (width, height) in inches, default (12, 6)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plots
    axs : list of matplotlib.axes.Axes
        The axes objects containing the plots
    """
    fig, axs = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Top subplot: Energy spectrogram (if provided)
    if energy_spectrogram is not None:
        # Apply log10 scale with handling for zeros/negatives
        min_nonzero = np.min(energy_spectrogram[energy_spectrogram > 0]) if np.any(energy_spectrogram > 0) else 1.0
        spectrogram_log = np.log10(np.maximum(energy_spectrogram, min_nonzero/10))
        
        im1 = axs[0].imshow(spectrogram_log.T, 
                          aspect='auto', 
                          origin='lower',
                          extent=[0, len(epoch), 0, energy_spectrogram.shape[1]],
                          cmap='plasma')
                          
        # Add colorbar
        cbar = fig.colorbar(im1, ax=axs[0], orientation='vertical', pad=0.01)
        cbar.set_label('Log10(counts)')
        
        # Add energy labels if available
        if energy_values is not None and len(energy_values) > 0:
            num_ticks = min(5, len(energy_values))
            tick_positions = np.linspace(0, energy_spectrogram.shape[1]-1, num_ticks).astype(int)
            
            # Create tick labels with proper units
            tick_labels = []
            for pos in tick_positions:
                if pos < len(energy_values):
                    value = energy_values[pos]
                    if value >= 1000:
                        tick_labels.append(f"{value/1000:.1f} keV")
                    else:
                        tick_labels.append(f"{value:.0f} eV")
                else:
                    tick_labels.append("")
            
            axs[0].set_yticks(tick_positions)
            axs[0].set_yticklabels(tick_labels)
            
        axs[0].set_ylabel('Energy')
        axs[0].set_title('Energy Spectrogram (log10 scale)')
    else:
        axs[0].set_visible(False)
    
    # Bottom subplot: Region classifications
    # Define color mapping: 0=Blue, 1=Black, 2=Yellow, 3=Red
    class_colors = ['blue', 'black', 'yellow', 'red']
    cmap = mcolors.ListedColormap(class_colors)
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Show as an image: shape (1, N)
    im2 = axs[1].imshow(label[np.newaxis, :], aspect='auto', cmap=cmap, norm=norm)
    axs[1].set_yticks([])
    axs[1].set_xlabel('Epoch')
    axs[1].set_title('CNN Class Output (Blue: SW, Black: 1/F, Yellow: MSH, Red: MSP)')
    
    # Custom legend for the bottom plot
    legend_elements = [
        Patch(facecolor='blue', edgecolor='k', label='0/SW'),
        Patch(facecolor='black', edgecolor='k', label='1/F'),
        Patch(facecolor='yellow', edgecolor='k', label='2/MSH'),
        Patch(facecolor='red', edgecolor='k', label='3/MSP')
    ]
    axs[1].legend(handles=legend_elements, loc='upper right', ncol=4, bbox_to_anchor=(1, 1.6))
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # Adjust spacing between subplots
    plt.show()
    
    return fig, axs
