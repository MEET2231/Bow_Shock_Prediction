"""
Demonstration script for plot_utils module

This script demonstrates how to use the plot_utils module to create visualizations
of MMS region classifications and spectrograms.
"""

import numpy as np
import matplotlib.pyplot as plt
import plot_utils

def demo_plot_utils():
    """
    Demonstrate the functionality of plot_utils module with sample data.
    """
    print("Demonstrating plot_utils module functionality...")
    
    # Create sample data
    n_samples = 1000
    
    # Sample epoch array (just sequential numbers for demo)
    epoch = np.arange(n_samples)
    
    # Sample classification labels (random for demonstration)
    np.random.seed(42)  # For reproducibility
    
    # Generate somewhat realistic labels with some structure
    # (not completely random to simulate real magnetosphere transitions)
    labels = np.zeros(n_samples, dtype=int)
    
    # Start with solar wind (0)
    current_label = 0
    for i in range(n_samples):
        # Small chance to change region
        if np.random.random() < 0.01:  # 1% chance to change
            current_label = np.random.randint(0, 4)  # Choose new region
        labels[i] = current_label
    
    # Add some magnetosheath (2) and magnetosphere (3) regions
    labels[300:350] = 2  # Magnetosheath region
    labels[350:400] = 3  # Magnetosphere region
    labels[700:750] = 2  # Another magnetosheath region
    
    print(f"Generated {n_samples} sample data points")
    print("Class distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        print(f"  Class {int(lbl)}: {cnt} samples ({cnt/len(labels):.2%})")
    
    # Demo 1: Simple region classification plot
    print("\nGenerating simple region classification plot...")
    plot_utils.plot_region_classifications(epoch, labels)
    
    # Demo 2: Generate sample energy spectrogram
    print("\nGenerating sample energy spectrogram...")
    n_energies = 32
    energy_values = np.logspace(1, 4, n_energies)  # 10 eV to 10 keV
    
    # Create a synthetic spectrogram (would be real data in actual use)
    spectrogram = np.zeros((n_samples, n_energies))
    
    # Add some structure to the spectrogram based on region
    for i in range(n_samples):
        # Base spectrum depends on region
        if labels[i] == 0:  # Solar wind
            # Solar wind: peaked at mid energies
            peak = np.exp(-0.5 * ((np.arange(n_energies) - 12) / 4)**2)
            spectrogram[i] = peak * (10 + np.random.random(n_energies) * 2)
        elif labels[i] == 1:  # Ion foreshock
            # Foreshock: broader energy distribution
            peak = np.exp(-0.5 * ((np.arange(n_energies) - 15) / 8)**2)
            spectrogram[i] = peak * (5 + np.random.random(n_energies) * 5)
        elif labels[i] == 2:  # Magnetosheath
            # Magnetosheath: shifted to higher energies
            peak = np.exp(-0.5 * ((np.arange(n_energies) - 20) / 6)**2)
            spectrogram[i] = peak * (20 + np.random.random(n_energies) * 10)
        elif labels[i] == 3:  # Magnetosphere
            # Magnetosphere: high energies
            peak = np.exp(-0.5 * ((np.arange(n_energies) - 25) / 5)**2)
            spectrogram[i] = peak * (15 + np.random.random(n_energies) * 5)
    
    # Demo 3: Combined plot with energy spectrogram and region classifications
    print("\nGenerating combined plot with energy spectrogram and region classifications...")
    plot_utils.plot_combined_spectrograms(epoch, labels, spectrogram, energy_values)
    
    print("\nDemonstration completed.")

if __name__ == "__main__":
    demo_plot_utils()
