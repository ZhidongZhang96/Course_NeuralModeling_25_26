import os
import csv
import math
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from numpy import floating

# Interval for perturbations
Interval_General = (5, 10)
Interval_Sudden = (15, 20)
# Interval_General = (40, 80)
# Interval_Sudden = (120, 160)

OUTPUT_DIR = 'output'   # Directory to save output files

# === Saving and Reading Error Lists ===
def save_list_to_csv(lst: list, output_dir: str = OUTPUT_DIR, file_name: str = 'error_angles.csv'):
    """Saves a list of floats to a CSV file, handling NaN values appropriately."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(os.path.join(output_dir, file_name), mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['attempt', 'error_angle_degrees'])
            for i, val in enumerate(lst):
                if isinstance(val, float) and math.isnan(val):
                    writer.writerow([i, ''])
                else:
                    writer.writerow([i, val])
        print(f"Saved {len(lst)} error angles to {file_name}")
    except Exception as e:
        print(f"Error saving list to CSV: {e}")

def read_error_list(output_dir: str = OUTPUT_DIR, file_name: str = 'error_angles.csv'):
    """ Read error angles (radians) from a CSV file """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []
    try:
        with open(os.path.join(output_dir, file_name), newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows:
                return results

            start_row = 0
            first = rows[0]

            # skip header
            for row in rows[1:]:
                val = ''
                if len(row) >= 2:
                    val = row[1].strip()
                elif len(row) == 1:
                    val = row[0].strip()
                if val == '':
                    results.append(float('nan'))
                else:
                    try:
                        results.append(float(val))
                    except:
                        results.append(float('nan'))
    except Exception as e:
        print(f"Error reading error angles from CSV: {e}")
    return results


# === Plotting and Analysis ===

def plot_errors(lst: list, file_name: str = '', output_dir: str = OUTPUT_DIR):
    """ Plot error angles from list of floats (degrees), save to pngfile """
    arr = np.array(lst, dtype=float)
    idx = np.arange(len(arr))
    valid = ~np.isnan(arr)

    plt.figure(figsize=(8, 4))
    plt.plot(idx[valid], arr[valid], marker='o', linestyle='--')

    # highlight the experiment segments
    for x in (Interval_General[0], Interval_General[1], Interval_Sudden[0], Interval_Sudden[1]):
        plt.axvline(x=x, color='red', linewidth=0.8)

    ylim_top = plt.ylim()[1]
    plt.text(Interval_General[0] + 0.3, ylim_top * 0.9, 'gradual\nperturbation', color='red', va='top')
    plt.text(Interval_General[1] + 0.3, ylim_top * 0.9, 'no\nperturbation', color='red', va='top')
    plt.text(Interval_Sudden[0] + 0.3, ylim_top * 0.9, 'sudden\nperturbation', color='red', va='top')
    plt.text(Interval_Sudden[1] + 0.3, ylim_top * 0.9, 'no\nperturbation', color='red', va='top')

    plt.xlabel('Attempt index')
    plt.ylabel('Error angle (degrees)')
    plt.title('Error angles over attempts')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Force x-axis ticks to be integers and align limits to indices
    from matplotlib.ticker import MaxNLocator
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(-0.5, max(len(arr) - 0.5, 0.5))

    if file_name:
        plt.savefig(os.path.join(output_dir, file_name))
        print(f"Saved plot to `{os.path.join(output_dir, file_name)}`")
    plt.show()


def compute_variability(lst: list) -> float | floating[Any]:
    """ Compute variability (standard deviation) of error angles, ignoring NaNs

    :param lst: List of error angles (**degrees**)
    :return: Standard deviation of valid error angles, or NaN if none are valid
    """
    arr = np.array(lst, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return float('nan')
    variability = np.std(valid)

    return variability