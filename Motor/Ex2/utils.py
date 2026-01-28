import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

# Interval for perturbations

ATTEMPTS_LIMIT = {'baseline': 160, 'interference': 300}
# ATTEMPTS_LIMIT = {'baseline': 20, 'interference': 30}

Interval_baseline_1 = (20, 80)  # baseline experiment
Interval_baseline_2 = (100, 160)
# Interval_baseline_1 = (5, 10)  # baseline experiment
# Interval_baseline_2 = (15, 20)

Interval_inference_1 = (20, 80) # interference experiment
Interval_inference_2 = (120, 180)
Interval_inference_3 = (220, 280)
# Interval_inference_1 = (5, 10) # interference experiment
# Interval_inference_2 = (15, 20)
# Interval_inference_3 = (25, 30)

OUTPUT_DIR = './output'   # Directory to save output files

# === Helper Functions ===
def get_subject_output_dir(subject_id: str, base_dir: str = OUTPUT_DIR) -> str:
    """Returns the output directory path for a specific subject."""
    return os.path.join(base_dir, f'S{subject_id}')

def normalize_angle(angle_degrees: float):
    """ Check if angle in degrees is within [-180, 180], if not change it """
    if angle_degrees < -180:
        angle_degrees += 360
    elif angle_degrees > 180:
        angle_degrees -= 360
    return angle_degrees


# === Saving and Reading Error Lists ===
def save_list_to_csv(lst: list, idx: list = [], header = ['attempt', 'angle_degrees'], 
                     output_dir: str = OUTPUT_DIR, file_name: str = 'error_angles.csv',
                     subject_id: str = None):
    """Saves a list of floats to a CSV file, handling NaN values appropriately.
    
    If subject_id is provided, files are saved to a subject-specific subdirectory.
    """
    # Use subject-specific subdirectory if subject_id is provided
    if subject_id:
        output_dir = get_subject_output_dir(subject_id, output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(os.path.join(output_dir, file_name), mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i, val in enumerate(lst):
                attempt_idx = idx[i] if idx else i+1
                if isinstance(val, float) and math.isnan(val):
                    writer.writerow([attempt_idx, ''])
                else:
                    writer.writerow([attempt_idx, val])
        print(f"Saved {len(lst)} error angles to {file_name}")
    except Exception as e:
        print(f"Error saving list to CSV: {e}")

def read_error_list(output_dir: str = OUTPUT_DIR, file_name: str = 'error_angles.csv',
                    subject_id: str = None):
    """ Read error angles (radians) from a CSV file.
    
    If subject_id is provided, files are read from a subject-specific subdirectory.
    """
    # Use subject-specific subdirectory if subject_id is provided
    if subject_id:
        output_dir = get_subject_output_dir(subject_id, output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    idxes = []
    results = []
    try:
        with open(os.path.join(output_dir, file_name), newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows:
                return idxes, results

            # skip header
            for row in rows[1:]:
                val = ''
                assert len(row) == 2, "Each row must have exactly two columns, one for attempt index and one for error angle value."
                val = row[1].strip()
                if val == '':
                    results.append(float('nan'))
                else:
                    try:
                        results.append(float(val))
                    except:
                        results.append(float('nan'))
                idxes.append(int(row[0].strip()))
    except Exception as e:
        print(f"Error reading error angles from CSV: {e}")
    return idxes, results


# === Plotting and Analysis ===

def plot_errors(idxes: list, lst: list, title: str = 'Error angles over attempts', exp_setup='baseline',
                file_name: str = '', output_dir: str = OUTPUT_DIR, subject_id: str = None):
    """ Plot error angles from list of floats (degrees), save to pngfile.
    
    If subject_id is provided, files are saved to a subject-specific subdirectory.
    """
    # Use subject-specific subdirectory if subject_id is provided
    if subject_id:
        output_dir = get_subject_output_dir(subject_id, output_dir)
    arr = np.array(lst, dtype=float)
    idx = np.array(idxes, dtype=int)
    valid = ~np.isnan(arr)

    plt.figure(figsize=(8, 4))
    plt.plot(idx[valid], arr[valid], marker='.')

    # highlight the experiment segments
    if exp_setup == 'baseline':
        for x in (Interval_baseline_1[0], Interval_baseline_1[1], Interval_baseline_2[0], Interval_baseline_2[1]):
            plt.axvline(x=x, color='red', linewidth=0.8, linestyle='--')

        ylim_top = plt.ylim()[1]
        plt.text((Interval_baseline_1[0] + Interval_baseline_1[1])/2 - 5, ylim_top * 0.9, 'sudden\nperturbation', color='red', va='top')
        # plt.text(Interval_baseline_1[1] + 0.3, ylim_top * 0.9, 'no\nperturbation', color='red', va='top')
        plt.text((Interval_baseline_2[0] + Interval_baseline_2[1])/2 - 5, ylim_top * 0.9, 'sudden\nperturbation', color='red', va='top')
        # plt.text(Interval_baseline_2[1] + 0.3, ylim_top * 0.9, 'no\nperturbation', color='red', va='top')
        # plt.xlim([0, Interval_baseline_2[1]+10])
    
    elif exp_setup == 'interference':
        for x in (Interval_inference_1[0], Interval_inference_1[1], Interval_inference_2[0], Interval_inference_2[1], Interval_inference_3[0], Interval_inference_3[1]):
            plt.axvline(x=x, color='red', linewidth=0.8, linestyle='--')

        ylim_top = plt.ylim()[1]
        plt.text((Interval_inference_1[0] + Interval_inference_1[1])/2 - 10, ylim_top * 0.9, 'sudden\nperturbation', color='red', va='top')
        # plt.text(Interval_inference_1[1] + 0.3, ylim_top * 0.9, 'no\nperturbation', color='red', va='top')
        plt.text((Interval_inference_2[0] + Interval_inference_2[1])/2 - 10, ylim_top * 0.9, 'interference\nperturbation', color='red', va='top')
        # plt.text(Interval_inference_2[1] + 0.3, ylim_top * 0.9, 'no\nperturbation', color='red', va='top')
        plt.text((Interval_inference_3[0] + Interval_inference_3[1])/2 - 10, ylim_top * 0.9, 'sudden\nperturbation', color='red', va='top')
        # plt.text(Interval_inference_3[1] + 0.3, ylim_top * 0.9, 'no\nperturbation', color='red', va='top')
        # plt.xlim([0, Interval_inference_3[1]+10])

    plt.xlabel('Attempt index')
    plt.ylabel('Error angle (degrees)')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Force x-axis ticks to be integers and align limits to indices
    from matplotlib.ticker import MaxNLocator
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(-0.5, max(max(idx), 0.5) + 0.5)

    if file_name:
        plt.savefig(os.path.join(output_dir, file_name))
        print(f"Saved plot to `{os.path.join(output_dir, file_name)}`")
    plt.show()


def compute_variability(lst: list):
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


def plot_errors_multi_subject(subject_data: list, title: str = 'Error angles over attempts', 
                               exp_setup='baseline', file_name: str = '', output_dir: str = OUTPUT_DIR):
    """Plot error angles from multiple subjects with different colors and average line.
    
    Args:
        subject_data: List of tuples, each containing (idxes, errors, subject_label) for one subject
                     e.g., [(idx1, err1, 'S1'), (idx2, err2, 'S2'), (idx3, err3, 'S3')]
        title: Plot title
        exp_setup: 'baseline' or 'interference'
        file_name: Output filename (if empty, plot is not saved)
        output_dir: Output directory
    """
    # Colors for individual subjects (with low opacity)
    subject_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    subject_alpha = 0.3  # Low opacity for individual subjects
    
    # Color for average line (high opacity)
    avg_color = '#d62728'  # Red
    avg_alpha = 1.0
    
    plt.figure(figsize=(10, 5))
    
    # Find common indices for averaging
    all_indices = set()
    for idxes, _, _ in subject_data:
        all_indices.update(idxes)
    common_indices = sorted(all_indices)
    
    # Prepare data for averaging
    avg_data = {}
    for idx in common_indices:
        avg_data[idx] = []
    
    # Plot each subject
    for i, (idxes, lst, label) in enumerate(subject_data):
        arr = np.array(lst, dtype=float)
        idx = np.array(idxes, dtype=int)
        valid = ~np.isnan(arr)
        
        color = subject_colors[i % len(subject_colors)]
        plt.plot(idx[valid], arr[valid], marker='.', color=color, alpha=subject_alpha, 
                 label=label, linewidth=1.5)
        
        # Collect data for averaging
        for j, index in enumerate(idxes):
            if not np.isnan(lst[j]):
                avg_data[index].append(lst[j])
    
    # Calculate and plot average
    avg_indices = []
    avg_values = []
    for idx in common_indices:
        if len(avg_data[idx]) > 0:
            avg_indices.append(idx)
            avg_values.append(np.mean(avg_data[idx]))
    
    avg_arr = np.array(avg_values, dtype=float)
    avg_idx = np.array(avg_indices, dtype=int)
    plt.plot(avg_idx, avg_arr, marker='o', color=avg_color, alpha=avg_alpha, 
             label='Average', linewidth=2.5, markersize=4)
    
    # Add experiment segment markers
    if exp_setup == 'baseline':
        for x in (Interval_baseline_1[0], Interval_baseline_1[1], Interval_baseline_2[0], Interval_baseline_2[1]):
            plt.axvline(x=x, color='red', linewidth=0.8, linestyle='--')

        ylim_top = plt.ylim()[1]
        plt.text((Interval_baseline_1[0] + Interval_baseline_1[1])/2 - 5, ylim_top * 0.9, 'sudden\nperturbation', color='red', va='top')
        plt.text((Interval_baseline_2[0] + Interval_baseline_2[1])/2 - 5, ylim_top * 0.9, 'sudden\nperturbation', color='red', va='top')
    
    elif exp_setup == 'interference':
        for x in (Interval_inference_1[0], Interval_inference_1[1], Interval_inference_2[0], Interval_inference_2[1], Interval_inference_3[0], Interval_inference_3[1]):
            plt.axvline(x=x, color='red', linewidth=0.8, linestyle='--')

        ylim_top = plt.ylim()[1]
        plt.text((Interval_inference_1[0] + Interval_inference_1[1])/2 - 10, ylim_top * 0.9, 'sudden\nperturbation', color='red', va='top')
        plt.text((Interval_inference_2[0] + Interval_inference_2[1])/2 - 10, ylim_top * 0.9, 'interference\nperturbation', color='red', va='top')
        plt.text((Interval_inference_3[0] + Interval_inference_3[1])/2 - 10, ylim_top * 0.9, 'sudden\nperturbation', color='red', va='top')

    plt.xlabel('Attempt index')
    plt.ylabel('Error angle (degrees)')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Force x-axis ticks to be integers
    from matplotlib.ticker import MaxNLocator
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(-0.5, max(max(common_indices), 0.5) + 0.5)

    if file_name:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, file_name))
        print(f"Saved plot to `{os.path.join(output_dir, file_name)}`")
    plt.show()