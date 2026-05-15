import csv
import os
from datetime import datetime
from collections import defaultdict

HISTORY_CACHE = None
HISTORY_FILE = "results/history.csv"


def make_results_folder():
    os.makedirs("results", exist_ok=True)


def load_history():
    global HISTORY_CACHE

    if HISTORY_CACHE is not None:
        return HISTORY_CACHE

    make_results_folder()

    if not os.path.exists(HISTORY_FILE):
        HISTORY_CACHE = []
        return HISTORY_CACHE

    history = []

    with open(HISTORY_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            row["image_height"] = int(row["image_height"])
            row["image_width"] = int(row["image_width"])
            row["kernel_size"] = int(row["kernel_size"])
            row["runtime"] = float(row["runtime"])
            history.append(row)

    HISTORY_CACHE = history
    return HISTORY_CACHE


def save_result(operation, image_height, image_width, kernel_size, method, runtime):
    global HISTORY_CACHE

    make_results_folder()

    file_exists = os.path.exists(HISTORY_FILE)

    new_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "operation": operation,
        "image_height": image_height,
        "image_width": image_width,
        "kernel_size": kernel_size,
        "method": method,
        "runtime": runtime
    }

    with open(HISTORY_FILE, "a", newline="") as file:
        fieldnames = [
            "timestamp",
            "operation",
            "image_height",
            "image_width",
            "kernel_size",
            "method",
            "runtime"
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(new_row)

    if HISTORY_CACHE is not None:
        HISTORY_CACHE.append(new_row)


def find_best_method_from_history(operation, image_height, image_width, kernel_size):

    history = load_history()

    similar_runs = []

    for row in history:
        same_operation = row["operation"] == operation
        similar_height = abs(row["image_height"] - image_height) <= 150
        similar_width = abs(row["image_width"] - image_width) <= 150
        similar_kernel = abs(row["kernel_size"] - kernel_size) <= 3

        if same_operation and similar_height and similar_width and similar_kernel:
            similar_runs.append(row)

    if len(similar_runs) == 0:
        return None

    # Average runtime by method
    method_times = defaultdict(list)

    for row in similar_runs:
        method_times[row["method"]].append(row["runtime"])

    avg_method_times = {
        method: sum(times) / len(times)
        for method, times in method_times.items()
    }

    best_method = min(avg_method_times, key=avg_method_times.get)

    return best_method