import os
import time
import csv
import numpy as np
from MedcoupleNLogN import medcouple

DATA_DIR = "data"
OUTPUT_CSV = "medcouple_comparison.csv"
results = []

for fname in os.listdir(DATA_DIR):
    fpath = os.path.join(DATA_DIR, fname)
    
    #try:
    data = np.loadtxt(fpath)
    N = data.shape[0]
    
    # Statsmodels docs says implementation can't handle large arrays
    if N > 55000:
        continue

    start = time.perf_counter()
    my_mc = medcouple(data, use_fast=True)
    my_time = time.perf_counter() - start

    start = time.perf_counter()
    sm_mc = medcouple(data, use_fast=False)
    sm_time = time.perf_counter() - start

    results.append({
        "filename": fname,
        "N": N,
        "custom_medcouple": my_mc,
        "statsmodels_medcouple": sm_mc,
        "my_time": my_time,
        "sm_time": sm_time
    })

    '''except Exception as e:
        print(f"Failed to process {fname}: {e}")
        results.append({
            "filename": fname,
            "N": None,
            "custom_medcouple": "error",
            "statsmodels_medcouple": "error",
            "my_time": None,
            "sm_time": None
        })'''

with open(OUTPUT_CSV, mode="w", newline="") as csvfile:
    fieldnames = ["filename", "N", "custom_medcouple", "statsmodels_medcouple", "my_time", "sm_time"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
