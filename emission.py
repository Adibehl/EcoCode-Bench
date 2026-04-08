import numpy as np
import time
from codecarbon import EmissionsTracker

# 1. Setup the Experiment
def matrix_multiplication_benchmark():
    size = 2000
    
    # --- TEST 1: Standard Python List (Inefficient) ---
    print("Starting Standard Python Test...")
    tracker = EmissionsTracker(project_name="Standard_Python")
    tracker.start()
    
    # Creating lists manually
    list_a = [[1.0] * size for _ in range(size)]
    list_b = [[1.0] * size for _ in range(size)]
    # (We only do a partial operation because full list-mult is too slow)
    res = [sum(a*b for a,b in zip(A_row, B_col)) for A_row in list_a[:10] for B_col in zip(*list_b)]
    
    emissions_standard = tracker.stop()
    print(f"Standard Python Emissions: {emissions_standard:.6f} kg CO2\n")

    # --- TEST 2: NumPy (Optimized for Green Computing) ---
    print("Starting NumPy Test...")
    tracker = EmissionsTracker(project_name="NumPy_Optimization")
    tracker.start()
    
    # Creating arrays and multiplying
    array_a = np.ones((size, size))
    array_b = np.ones((size, size))
    res_np = np.dot(array_a, array_b)
    
    emissions_numpy = tracker.stop()
    print(f"NumPy Emissions: {emissions_numpy:.6f} kg CO2\n")

    # --- RESULTS ---
    print("--- RESEARCH SUMMARY ---")
    improvement = (emissions_standard - emissions_numpy) / emissions_standard * 100
    print(f"Optimization reduced Carbon Footprint by: {improvement:.2f}%")

if __name__ == "__main__":
    matrix_multiplication_benchmark()
