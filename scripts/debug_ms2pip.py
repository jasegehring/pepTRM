"""
Debug script to test MS2PIP for NaN/inf values on specific peptide lengths.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from src.data.ms2pip_dataset import MS2PIPSyntheticDataset

def test_peptide_length(length: int, num_peptides: int = 1000):
    """
    Tests a given peptide length for NaN/inf issues from MS2PIP.
    """
    print(f"Testing {num_peptides} peptides of length {length}...")

    # We need to instantiate the dataset to use its methods
    dataset = MS2PIPSyntheticDataset(min_length=length, max_length=length)

    found_issue = False
    for i in range(num_peptides):
        peptide = dataset._sample_peptide()
        charge = dataset._sample_charge()

        try:
            masses, intensities = dataset._ms2pip_predict(peptide, charge)

            if not np.all(np.isfinite(intensities)):
                print(f"  [ERROR] Found non-finite values for peptide: {peptide} (charge {charge})")
                print(f"  Intensities: {intensities}")
                found_issue = True

        except Exception as e:
            print(f"  [ERROR] Exception during MS2PIP prediction for peptide: {peptide}")
            print(f"  Exception: {e}")
            found_issue = True

        if (i + 1) % 100 == 0:
            print(f"  ...tested {i+1}/{num_peptides} peptides.")

    if not found_issue:
        print(f"  [SUCCESS] No issues found for peptides of length {length}.")

if __name__ == "__main__":
    # The curriculum fails in Stage 3, which introduces length 11 peptides.
    test_peptide_length(10)
    test_peptide_length(11)
