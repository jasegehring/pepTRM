"""
Quick test of MS2PIP API to understand data structures and outputs.
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ms2pip import predict_single, predict_batch

def test_basic_prediction():
    """Test basic MS2PIP prediction."""
    print("=" * 80)
    print("TESTING MS2PIP API")
    print("=" * 80)

    # Test peptide
    test_peptide = "PEPTIDE"
    test_charge = 2

    print(f"\n1. Predicting spectrum for: {test_peptide} (charge +{test_charge})")

    # Make prediction using predict_single
    try:
        result = predict_single(
            peptide=test_peptide,
            charge=test_charge,
            model="HCD2021"
        )

        print(f"   ✓ Prediction successful")
        print(f"\n2. Analyzing result structure...")
        print(f"   Type: {type(result)}")

        # Check what result contains
        if hasattr(result, '__dict__'):
            print(f"   Attributes: {list(result.__dict__.keys())}")
        elif isinstance(result, dict):
            print(f"   Keys: {list(result.keys())}")

        # Try to access the data
        df = None
        if hasattr(result, 'to_dataframe'):
            df = result.to_dataframe()
        elif isinstance(result, dict) and 'predictions' in result:
            # May need to extract dataframe differently
            print(f"   Result is dict with 'predictions' key")
            df = result['predictions']

        if df is not None:
            print(f"\n3. DataFrame shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"\n4. First 20 rows:")
            print(df.head(20))

            # Check fragment types
            if 'ion' in df.columns:
                fragment_types = df['ion'].unique()
                print(f"\n5. Fragment types present: {fragment_types}")

                # Count each type
                for frag_type in fragment_types:
                    count = len(df[df['ion'] == frag_type])
                    print(f"   {frag_type}: {count} fragments")

            # Check charge states
            if 'charge' in df.columns:
                charges = df['charge'].unique()
                print(f"\n6. Charge states: {charges}")

            # Check intensity range
            if 'prediction' in df.columns:
                print(f"\n7. Intensity statistics:")
                print(f"   Min: {df['prediction'].min():.4f}")
                print(f"   Max: {df['prediction'].max():.4f}")
                print(f"   Mean: {df['prediction'].mean():.4f}")
                print(f"   Median: {df['prediction'].median():.4f}")
        else:
            print(f"\n3. Could not extract DataFrame from result")
            print(f"   Result type: {type(result)}")
            if isinstance(result, dict):
                print(f"   Result keys: {result.keys()}")
            print(f"   Result content: {result}")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)

def test_multiple_peptides():
    """Test with multiple peptides and charges."""
    print("\n" + "=" * 80)
    print("TESTING MULTIPLE PEPTIDES")
    print("=" * 80)

    peptides = ["PEPTIDE", "SEQUENCE", "ANALYSIS"]
    charges = [2, 3, 2]

    print(f"\nPredicting for {len(peptides)} peptides...")

    try:
        # predict_batch takes a list of (peptide, charge) tuples or a DataFrame
        results = predict_batch(
            peptides=peptides,
            charges=charges,
            model="HCD2021"
        )

        print("✓ Batch prediction successful")
        print(f"Result type: {type(results)}")

        # Try to extract dataframe
        df = None
        if hasattr(results, 'to_dataframe'):
            df = results.to_dataframe()
        elif isinstance(results, dict) and 'predictions' in results:
            df = results['predictions']

        if df is not None:
            print(f"Total fragments predicted: {len(df)}")

            # Group by peptide
            if 'sequence' in df.columns or 'peptide' in df.columns:
                seq_col = 'sequence' if 'sequence' in df.columns else 'peptide'
                for pep in peptides:
                    pep_df = df[df[seq_col] == pep]
                    print(f"  {pep}: {len(pep_df)} fragments")
        else:
            print(f"Could not extract DataFrame")
            print(f"Results: {results}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_basic_prediction()
    test_multiple_peptides()
    print("\nTest complete!")
