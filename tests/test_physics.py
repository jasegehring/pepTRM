"""
CRITICAL: Physics tests - run these FIRST before any model development.

These tests verify that mass calculations are correct. If these fail,
all downstream results will be meaningless.
"""

import pytest
from src.constants import AMINO_ACID_MASSES, WATER_MASS, PROTON_MASS


def compute_peptide_mass(peptide: str) -> float:
    """
    Compute monoisotopic mass of peptide.

    Peptide mass = sum of residue masses + H2O (for N and C termini)
    """
    residue_mass = sum(AMINO_ACID_MASSES[aa] for aa in peptide)
    return residue_mass + WATER_MASS


class TestMassCalculations:
    """Test basic mass calculations."""

    def test_single_amino_acid_masses(self):
        """Verify individual amino acid masses against known values."""
        # Reference values from UniProt
        known_masses = {
            'G': 57.02146,
            'A': 71.03711,
            'S': 87.03203,
            'P': 97.05276,
            'V': 99.06841,
            'T': 101.04768,
            'C': 103.00919,
            'I': 113.08406,
            'L': 113.08406,
            'N': 114.04293,
            'D': 115.02694,
            'Q': 128.05858,
            'K': 128.09496,
            'E': 129.04259,
            'M': 131.04049,
            'H': 137.05891,
            'F': 147.06841,
            'R': 156.10111,
            'Y': 163.06333,
            'W': 186.07931,
        }

        for aa, expected in known_masses.items():
            assert abs(AMINO_ACID_MASSES[aa] - expected) < 0.00001, \
                f"Mass of {aa} is wrong: {AMINO_ACID_MASSES[aa]} vs {expected}"

    def test_peptide_mass(self):
        """Test peptide mass calculation against known values."""
        # PEPTIDE: Calculate manually
        # P(97.05) + E(129.04) + P(97.05) + T(101.05) + I(113.08) + D(115.03) + E(129.04)
        # = 781.34 + H2O(18.01) = 799.35 Da
        expected = 799.35
        calculated = compute_peptide_mass("PEPTIDE")
        assert abs(calculated - expected) < 0.1, \
            f"PEPTIDE mass wrong: {calculated} vs {expected}"

    def test_glycine_polymer(self):
        """Test with simple glycine polymer."""
        # 5 glycines + water
        expected = 5 * 57.02146 + WATER_MASS
        calculated = compute_peptide_mass("GGGGG")
        assert abs(calculated - expected) < 0.01, \
            f"G5 mass wrong: {calculated} vs {expected}"

    def test_water_mass(self):
        """Verify water mass constant."""
        expected = 18.01056
        assert abs(WATER_MASS - expected) < 0.00001, \
            f"Water mass wrong: {WATER_MASS} vs {expected}"

    def test_proton_mass(self):
        """Verify proton mass constant."""
        expected = 1.00727
        assert abs(PROTON_MASS - expected) < 0.00001, \
            f"Proton mass wrong: {PROTON_MASS} vs {expected}"

    def test_isobaric_masses(self):
        """Test that isoleucine and leucine have identical masses."""
        assert AMINO_ACID_MASSES['I'] == AMINO_ACID_MASSES['L'], \
            "I and L should have identical masses"

        # Verify the exact value
        expected = 113.08406
        assert abs(AMINO_ACID_MASSES['I'] - expected) < 0.00001
        assert abs(AMINO_ACID_MASSES['L'] - expected) < 0.00001


class TestBIonMasses:
    """Test b-ion mass calculations."""

    def test_simple_b_ions(self):
        """Test b-ion masses for simple peptide."""
        peptide = "AG"  # Alanine-Glycine

        # b1 = A = 71.03711
        b1 = AMINO_ACID_MASSES['A']
        expected_b1 = 71.03711
        assert abs(b1 - expected_b1) < 0.00001, \
            f"b1 mass wrong: {b1} vs {expected_b1}"


class TestYIonMasses:
    """Test y-ion mass calculations."""

    def test_simple_y_ions(self):
        """Test y-ion masses for simple peptide."""
        peptide = "AG"  # Alanine-Glycine

        # y1 = G + H2O = 57.02146 + 18.01056
        y1 = AMINO_ACID_MASSES['G'] + WATER_MASS
        expected_y1 = 75.03202
        assert abs(y1 - expected_y1) < 0.00001, \
            f"y1 mass wrong: {y1} vs {expected_y1}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
