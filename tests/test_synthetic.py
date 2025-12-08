"""Tests for synthetic data generation."""

import pytest
from src.data.synthetic import (
    generate_theoretical_spectrum,
    compute_peptide_mass,
    generate_random_peptide,
)
from src.constants import AMINO_ACID_MASSES, WATER_MASS


class TestTheoreticalSpectrum:
    """Test theoretical spectrum generation."""

    def test_b_ion_series(self):
        """Verify b-ion masses."""
        from src.constants import PROTON_MASS
        spectrum = generate_theoretical_spectrum("AG", ion_types=['b'])

        # b1 = A = 71.04
        b_ions = [(m, a) for m, a in zip(
            [p[0] for p in spectrum.peaks],
            [spectrum.ion_annotations.get(p[0], '') for p in spectrum.peaks]
        ) if 'b' in a]

        assert len(b_ions) >= 1
        # b1 should be mass of A + H+
        expected_b1_mz = AMINO_ACID_MASSES['A'] + PROTON_MASS
        assert abs(b_ions[0][0] - expected_b1_mz) < 0.01

    def test_y_ion_series(self):
        """Verify y-ion masses (include water)."""
        from src.constants import PROTON_MASS
        spectrum = generate_theoretical_spectrum("AG", ion_types=['y'])

        y_ions = [(m, a) for m, a in zip(
            [p[0] for p in spectrum.peaks],
            [spectrum.ion_annotations.get(p[0], '') for p in spectrum.peaks]
        ) if 'y' in a]

        assert len(y_ions) >= 1
        # y1 = G + H2O + H+
        expected_y1_mz = AMINO_ACID_MASSES['G'] + WATER_MASS + PROTON_MASS
        assert abs(y_ions[0][0] - expected_y1_mz) < 0.01

    def test_precursor_mass(self):
        """Verify precursor mass calculation."""
        spectrum = generate_theoretical_spectrum("ACDEF")
        calculated_precursor = compute_peptide_mass("ACDEF")

        assert abs(spectrum.precursor_mass - calculated_precursor) < 0.01

    def test_spectrum_has_peaks(self):
        """Ensure spectrum generation produces peaks."""
        spectrum = generate_theoretical_spectrum("PEPTIDE")

        assert len(spectrum.peaks) > 0
        # For a 7-residue peptide with b and y ions, expect ~12 peaks
        assert len(spectrum.peaks) >= 10

    def test_peak_dropout(self):
        """Test that peak dropout reduces number of peaks."""
        full_spectrum = generate_theoretical_spectrum("PEPTIDE", peak_dropout=0.0)
        dropped_spectrum = generate_theoretical_spectrum("PEPTIDE", peak_dropout=0.5)

        # With 50% dropout, expect roughly half the peaks (with some variance)
        assert len(dropped_spectrum.peaks) < len(full_spectrum.peaks)

    def test_noise_peaks(self):
        """Test that noise peaks are added."""
        clean_spectrum = generate_theoretical_spectrum("PEP", noise_peaks=0)
        noisy_spectrum = generate_theoretical_spectrum("PEP", noise_peaks=10)

        assert len(noisy_spectrum.peaks) > len(clean_spectrum.peaks)

    def test_charge_state(self):
        """Test that charge state is stored correctly."""
        spectrum = generate_theoretical_spectrum("PEPTIDE", charge=3)
        assert spectrum.charge == 3


class TestRandomPeptide:
    """Test random peptide generation."""

    def test_length_constraints(self):
        """Test length constraints."""
        for _ in range(100):
            peptide = generate_random_peptide(min_length=7, max_length=10)
            assert 7 <= len(peptide) <= 10

    def test_valid_amino_acids(self):
        """Test that only valid amino acids are generated."""
        peptide = generate_random_peptide()
        for aa in peptide:
            assert aa in AMINO_ACID_MASSES, f"Invalid amino acid: {aa}"

    def test_exclude_amino_acids(self):
        """Test excluding certain amino acids."""
        exclude = {'C', 'M'}
        for _ in range(50):
            peptide = generate_random_peptide(min_length=10, max_length=15, exclude_aa=exclude)
            for aa in peptide:
                assert aa not in exclude


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
