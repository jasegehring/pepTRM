"""
Flexible ion type system for fragment ion computation.

This module provides a configurable way to compute theoretical fragment ions
based on the MS2PIP model or custom ion specifications.
"""

from dataclasses import dataclass
from typing import List, Optional
import torch
from torch import Tensor

from ..constants import PROTON_MASS, WATER_MASS, CO_MASS, AMMONIA_MASS


@dataclass
class IonType:
    """
    Specification for a fragment ion type.

    Attributes:
        name: Ion name (e.g., 'b', 'y', 'b++', 'y-H2O')
        series: Ion series ('b' or 'y')
        charge: Charge state (1, 2, 3, etc.)
        neutral_loss: Optional neutral loss mass (e.g., WATER_MASS for -H2O)
        modification: Optional modification mass (e.g., -CO_MASS for a-ions)
    """
    name: str
    series: str  # 'b' or 'y'
    charge: int = 1
    neutral_loss: float = 0.0
    modification: float = 0.0  # e.g., -CO_MASS for a-ions

    def __post_init__(self):
        if self.series not in ['b', 'y']:
            raise ValueError(f"Ion series must be 'b' or 'y', got '{self.series}'")
        if self.charge < 1:
            raise ValueError(f"Charge must be >= 1, got {self.charge}")


# Common ion types
ION_TYPES = {
    # Singly charged
    'b': IonType(name='b', series='b', charge=1),
    'y': IonType(name='y', series='y', charge=1),
    'a': IonType(name='a', series='b', charge=1, modification=-CO_MASS),

    # Doubly charged
    'b++': IonType(name='b++', series='b', charge=2),
    'y++': IonType(name='y++', series='y', charge=2),
    'a++': IonType(name='a++', series='b', charge=2, modification=-CO_MASS),

    # Triply charged
    'b+++': IonType(name='b+++', series='b', charge=3),
    'y+++': IonType(name='y+++', series='y', charge=3),

    # Neutral losses (singly charged)
    'b-H2O': IonType(name='b-H2O', series='b', charge=1, neutral_loss=WATER_MASS),
    'y-H2O': IonType(name='y-H2O', series='y', charge=1, neutral_loss=WATER_MASS),
    'b-NH3': IonType(name='b-NH3', series='b', charge=1, neutral_loss=AMMONIA_MASS),
    'y-NH3': IonType(name='y-NH3', series='y', charge=1, neutral_loss=AMMONIA_MASS),

    # Neutral losses (doubly charged)
    'b++-H2O': IonType(name='b++-H2O', series='b', charge=2, neutral_loss=WATER_MASS),
    'y++-H2O': IonType(name='y++-H2O', series='y', charge=2, neutral_loss=WATER_MASS),
}


# MS2PIP model to ion type mapping
MS2PIP_ION_TYPES = {
    'HCD2021': ['b', 'y'],  # Only singly charged
    'HCDch2': ['b', 'y', 'b++', 'y++'],  # Include doubly charged
    'CID': ['b', 'y'],  # CID fragmentation
    'CIDch2': ['b', 'y', 'b++', 'y++'],  # CID with doubly charged
}


def get_ion_types_for_model(ms2pip_model: str) -> List[str]:
    """
    Get the ion types that should be used for a given MS2PIP model.

    Args:
        ms2pip_model: MS2PIP model name (e.g., 'HCDch2')

    Returns:
        List of ion type names (e.g., ['b', 'y', 'b++', 'y++'])
    """
    if ms2pip_model not in MS2PIP_ION_TYPES:
        raise ValueError(
            f"Unknown MS2PIP model: {ms2pip_model}. "
            f"Supported models: {list(MS2PIP_ION_TYPES.keys())}"
        )
    return MS2PIP_ION_TYPES[ms2pip_model]


def compute_theoretical_peaks(
    sequence_probs: Tensor,
    aa_masses: Tensor,
    ion_type_names: List[str],
    sequence_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute theoretical fragment ion m/z values from sequence probabilities.

    This function supports arbitrary ion types (b, y, a, doubly charged, neutral losses, etc.)
    by using the flexible IonType system.

    Args:
        sequence_probs: (batch, seq_len, vocab_size) - probability distribution over sequences
        aa_masses: (vocab_size,) - mass of each amino acid token
        ion_type_names: List of ion type names to compute (e.g., ['b', 'y', 'b++', 'y++'])
        sequence_mask: (batch, seq_len) - optional mask indicating valid positions (True/1 = valid)

    Returns:
        theoretical_peaks: (batch, num_fragments) - m/z values of theoretical fragments

    Example:
        >>> # For HCDch2 model
        >>> ion_types = get_ion_types_for_model('HCDch2')
        >>> peaks = compute_theoretical_peaks(probs, masses, ion_types, mask)
        >>> # Returns b, y, b++, y++ ions
    """
    batch_size, seq_len, vocab_size = sequence_probs.shape

    # Compute expected mass at each position (soft embedding approach)
    # This allows gradients to flow through sequence probabilities
    expected_masses = torch.einsum('bsv,v->bs', sequence_probs, aa_masses)

    # Zero out PAD positions using mask (if provided)
    # This prevents padding tokens from contributing to fragment mass calculations
    if sequence_mask is not None:
        expected_masses = expected_masses * sequence_mask.float()

    # Remove <SOS> and <EOS> tokens to get residue masses
    # Assuming first and last positions are SOS/EOS
    residue_masses = expected_masses[:, 1:-1]  # (batch, seq_len-2)

    all_fragments = []

    for ion_type_name in ion_type_names:
        if ion_type_name not in ION_TYPES:
            raise ValueError(
                f"Unknown ion type: {ion_type_name}. "
                f"Available: {list(ION_TYPES.keys())}"
            )

        ion_type = ION_TYPES[ion_type_name]
        fragments = compute_ion_type(residue_masses, ion_type)
        all_fragments.append(fragments)

    # Concatenate all ion types
    theoretical_peaks = torch.cat(all_fragments, dim=1)

    return theoretical_peaks


def compute_ion_type(residue_masses: Tensor, ion_type: IonType) -> Tensor:
    """
    Compute m/z values for a specific ion type.

    Args:
        residue_masses: (batch, num_residues) - mass of each residue in sequence
        ion_type: IonType specification

    Returns:
        fragments: (batch, num_fragments) - m/z values for this ion type
    """
    batch_size, num_residues = residue_masses.shape

    if ion_type.series == 'b':
        # b-ions: cumulative sum from N-terminus
        # b1 = M(AA1), b2 = M(AA1) + M(AA2), etc.
        cumulative = torch.cumsum(residue_masses, dim=1)
        # Exclude last position (full sequence)
        neutral_masses = cumulative[:, :-1]

    elif ion_type.series == 'y':
        # y-ions: cumulative sum from C-terminus + water
        # y1 = M(AAn) + H2O, y2 = M(AAn-1) + M(AAn) + H2O, etc.
        cumulative = torch.cumsum(torch.flip(residue_masses, [1]), dim=1)
        # Flip back and exclude last position
        cumulative = torch.flip(cumulative, [1])[:, 1:]
        # Add water for y-ions (C-terminal OH + H)
        neutral_masses = cumulative + WATER_MASS

    else:
        raise ValueError(f"Invalid ion series: {ion_type.series}")

    # Apply modification (e.g., -CO for a-ions)
    if ion_type.modification != 0.0:
        neutral_masses = neutral_masses + ion_type.modification

    # Apply neutral loss (e.g., -H2O)
    if ion_type.neutral_loss != 0.0:
        neutral_masses = neutral_masses - ion_type.neutral_loss

    # Convert to m/z by adding protons and dividing by charge
    # m/z = (neutral_mass + charge * PROTON_MASS) / charge
    mz = (neutral_masses + ion_type.charge * PROTON_MASS) / ion_type.charge

    return mz


def validate_ion_types(ion_type_names: List[str]) -> None:
    """
    Validate that all ion type names are recognized.

    Args:
        ion_type_names: List of ion type names

    Raises:
        ValueError if any ion type is not recognized
    """
    unknown = [name for name in ion_type_names if name not in ION_TYPES]
    if unknown:
        raise ValueError(
            f"Unknown ion types: {unknown}. "
            f"Available: {list(ION_TYPES.keys())}"
        )


def add_custom_ion_type(ion_type: IonType) -> None:
    """
    Register a custom ion type.

    Args:
        ion_type: IonType specification

    Example:
        >>> # Add a custom neutral loss
        >>> custom = IonType(name='b-HPO3', series='b', charge=1, neutral_loss=79.9663)
        >>> add_custom_ion_type(custom)
    """
    if ion_type.name in ION_TYPES:
        raise ValueError(f"Ion type '{ion_type.name}' already exists")
    ION_TYPES[ion_type.name] = ion_type
