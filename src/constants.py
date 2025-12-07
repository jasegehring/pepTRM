"""
Physical constants for peptide mass spectrometry.
All masses are monoisotopic masses in Daltons (Da).
"""

# Standard amino acid masses (monoisotopic, from UniProt)
AMINO_ACID_MASSES: dict[str, float] = {
    'A': 71.03711,    # Alanine
    'R': 156.10111,   # Arginine
    'N': 114.04293,   # Asparagine
    'D': 115.02694,   # Aspartic acid
    'C': 103.00919,   # Cysteine
    'E': 129.04259,   # Glutamic acid
    'Q': 128.05858,   # Glutamine
    'G': 57.02146,    # Glycine
    'H': 137.05891,   # Histidine
    'I': 113.08406,   # Isoleucine  (SAME MASS AS LEUCINE)
    'L': 113.08406,   # Leucine     (SAME MASS AS ISOLEUCINE)
    'K': 128.09496,   # Lysine
    'M': 131.04049,   # Methionine
    'F': 147.06841,   # Phenylalanine
    'P': 97.05276,    # Proline
    'S': 87.03203,    # Serine
    'T': 101.04768,   # Threonine
    'W': 186.07931,   # Tryptophan
    'Y': 163.06333,   # Tyrosine
    'V': 99.06841,    # Valine
}

# Physical constants
WATER_MASS = 18.01056       # H2O
PROTON_MASS = 1.00727       # H+
AMMONIA_MASS = 17.02655     # NH3 (for neutral loss)
CO_MASS = 27.99491          # CO (for a-ion calculation)

# Vocabulary for model
STANDARD_VOCAB = list(AMINO_ACID_MASSES.keys())  # 20 amino acids
SPECIAL_TOKENS = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
VOCAB = SPECIAL_TOKENS + STANDARD_VOCAB

# Token indices
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
AA_START_IDX = 4

VOCAB_SIZE = len(VOCAB)  # 24 (4 special + 20 AA)

# Isobaric groups (amino acids with same/similar masses)
ISOBARIC_GROUPS = {
    'IL': {'I', 'L'},           # Exactly same mass (113.08406)
    'KQ': {'K', 'Q'},           # Very similar (128.095 vs 128.059, Δ=0.036)
}

# Create lookup dictionaries
AA_TO_IDX = {aa: i for i, aa in enumerate(VOCAB)}
IDX_TO_AA = {i: aa for i, aa in enumerate(VOCAB)}
AA_MASS_TENSOR_ORDER = [AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB]
