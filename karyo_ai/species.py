from dataclasses import dataclass

@dataclass(frozen=True)
class SpeciesConfig:
    name: str
    expected_count: int
    autosome_labels: tuple[str, ...]

HUMAN = SpeciesConfig("human", 46, tuple(str(i) for i in range(1, 23)))
MOUSE = SpeciesConfig("mouse", 40, tuple(str(i) for i in range(1, 20)))

def get_species(name: str) -> SpeciesConfig:
    key = name.lower().strip()
    if key == "human": return HUMAN
    if key in {"mouse", "murine"}: return MOUSE
    raise ValueError("species must be 'human' or 'mouse'")

def chromosome_sort_key(label) -> tuple:
    """Natural karyotype order: 1, 2, ... 22, X, Y, then anything unassigned.

    Plain string sorting puts "10" before "2", which scrambles the draft
    karyogram layout and the report's count table. This key restores the
    conventional Denver-style ordering regardless of species.
    """
    if label is None:
        return (2, "")
    text = str(label).strip()
    if text.isdigit():
        return (0, int(text))
    if text.upper() in {"X", "Y"}:
        return (1, 0 if text.upper() == "X" else 1)
    return (2, text)
