"""Species configuration for the chromosome-identity classifier (spec section 4).

Deliberately separate from `karyo_ai.species` (the classical baseline's
config): the two tracks may need different fields as this evolves, and the
baseline must keep working unmodified while this is built out.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DNNSpeciesConfig:
    name: str
    autosome_count: int
    sex_chromosomes: Tuple[str, ...] = ("X", "Y")

    @property
    def chromosome_count(self) -> int:
        """Total diploid count, e.g. 46 for human (2*22 autosomes + X + Y)."""
        return 2 * self.autosome_count + len(self.sex_chromosomes)

    @property
    def autosome_labels(self) -> Tuple[str, ...]:
        return tuple(str(i) for i in range(1, self.autosome_count + 1))

    @property
    def all_labels(self) -> Tuple[str, ...]:
        return self.autosome_labels + self.sex_chromosomes


HUMAN = DNNSpeciesConfig(name="human", autosome_count=22)
MOUSE = DNNSpeciesConfig(name="mouse", autosome_count=19)

_REGISTRY = {"human": HUMAN, "mouse": MOUSE}


def get_species(name: str) -> DNNSpeciesConfig:
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown species '{name}'; supported: {sorted(_REGISTRY)}")
    return _REGISTRY[key]
