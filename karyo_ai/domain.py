from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class ChromosomeObject:
    object_id: int
    bbox: tuple[int, int, int, int]
    area: int
    centroid: tuple[float, float]
    label: Optional[str] = None
    homolog: Optional[int] = None  # 1 or 2 within a pair; None if unpaired/leftover
    confidence: float = 0.0
    status: str = "full"
    migrated: bool = False
    dislocated: bool = False
    touching: bool = False
    features: dict = field(default_factory=dict)
    notes: str = ""
    mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)  # local boolean mask, bbox-sized; not serialized

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items() if k != "mask"}
        return d

@dataclass
class AnalysisResult:
    source: str
    species: str
    modality: str
    objects: list[ChromosomeObject]
    qc: dict
    flags: list[str]
    reviewer: str = ""
    approved: bool = False

    def to_dict(self):
        return {"source": self.source, "species": self.species, "modality": self.modality,
                "objects": [x.to_dict() for x in self.objects], "qc": self.qc,
                "flags": self.flags, "reviewer": self.reviewer, "approved": self.approved}
