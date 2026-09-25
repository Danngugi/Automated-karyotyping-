from .domain import ChromosomeObject
from .species import get_species

# What a size-rank heuristic can and cannot do:
#
# Sorting chromosomes by length and pairing neighbouring ranks reproduces the
# *shape* of a karyogram (23 pairs, largest first) but it is NOT chromosome
# identification. Real identification uses banding pattern and centromere
# position; several human chromosomes are near-identical in length (notably
# the 9-12 group and 13-15), so rank order alone cannot distinguish them.
# The sex pair especially cannot be called from size: X sits in the size
# range of the C group, and Y is small and easily confused with G-group
# autosomes. Every label produced here is therefore a *slot assignment* for
# review, carries a low confidence, and must be corrected by a cytogeneticist
# via --annotations before it means anything.
SLOT_CONFIDENCE = 0.15

# Real metaphase chromosomes are compact: even a long submetacentric one
# rarely exceeds an aspect ratio of about 7. Much thinner objects are
# almost always non-biological -- image borders, scale bars, or text
# burnt into a screenshot. These matter disproportionately because they
# tend to be *long*, so a size-ranked assignment hands them the
# chromosome-1 slots and shifts every subsequent pair by one.
MAX_PLAUSIBLE_ASPECT = 8.0


def is_probable_artifact(obj: ChromosomeObject) -> bool:
    aspect = obj.features.get("aspect_ratio", 0)
    return bool(aspect and aspect > MAX_PLAUSIBLE_ASPECT)


def assign_pairs(objects: list[ChromosomeObject], species: str) -> None:
    """Assign each object a karyogram slot: chromosome label + homolog index.

    Objects are ranked by length and filled into the species' expected slot
    sequence two at a time -- 1:1, 1:2, 2:1, 2:2, ... 22:1, 22:2, then the
    sex pair. Any objects beyond the expected count (over-segmentation, or a
    genuinely abnormal spread) are left unlabelled rather than forced into a
    slot, so that an extra object shows up as an extra object instead of
    silently corrupting a real pair.

    Implausibly elongated objects are excluded from slot assignment and
    flagged rather than deleted: the reviewer should see that something was
    detected and rejected, not have it vanish silently.
    """
    config = get_species(species)

    artifacts = [o for o in objects if is_probable_artifact(o)]
    for obj in artifacts:
        if obj.label:
            continue
        obj.label = None
        obj.homolog = None
        obj.confidence = 0.0
        aspect = obj.features.get("aspect_ratio", 0)
        obj.notes = (obj.notes + "; " if obj.notes else "") + (
            f"excluded from pairing: aspect ratio {aspect} exceeds {MAX_PLAUSIBLE_ASPECT}, "
            "likely an image border, scale bar, or burnt-in text rather than a chromosome"
        )

    candidates = [o for o in objects if not is_probable_artifact(o)]
    ordered = sorted(candidates, key=lambda o: o.features.get("length_px", 0), reverse=True)

    # Expected slots: each autosome twice, largest-numbered last, then the sex pair.
    slots: list[tuple[str, int]] = []
    for label in config.autosome_labels:
        slots.append((label, 1))
        slots.append((label, 2))
    # Sex chromosomes cannot be called from morphology alone (see note above),
    # so both slots get a single undetermined label for the reviewer to resolve.
    slots.append(("X/Y?", 1))
    slots.append(("X/Y?", 2))

    for obj, (label, homolog) in zip(ordered, slots):
        if obj.label:  # expert annotation already present; never overwrite it
            continue
        obj.label = label
        obj.homolog = homolog
        obj.confidence = SLOT_CONFIDENCE

    for obj in ordered[len(slots):]:
        if obj.label:
            continue
        obj.label = None
        obj.homolog = None
        obj.confidence = 0.0
        obj.notes = (obj.notes + "; " if obj.notes else "") + (
            f"beyond the {len(slots)} expected slots for {species}; unassigned pending review"
        )


# Backwards-compatible alias: earlier code and any external callers used this name.
classify_heuristic = assign_pairs


def apply_annotations(objects: list[ChromosomeObject], annotations: dict[int, dict]) -> None:
    for obj in objects:
        row = annotations.get(obj.object_id)
        if not row:
            continue
        obj.label = row.get("chromosome_label") or obj.label
        obj.status = row.get("status") or obj.status
        obj.migrated = str(row.get("migrated", "")).lower() in {"1", "true", "yes", "y"}
        obj.dislocated = str(row.get("dislocated", "")).lower() in {"1", "true", "yes", "y"}
        obj.notes = row.get("notes", "")
        obj.confidence = 1.0
