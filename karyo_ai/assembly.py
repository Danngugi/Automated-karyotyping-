from collections import Counter
from PIL import Image, ImageDraw
from .domain import ChromosomeObject
from .species import get_species


def count_flags(objects: list[ChromosomeObject], species: str) -> list[str]:
    config = get_species(species)
    counts = Counter(o.label for o in objects if o.label)
    flags = []
    for label in config.autosome_labels:
        n = counts.get(label, 0)
        if n != 2:
            flags.append(f"Draft count for chromosome {label}: {n}; expected 2. Human review required.")
    sex_n = counts.get("X/Y?", 0)
    if sex_n != 2:
        flags.append(f"Draft count for the sex pair: {sex_n}; expected 2. Human review required.")
    unassigned = sum(1 for o in objects if not o.label)
    if unassigned:
        flags.append(f"{unassigned} object(s) beyond the expected {config.expected_count} slots were left unassigned.")
    total = len([o for o in objects])
    if species == "human" and total == config.expected_count - 1:
        flags.append(
            f"Total candidate objects: {total}; expected {config.expected_count}. Before treating this as a "
            "possible monosomy (e.g. 45,X / Turner syndrome), rule out a segmentation artifact -- two "
            "touching chromosomes merged into one detected object is far more common than a true missing "
            "chromosome. Requires expert review either way."
        )
    elif species == "human" and total == config.expected_count + 1:
        flags.append(
            f"Total candidate objects: {total}; expected {config.expected_count}. Before treating this as a "
            "possible trisomy (e.g. trisomy 21 / Down syndrome), rule out a false positive -- debris, stain "
            "precipitate, or an over-split object counted twice is far more common than a true extra "
            "chromosome. Requires expert review either way."
        )
    elif total != config.expected_count:
        flags.append(
            f"Total candidate objects: {total}; expected {config.expected_count} for {species}. "
            "This is a segmentation/counting draft, not an aneuploidy finding -- a count that differs "
            "from the expected number is far more likely to be a segmentation error than a real "
            "chromosomal abnormality, and must not be reported as one without expert review."
        )
    if any(o.touching for o in objects):
        flags.append("Touching/overlapping candidate objects detected; segmentation needs review.")
    return flags


def _slot_text(obj: ChromosomeObject) -> str:
    if not obj.label:
        return f"{obj.object_id}: unassigned"
    if obj.homolog:
        return f"{obj.label}:{obj.homolog}"
    return f"{obj.label}"


def overlap_quality_gate(objects: list[ChromosomeObject], species: str, max_unresolved_fraction: float = 0.15):
    """Whole-spread quality gate for chromosome overlap, mirroring a real
    cytogenetics lab practice: reject/flag poorly-spread metaphases upstream
    rather than presenting a low-quality analysis as if it were reliable.

    This is a deliberately simple, conservative heuristic -- not the trained
    YOLOv8 overlap detector described in Rosas-Alatriste et al. 2026 (Sci
    Rep 16:22469), which needs a large annotated dataset and a GPU this
    project doesn't have. What's implemented here is the underlying
    *decision*, using signal this pipeline already computes: the fraction
    of total detected foreground area sitting in clusters the watershed
    step could not cleanly separate (`touching=True`).

    Returns (verdict, detail) where verdict is one of "good", "review",
    "poor" and detail is a human-readable explanation.
    """
    total_area = sum(o.area for o in objects) or 1
    unresolved_area = sum(o.area for o in objects if o.touching)
    fraction = unresolved_area / total_area
    config = get_species(species)
    count_ratio = len(objects) / config.expected_count

    if fraction > max_unresolved_fraction * 2 or count_ratio < 0.6:
        return "poor", (
            f"{fraction:.0%} of detected chromosome area sits in unresolved overlapping clusters "
            f"(only {len(objects)}/{config.expected_count} expected objects found). Per standard "
            "cytogenetic practice (ACMG technical standards), a spread this poorly resolved should "
            "be re-imaged or manually reviewed rather than analyzed as-is -- treat everything below "
            "as illustrative only."
        )
    if fraction > max_unresolved_fraction or count_ratio < 0.85:
        return "review", (
            f"{fraction:.0%} of detected chromosome area sits in unresolved overlapping clusters. "
            "Usable for draft review, but expect more correction than usual in the flagged clusters."
        )
    return "good", f"Only {fraction:.0%} of detected area is unresolved overlap; typical for this pipeline."


def render_karyogram(image: Image.Image, objects: list[ChromosomeObject], species: str, path: str) -> None:
    """Render a draft karyogram as a full, fixed-size grid -- every expected
    slot (1:1, 1:2, 2:1, 2:2, ... through the sex pair) is drawn, whether or
    not an object was actually assigned to it.

    An earlier version only drew tiles for objects that were actually
    detected and labelled, so on a spread where most chromosomes were never
    found, the image just silently stopped partway through instead of
    looking like an incomplete karyogram -- there was no way to tell "this
    image render is fine, the underlying data is what's incomplete" from
    looking at it. Missing slots now render as a visibly empty grey tile,
    so gaps are exactly where a reviewer would look for them: in the grid,
    not in a truncated image.
    """
    config = get_species(species)
    slots = [(label, h) for label in config.autosome_labels for h in (1, 2)] + [("X/Y?", 1), ("X/Y?", 2)]
    by_slot = {(o.label, o.homolog): o for o in objects if o.label and o.homolog}

    def make_tile(obj: ChromosomeObject) -> Image.Image:
        x0, y0, x1, y1 = obj.bbox
        crop = image.crop((x0, y0, x1, y1))
        crop.thumbnail((100, 150))
        tile = Image.new("RGB", (120, 190), "white")
        tile.paste(crop, ((120 - crop.width) // 2, 10))
        d = ImageDraw.Draw(tile)
        d.text((5, 165), _slot_text(obj), fill="black")
        return tile

    def make_empty_tile(label: str, homolog: int) -> Image.Image:
        tile = Image.new("RGB", (120, 190), (222, 222, 222))
        d = ImageDraw.Draw(tile)
        d.rectangle((10, 10, 109, 149), outline=(160, 160, 160), width=1)
        d.text((25, 75), "not detected", fill=(130, 130, 130))
        d.text((5, 165), f"{label}:{homolog}", fill=(110, 110, 110))
        return tile

    tiles = [make_tile(by_slot[slot]) if slot in by_slot else make_empty_tile(*slot) for slot in slots]

    # Extra objects beyond the expected slots (over-segmentation, or objects
    # left unassigned/unlabelled) go in a trailing section so nothing found
    # is silently dropped from the image either.
    extras = [o for o in objects if not (o.label and o.homolog)]
    for obj in sorted(extras, key=lambda o: o.object_id):
        tile = image.crop(obj.bbox)
        tile.thumbnail((100, 150))
        wrapped = Image.new("RGB", (120, 190), (255, 245, 225))
        wrapped.paste(tile, ((120 - tile.width) // 2, 10))
        ImageDraw.Draw(wrapped).text((5, 165), f"#{obj.object_id}: unassigned", fill=(150, 90, 0))
        tiles.append(wrapped)

    cols = 8
    rows = max(1, (len(tiles) + cols - 1) // cols)
    out = Image.new("RGB", (cols * 120, rows * 190), (245, 245, 245))
    for i, tile in enumerate(tiles):
        out.paste(tile, ((i % cols) * 120, (i // cols) * 190))
    out.save(path)
