import json
from collections import Counter
from pathlib import Path
from .domain import AnalysisResult
from .species import chromosome_sort_key, get_species

DISCLAIMER = "RESEARCH/WORKFLOW-ASSISTANCE ONLY — not a diagnostic result. A certified cytogeneticist must independently review this draft before any use, and formal analytical validation is required before clinical/SOP use."


def write_json(result: AnalysisResult, path: str) -> None:
    Path(path).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def _provisional_notation(result: AnalysisResult) -> str:
    """A single ISCN-style summary line, clearly marked provisional.

    This is a formatting convenience for a reviewer scanning many reports,
    not a claim of a determined karyotype -- sex chromosome complement is
    never inferred by this pipeline (see the "X/Y?" label used throughout),
    so it is rendered as unresolved here too.
    """
    config = get_species(result.species)
    total = len(result.objects)
    count_str = str(total) if total == config.expected_count else f"{total}(exp.{config.expected_count})"
    return f"{count_str},{result.species},sex-undetermined [DRAFT — unconfirmed, pending expert review]"


def write_report(result: AnalysisResult, path: str) -> None:
    objs = result.objects
    counts = Counter(o.label for o in objs if o.label)
    quality_flag = result.flags[0] if result.flags and result.flags[0].startswith("QUALITY GATE") else None
    other_flags = result.flags[1:] if quality_flag else result.flags

    lines = [
        "# Karyotype Image Analysis — Draft Report",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "## Summary",
        "",
        "| | |",
        "|---|---|",
        f"| Provisional notation | `{_provisional_notation(result)}` |",
        f"| Source image | `{result.source}` |",
        f"| Species | {result.species} |",
        f"| Detected modality | {result.modality} |",
        f"| Candidate objects | {len(objs)} of {get_species(result.species).expected_count} expected |",
        f"| Reviewer sign-off | {'Approved — ' + result.reviewer if result.approved and result.reviewer else 'Not yet reviewed'} |",
    ]

    if quality_flag:
        verdict = quality_flag.split("—")[1].strip().split(":")[0].strip() if "—" in quality_flag else ""
        lines += ["", f"## Image Quality Gate — {verdict}", "", quality_flag.split(': ', 1)[-1]]

    lines += ["", "## Acquisition & QC"]
    lines += [f"- {k.replace('_', ' ').title()}: {v}" for k, v in result.qc.items()]

    lines += [
        "",
        "## Candidate Chromosome Objects",
        f"- Segmented: {len(objs)}",
        f"- Full: {sum(o.status == 'full' for o in objs)}  ·  Half/fragment: {sum(o.status == 'half' for o in objs)}",
        f"- Marked migrated: {sum(o.migrated for o in objs)}  ·  Marked dislocated: {sum(o.dislocated for o in objs)}",
        "",
        "## Draft Pair Counts",
    ]
    lines += [f"- `{key}`: {value}" for key, value in sorted(counts.items(), key=lambda x: chromosome_sort_key(x[0]))] or ["- No labels assigned."]

    lines += ["", "## Review Flags"]
    lines += [f"- {x}" for x in other_flags] if other_flags else ["- No additional automatic flags; this does not establish normality."]

    lines += [
        "",
        "## Object Review Table",
        "| Object | Draft label | Confidence | Status | Migrated | Dislocated | Touching | Notes |",
        "|---:|---|---:|---|---|---|---|---|",
    ]
    for o in objs:
        lines.append(f"| {o.object_id} | {o.label or 'Unassigned'} | {o.confidence:.2f} | {o.status} | {o.migrated} | {o.dislocated} | {o.touching} | {o.notes} |")

    lines += [
        "",
        "## Sign-off",
        f"- Reviewer: {result.reviewer or 'Not supplied'}",
        f"- Approved after manual review: {'Yes' if result.approved else 'No — draft only'}",
        "",
        "---",
        DISCLAIMER,
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
