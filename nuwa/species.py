"""Shared species selections for annotation subsets and model provenance."""

import hashlib
import re

from .paths import DEFAULT_SPECIES


def normalize_species(value=DEFAULT_SPECIES):
    """Accept legacy strings or lists; produce a stable, case-insensitive set."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Use one or more species names, or all")
    if any(not isinstance(name, str) or not name.strip() for name in value):
        raise ValueError("Species names must be nonempty strings")
    names = sorted({name.strip().casefold() for name in value})
    if "all" in names and len(names) != 1:
        raise ValueError("Use all by itself, without other species names")
    return names


def species_slug(value):
    """Keep single-species paths unchanged and identify groups independent of order."""
    parts = [re.sub(r"[^a-z0-9]+", "_", name).strip("_")
             for name in normalize_species(value)]
    if not all(parts):
        raise ValueError("Species names must contain letters or numbers")
    slug = "__".join(parts)
    if len(slug) > 100:
        digest = hashlib.sha256("\0".join(normalize_species(value)).encode("utf-8")).hexdigest()[:12]
        slug = slug[:80].rstrip("_") + "__" + digest
    return slug


def category_ids(categories, species):
    """Resolve species against the original category table without renumbering."""
    names = normalize_species(species)
    if names == ["all"]:
        return sorted(category["id"] for category in categories)
    result = []
    for name in names:
        matches = [category["id"] for category in categories
                   if category["name"].strip().casefold() == name]
        if len(matches) != 1:
            raise ValueError("Species {!r} was not found uniquely. Run prepare --list first.".format(name))
        result.append(matches[0])
    return sorted(result)


def add_species_argument(parser, default=None):
    return parser.add_argument(
        "--species", nargs="+", default=default, metavar="SPECIES",
        help="One or more species names (quote names containing spaces), or all; "
             "default: checkpoint species when available, otherwise dog")
