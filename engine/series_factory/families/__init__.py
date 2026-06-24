"""series_factory.families — drop a new file here to add a family.

Each module side-effect-registers its subtype builders via
@register_subtype(family, subtype) decorators. Importing this package
triggers them all.

Adding a new family is a 1-file PR — never touches the core registry.
"""

# Side-effect imports — order doesn't matter
from engine.series_factory.families import carry            # noqa: F401
# Future:
# from engine.series_factory.families import vrp            # noqa: F401
# from engine.series_factory.families import tsmom          # noqa: F401
# from engine.series_factory.families import equity_factor  # noqa: F401
