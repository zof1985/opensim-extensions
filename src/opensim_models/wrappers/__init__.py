"""Public component-wrapper namespace.

This transitional re-export keeps the current component API available while
native-inheriting wrapper classes are generated and migrated.
"""

from .. import components
from ..components import *

__all__ = [*components.__all__]
