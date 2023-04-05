from typing import Any, TypeAlias

from numpy import floating
from numpy.typing import NDArray

scalarT: TypeAlias = float | floating[Any]
NDAf: TypeAlias = NDArray[floating[Any]]
