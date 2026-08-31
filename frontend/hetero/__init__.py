"""Versioned control-plane package for GPU-ATLAS-HeteroSim."""

from .bandwidth import BandwidthContract, BandwidthContractError
from .operator_capability import (
    OperatorCapabilityCatalog,
    OperatorCapabilityError,
)

__version__ = "0.24.0"

__all__ = [
    "BandwidthContract",
    "BandwidthContractError",
    "OperatorCapabilityCatalog",
    "OperatorCapabilityError",
]
