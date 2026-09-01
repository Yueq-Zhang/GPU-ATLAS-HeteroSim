"""Versioned control-plane package for GPU-ATLAS-HeteroSim."""

from .bandwidth import BandwidthContract, BandwidthContractError
from .operator_capability import (
    OperatorCapabilityCatalog,
    OperatorCapabilityError,
)
from .performance_calibration import (
    PerformanceCalibration,
    PerformanceCalibrationError,
    evaluate_performance_gate,
)

__version__ = "0.27.0"

__all__ = [
    "BandwidthContract",
    "BandwidthContractError",
    "OperatorCapabilityCatalog",
    "OperatorCapabilityError",
    "PerformanceCalibration",
    "PerformanceCalibrationError",
    "evaluate_performance_gate",
]
