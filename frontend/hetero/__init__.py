"""Versioned control-plane package for GPU-ATLAS-HeteroSim."""

from .bandwidth import BandwidthContract, BandwidthContractError

__version__ = "0.6.2"

__all__ = ["BandwidthContract", "BandwidthContractError"]
