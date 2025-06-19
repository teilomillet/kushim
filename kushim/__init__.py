"""
Kushim: The main package for the LLM Evaluation Dataset Framework.

This package provides the core modules for building high-quality,
verifiable evaluation datasets for LLMs, following a systematic
extraction and validation pipeline.
"""
# This version number is centrally managed and used in package metadata.
__version__ = "0.0.3"

from . import pipeline

__all__ = ["pipeline", "__version__"]