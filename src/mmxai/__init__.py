"""Explainable multimodal pneumonia diagnosis from chest radiographs, triage text and structured data."""

__version__ = "0.1.0"

# Canonical modality order. Every subset, table row and token sequence follows it.
MODALITIES = ("image", "text", "tab")
MODALITY_NAMES = {"image": "Image", "text": "Clinical text", "tab": "Structured data"}
MODALITY_LETTERS = {"image": "I", "text": "T", "tab": "S"}
