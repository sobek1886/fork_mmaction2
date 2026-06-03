#!/usr/bin/env python3
"""Diagnose mmaction2 installation and MViT registry status."""

import sys
print("Python:", sys.executable)
print("sys.path:", sys.path[:5])

import mmaction
print("\nmmaction version:", mmaction.__version__)
print("mmaction loaded from:", mmaction.__file__)

import mmaction.models
from mmaction.registry import MODELS

registered = sorted(MODELS._module_dict.keys())
print("\nMViT in registry:", 'MViT' in MODELS._module_dict)
print(f"Total registered models: {len(registered)}")
print("First 20:", registered[:20])

# Check if it's an editable install
import importlib.util
spec = importlib.util.find_spec('mmaction')
print("\nmmaction spec origin:", spec.origin if spec else "NOT FOUND")
