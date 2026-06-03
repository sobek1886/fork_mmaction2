#!/usr/bin/env python3
"""Diagnose mmaction2 installation and MViT registry status."""

import sys
import importlib.util
import os

print("Python:", sys.executable)
print("\nFull sys.path:")
for p in sys.path:
    print(" ", p)

# Check if mmaction/ directory still exists in fork_mmaction2
fork_mmaction2_dir = os.path.expanduser("~/fork_mmaction2")
mmaction_stub = os.path.join(fork_mmaction2_dir, "mmaction")
print(f"\nmmaction/ in fork_mmaction2: {'EXISTS' if os.path.isdir(mmaction_stub) else 'gone (good)'}")

# Check Fork_SignCLIP for mmaction/
fork_signclip_dir = os.path.expanduser("~/Fork_SignCLIP")
signclip_mmaction = os.path.join(fork_signclip_dir, "mmaction")
print(f"mmaction/ in Fork_SignCLIP:  {'EXISTS' if os.path.isdir(signclip_mmaction) else 'not present'}")

# Find which mmaction Python will load
spec = importlib.util.find_spec('mmaction')
print(f"\nmmaction resolves to: {spec.origin if spec else 'NOT FOUND'}")

# Import and inspect
import mmaction
print(f"mmaction __file__: {mmaction.__file__}")
print(f"mmaction attributes: {[x for x in dir(mmaction) if not x.startswith('_')]}")

try:
    print(f"mmaction version: {mmaction.__version__}")
except AttributeError:
    print("mmaction has no __version__ — wrong module is being loaded")

# Try importing models
try:
    import mmaction.models
    from mmaction.registry import MODELS
    registered = sorted(MODELS._module_dict.keys())
    print(f"\nMViT in registry: {'MViT' in MODELS._module_dict}")
    print(f"Total registered models: {len(registered)}")
    if registered:
        print("First 20:", registered[:20])
except Exception as e:
    print(f"\nFailed to import mmaction.models: {e}")
