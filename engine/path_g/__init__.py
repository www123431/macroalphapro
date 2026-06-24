"""
engine/path_g/ — Path G VIX Term Structure with Vol-Targeted Position v1.

Pre-registration: docs/spec_path_g_vix_voltgt_v1.md (id=66 hash 9a7ca1fe)

Cheng 2019 RFS VIX premium signal (same as Path F) + Moskowitz-Ooi-Pedersen
2012 JFE vol-targeting (target 12% portfolio vol annualized, canonical
convention).

Distinct hypothesis from Path F: position-sizing rule structurally
different (vol-scaled vs binary). Same VIX/VIX3M signal + Cheng 2019
anchor. Locked ex-ante per 3-test HARKing framework.
"""
from __future__ import annotations
