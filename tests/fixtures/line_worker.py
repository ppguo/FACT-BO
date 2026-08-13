#!/usr/bin/env python3
"""Tiny protocol worker used by evaluator tests."""

import sys


print("READY", flush=True)
for line in sys.stdin:
    request = line.strip()
    if request == "EXIT":
        break
    if request:
        print(sum(float(value) for value in request.split(",")), flush=True)
