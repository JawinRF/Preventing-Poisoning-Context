import sys
import os

# Allow tests to be run from the repo root: `pytest memshield/tests`
# Inserts memshield/src so `from src.memshield.X import Y` resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
