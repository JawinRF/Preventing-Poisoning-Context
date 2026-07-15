import sys
import os

# Allow `pytest memshield/tests` to be run from the repo root.
# Inserts memshield/src so `from src.memshield.X import Y` resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "memshield", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
