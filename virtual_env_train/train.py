"""Helper script to train the DCSASS MIL model in an isolated environment."""
from __future__ import annotations

import importlib
import sys

if __name__ == "__main__":
    module = importlib.import_module("surveillance_tf.train.train_mil_ucfcrime")
    sys.exit(module.main())
