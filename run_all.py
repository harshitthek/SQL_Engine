"""Master Pipeline Runner for Text-to-SQL Data Preparation (Tasks 02–09).

Delegates to data.processing.run_pipeline.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.processing.run_pipeline import main

if __name__ == "__main__":
    main()
