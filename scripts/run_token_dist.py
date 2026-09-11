"""Task 06 Runner: Delegates to data.processing.run_pipeline."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.processing.run_pipeline import run_task_06 as main

if __name__ == "__main__":
    main()
