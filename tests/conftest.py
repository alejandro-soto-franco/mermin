"""Configure pytest to find mermin modules."""
import sys
from pathlib import Path

# Add the python directory to the path so we can import mermin
python_dir = Path(__file__).parent.parent / "python"
sys.path.insert(0, str(python_dir))
