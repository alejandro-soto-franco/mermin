"""Configure pytest to find mermin modules."""
import sys
from pathlib import Path

# Add the python directory to the path so we can import mermin
python_dir = Path(__file__).parent.parent / "python"
sys.path.insert(0, str(python_dir))


def pytest_configure(config):
    """Register markers declared canonically in mermin-py/pyproject.toml.

    That file is the package's own pytest config, but `tests/` sits at the
    repository root with no root-level pyproject.toml or pytest.ini for
    pytest's own config discovery to find, so a bare `pytest tests/` never
    reads it. Mirroring the declaration here is what actually silences the
    unknown-mark warning for that invocation.
    """
    config.addinivalue_line(
        "markers", "corpus: requires the ASF-EX1 mermin-corpus volume mounted"
    )
