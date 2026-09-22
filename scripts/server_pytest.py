import os
import sys

import pytest


result = pytest.main(sys.argv[1:])
sys.stdout.flush()
sys.stderr.flush()
# Exit only after pytest completes; legacy fixtures can leave idle SQLite worker threads alive.
os._exit(int(result))
