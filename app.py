import os
import sys

# Add the 'backend' folder to the python path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

# Now we can mock 'app.py' to export what Render expects
from app.main import app
