"""
Compatibility launcher for the dashboard.

Preferred command:
    streamlit run app/dashboard.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.dashboard import main


if __name__ == "__main__":
    main()
