"""
Root entry point for the Social Media Sentiment Dashboard.

Run:
    streamlit run main.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from app.dashboard import main


if __name__ == "__main__":
    main()
