"""Streamlit Community Cloud entrypoint for the CardioRisk website.

Community Cloud installs the dependency file found next to its entrypoint first,
so this folder carries the website's own pinned requirements.txt (core packages
plus Streamlit) while the repository root stays set up for the Vercel API.

Settings come from the app's Secrets; root-level keys are copied into the
environment before the app's modules read them. The site only needs
HEART_API_URL (the deployed API); it scores and saves through that API.
"""

import os
import runpy
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except FileNotFoundError:
    pass  # no secrets configured (e.g. running locally with .env instead)

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
