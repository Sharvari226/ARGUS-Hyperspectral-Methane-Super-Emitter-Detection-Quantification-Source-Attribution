import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.system("python -m streamlit run dashboard/app.py")