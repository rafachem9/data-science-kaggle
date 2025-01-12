import os
from pathlib import Path


# Get the current directory with pathlib
PROJECT_DIR = os.getcwd()
PROJECT_DIR_FIX = Path(PROJECT_DIR).resolve().parents[0]

DATA_DIR = os.path.join(PROJECT_DIR, 'data')
DATABASE_DIR = os.path.join(PROJECT_DIR, 'database')
CONFIG_DIR = os.path.join(PROJECT_DIR, 'settings', 'settings.cfg')
SQL_DIR = os.path.join(PROJECT_DIR, 'sql')


