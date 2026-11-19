@echo off
rem Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)
rem Ensure dependencies are installed (silent)
python -m pip install -q fastapi uvicorn[standard] python-multipart torch

echo Starting VISHA FastAPI server on http://0.0.0.0:8000 ...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
