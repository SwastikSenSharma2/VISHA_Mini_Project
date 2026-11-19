from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os

# Load antivenom mapping (generated earlier)
try:
    from .antivenom_lookup import ANTIVENOM_MAP
except ImportError:
    ANTIVENOM_MAP = {}

app = FastAPI(title="VISHA Antivenom Predictor")

# Mount static files (if any)
static_path = Path(__file__).parent / "static"
if static_path.is_dir():
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# Serve a simple HTML form
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <title>VISHA Antivenom Predictor</title>
        <style>
            body {font-family: Arial, sans-serif; margin: 40px;}
            label, input {display:block; margin:10px 0;}
            button {padding:8px 16px;}
        </style>
    </head>
    <body>
        <h1>Antivenom Lookup</h1>
        <form action='/predict' method='post'>
            <label for='species'>Snake Species (exact name from database)</label>
            <input type='text' id='species' name='species' required />
            <button type='submit'>Get Antivenom</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# API endpoint that also works with the form submission
@app.post("/predict")
async def predict_antivenom(species: str = Form(...)):
    antivenom = ANTIVENOM_MAP.get(species.strip())
    if antivenom:
        return JSONResponse(content={"species": species, "antivenom": antivenom})
    else:
        return JSONResponse(content={"error": f"No antivenom entry found for '{species}'."}, status_code=404)
