"""
SentinX: FastAPI app, wires the routers together and serves the static dashboard.

Run (from repo root):
    uvicorn api.main:app --reload
Docs: http://localhost:8000/docs   Dashboard: http://localhost:8000/dashboard/
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.data import ROOT
from api.routers import countries, overview

app = FastAPI(title="SentinX API", version="0.2.0",
              description="Country-level supply-chain disruption risk (Asia-Pacific prototype)")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(overview.router)
app.include_router(countries.router)

# static dashboard served from the same origin, so no CORS setup is needed for the demo
DASH = ROOT / "dashboard"
if (DASH / "index.html").exists():
    app.mount("/dashboard", StaticFiles(directory=DASH, html=True), name="dashboard")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/dashboard/")