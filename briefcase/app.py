from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from briefcase import __version__
from briefcase.api.routes import router
from briefcase.db import init_db

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app() -> FastAPI:
    app = FastAPI(title="Briefcase", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        init_db()

    app.include_router(router)

    if _WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=_WEB_DIR), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(_WEB_DIR / "index.html")

    return app


app = create_app()
