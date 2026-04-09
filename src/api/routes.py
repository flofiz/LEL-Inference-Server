
from fastapi import FastAPI
from starlette.responses import StreamingResponse, Response
from starlette.requests import Request
from ray import serve
from inference.engine import HTREngine
from starlette.applications import Starlette





def create_routes(app: FastAPI, engine: HTREngine):
    
    @app.post("/transcribe")
    async def transcribe( request: Request) -> Response:
        return await engine(request)
    
    @app.get("/health")
    async def health():
        return {"status": "ok"}