import os
from ray import serve
from api.routes import create_routes
from starlette.responses import StreamingResponse, Response
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from inference.engine import HTREngine
from ray.serve.config import HTTPOptions

MAINTENANCE_FILE = "/tmp/maintenance"

app = FastAPI()

class MaintenanceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if os.path.exists(MAINTENANCE_FILE):
            return Response(
                content="Service en maintenance. Veuillez réessayer plus tard.",
                status_code=200,
                headers={"Retry-After": "3600"},
            )
        return await call_next(request)

app.add_middleware(MaintenanceMiddleware)
# Configuration CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # En prod, liste les domaines autorisés
#     # allow_origins=["https://ton-site.com", "https://autre-site.com"],
#     allow_credentials=True,
#     allow_methods=["*"],  # Ou ["GET", "POST"]
#     allow_headers=["*"],
# )

@serve.deployment(ray_actor_options={"num_gpus": 1}, logging_config={"enable_access_log": False})
@serve.ingress(app)
class VLLMPredictDeployment:
    def __init__(self, **engine_kwargs):
        import logging
    
        class FilterHealthCheck(logging.Filter):
            def filter(self, record):
                msg = record.getMessage()
                return not ("GET / 200" in msg or "GET / 404" in msg)
        logging.getLogger("ray.serve").addFilter(FilterHealthCheck())
        self.engine = HTREngine(name = "QWEN2.5-VL-3B-01122025",**engine_kwargs)
    @app.post("/transcribe")
    async def transcribe(self, request: Request) -> Response:
        return await self.engine.transcribe(request)

    @app.get("/")
    async def health(self):
        return Response(content="ok", status_code=200)
    
    @app.post("/transcribe/stream")
    async def transcribe_stream(self, request: Request) -> StreamingResponse:
        return await self.engine.transcribe(request, stream=True)
    
    @app.post("/retranscribe")
    async def retranscribe(self, request: Request) -> Response:
        return await self.engine.retranscribe(request)
    
    @app.get("/health")
    async def health(self):
        return {"status": "ok"}
    
    @app.get("/model_info")
    async def model_info(self):
        return self.engine.get_model_info()
    
if __name__ == "__main__":
    deployment = VLLMPredictDeployment.bind(
        model="/home/fizainef/LLM/Weights/Qwen2.5-VL-3B_tsv_grpo-27032026",#"/home/fizainef/LLM/Weights/Qwen2-5-VL-3B-GRPO-TSV",
        max_num_seqs=90,
        max_model_len=8192,
        max_num_batched_tokens=16384*2,
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        enable_chunked_prefill=True,
        limit_mm_per_prompt={"image": 1}
    )
    # serve.start(http_options=HTTPOptions(host="0.0.0.0",
    #                                      port=443,
    #                                      tls_cert="/home/fizainef/LLM/cert.pem",
    #                                      tls_key="/home/fizainef/LLM/key.pem"))
#     serve.start(http_options={
#     "host": "127.0.0.1",
#     "port": 8000
# })
    serve.run(deployment, blocking=True)