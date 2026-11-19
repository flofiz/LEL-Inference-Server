"""Allow the emailed demo HTML to read only the public HTML stream."""

from starlette.middleware.base import BaseHTTPMiddleware


class PortableDemoCors(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.method == "POST" and request.url.path == "/transcribe/html/stream":
            response.headers.add_vary_header("Origin")
            if request.headers.get("Origin") == "null":
                response.headers["Access-Control-Allow-Origin"] = "null"
        return response
