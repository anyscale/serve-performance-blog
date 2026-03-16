from ray import serve
import logging
logger = logging.getLogger("ray.serve")


@serve.deployment(max_ongoing_requests=100_000)
class Echo:
    def echo(self, message):
        return f"Echo: {message}"


def build_asgi_app():
    from fastapi import FastAPI

    app = FastAPI()

    handle = None

    def get_sub_deployment_handle():
        nonlocal handle
        if handle is None:
            handle = serve.get_deployment_handle(deployment_name="Echo", app_name="default", _record_telemetry=False)
        return handle

    @app.get("/echo")
    async def say_hi(message: str):
        handle = get_sub_deployment_handle()
        response = await handle.echo.remote(message)
        return {"response": response}

    return app


entrypoint = serve.deployment(max_ongoing_requests=100_000)(
    serve.ingress(build_asgi_app)()
).bind(Echo.bind())