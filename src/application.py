# coding: utf-8
"""
FastAPI initializer
"""
import json
from typing import Callable, Dict, List, Optional, Sequence, Type, Union

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import BaseRoute

from .health import health

VERSION = ''

def check_if_system_ready():
    """
    model readiness
    ex. Returns true on initialized model found in global scope
    try:
        from ..api import res
        if "model" in res:
            return True
        else:
            return False
    except ImportError:
        return False
    """
    # TODO: Not implemented
    try:
        from .api import res
        if "model" in res:
            return True
        else:
            return False
    except ImportError:
        return False
    return False


def liveness_check():
    """returns True as long as the server has not crashed
    """
    return True


class JSONValidatorRequest(Request):
    async def body(self) -> bytes:
        if not hasattr(self, "_body"):
            body = await super().body()
            try:
                let = json.loads(body)
                assert isinstance(let, dict)
            except Exception:
                # Mask body as a byte stream that would incur JSON-decode-failure
                self._body = b""
            else:
                self._body = body
        return self._body


class JSONRoute(APIRoute):
    """
    Makes sure that requests with 'content-type: application/json'
    won't accept any invalid JSON data as body
    - Null, Empty or an empty JSON are also considered as an invalid data body.
    """

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request):
            content_type = request.headers.get("content-type")  # None if not set
            if (
                request.method and request.method == "GET" and
                ( not isinstance(content_type, str) or
                  content_type.lower() != "application/json")
            ):
                pass
            elif (
                not isinstance(content_type, str) or
                content_type.lower() != "application/json"
            ):
                raise HTTPException(
                    status_code=415,
                    detail="UNSUPPORTED_CONTENT_TYPE:Only support 'application/json'"
                )
            else:
                request = JSONValidatorRequest(request.scope, request.receive)
            return await original_route_handler(request)

        return custom_route_handler


def assert_content_type(content_type: str=Header(None)):
    """
    Assert request header to be in application/json only
    """
    # NOTE: content-type errors: boundary info is concat to the content-type value.
    # https://www.w3.org/Protocols/rfc1341/7_2_Multipart.html
    if content_type is None:
        pass
    elif (
        not isinstance(content_type, str) or
        content_type != "application/json"
    ):
        raise HTTPException(
            status_code=415,
            detail="UNSUPPORTED_CONTENT_TYPE:Only support 'application/json'"
        )


async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        # NOTE: [starlette/pydantic] middleware errors are request parsing errors
        return JSONResponse(
            status_code=400,
            content={
                "code": "BAD_REQUEST",
                "message": f"Invalid input payload: {str(e)}",
                "version": VERSION,
            },
        )


async def not_found(request: Request, exc: HTTPException):
    # NOTE: If you want a HTML response,
    # return HTMLResponse(content=HTML_404_PAGE, status_code=exc.status_code)
    return JSONResponse(
        status_code=404, content={"code": "NOT_FOUND", "message": "Unsupported", "version": VERSION}
    )


async def server_error(request: Request, exc: HTTPException):
    # NOTE: If you want a HTML response,
    # return HTMLResponse(content=HTML_500_PAGE, status_code=exc.status_code)
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_SERVER_ERROR",
            "message": "Unexpected internal exception",
            "version": VERSION,
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    try:
        # NOTE: Takes HTTPException detail in the foramt "err_code:err_message"
        if ":" in exc.detail:
            code, message = exc.detail.split(":", 1)
        else:
            code, message = "HTTP_EXCEPTION", exc.detail
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "code": "INTERNAL_SERVER_ERROR",
                "message": f"Unexpected internal exception: {str(e)}",
                "version": VERSION,
            },
        )
    else:
        return JSONResponse(
            status_code=exc.status_code, content={"code": code, "message": message, "version": VERSION}
        )


async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    err = exc.errors()[0]
    fields = ".".join([loc for loc in err["loc"] if loc != "payload"])

    # Error due to the data body itself are 400
    status = 400
    code = "BAD_REQUEST"
    message = "Invalid input payload"

    if fields != "body":
        if "type" in err["type"]:
            code = "INVALID_PARAMETER_TYPE"

        if any(k in err["type"] for k in {"value", "enum"}):
            code = "INVALID_PARAMETER_VALUE"

        status = 422
        message = f"{fields}: {err['msg']}"

    return JSONResponse(
        status_code=status,
        content={"code": code, "message": message, "version": VERSION},
    )


# NOTE
# Wrapper around FastAPI app where JSON validator and
# custom exception handlers are added.
# Returns an initialized FastAPI app instance
def FastJsonAPI(
    *,  # No positional or implicit keyword arguments allowed
    debug: bool = False,
    routes: List[BaseRoute] = None,
    title: str = "DRY-FastAPI",
    description: str = "",
    version: str = "0.1.0",
    default_response_class: Type[Response] = JSONResponse,
    middleware: Sequence[Middleware] = None,
    exception_handlers: Dict[Union[int, Type[Exception]], Callable] = None,
    on_startup: Sequence[Callable] = None,
    on_shutdown: Sequence[Callable] = None,
    health_check: Optional[Callable] = None,
):
    global VERSION
    """
    Initialize as you would a FastAPI app instance.
    on_startup, on_shutdown and other middleware,
    exception_handlers can be explicitly given
    """
    # NOTE: customize following callbacks or explicitly give custom callbacks through intialization
    if not exception_handlers:
        exception_handlers = {404: not_found, 500: server_error}

    # FastAPI app
    app = FastAPI(
        debug=debug,
        routes=routes,
        title=title,
        description=description,
        version=version,
        default_response_class=default_response_class,
        middleware=middleware,
        exception_handlers=exception_handlers,
        dependencies=[Depends(assert_content_type)],
        on_startup=on_startup,
        on_shutdown=on_shutdown,
    )
    VERSION = app.version

    # router -> JSON validator (before the request reaches the main validations)
    app.router.route_class = JSONRoute

    # error handlers
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(
        RequestValidationError, request_validation_exception_handler
    )

    #app.middleware("http")(catch_exceptions_middleware)

    app.add_api_route("/hcheck", health([liveness_check]))
    if health_check is None:
        # NOTE: model readiness check route
        health_check = check_if_system_ready
        app.add_api_route("/ready", health([health_check]))

    return app
