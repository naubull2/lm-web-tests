# coding: utf-8
import threading

import gradio as gr
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .application import FastJsonAPI
from .chat_stream_demo import demo, model
from .schema import GenerationPayload


app = FastJsonAPI(
    title="LM-web-tester",
    description="Simple Language Model API",
    version="0.1.0",
    debug=True,
)

res = dict()


@app.on_event("startup")
def init_model():
    res["lock"] = threading.Lock()
    res["model"] = model
    return res


@app.post("/v1/chat/completion")
async def v1_chat_completion(request: Request, payload: GenerationPayload = None):
    output_obj = {}
    try:
        res["lock"].acquire()
        output_obj.update(
            res["model"].generate(
                payload.messages,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
                num_return_sequences=payload.n,
                stop_tokens=payload.stop,
                repetition_penalty=payload.repetition_penalty,
                extractive_penalty=payload.extractive_penalty,
                prefix_k=payload.prefix_k,
                contrastive_alpha=payload.contrastive_alpha,
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"INTERNAL_SERVER_ERROR:Internal server error: {str(e)}",
        )
    finally:
        res["lock"].release()

    return JSONResponse(status_code=200, content=output_obj)


@app.post("/v1/chat/stream")
async def v1_chat_stream(request: Request, payload: GenerationPayload = None):
    return StreamingResponse(
        res["model"].stream_generate(
            payload.messages,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
            num_return_sequences=payload.n,
            stop_tokens=payload.stop,
            repetition_penalty=payload.repetition_penalty,
            extractive_penalty=payload.extractive_penalty,
            prefix_k=payload.prefix_k,
            contrastive_alpha=payload.contrastive_alpha,
        ),
        media_type="text/plain",
    )


app = gr.mount_gradio_app(app, demo, path="/demo")
