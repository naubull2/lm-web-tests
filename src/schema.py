# coding: utf-8
from typing import List, Optional, Union

from pydantic import BaseModel


class GenerationPayload(BaseModel):
    class Config:
        extra = "ignore"

    messages: List[dict]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = 1.0
    extractive_penalty: Optional[float] = 0.0
    prefix_k: Optional[int] = None
    contrastive_alpha: Optional[float] = None
    # do_sample is always true as beam / greedy is not an option here

