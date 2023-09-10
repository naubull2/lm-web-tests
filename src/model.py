# coding: utf-8
import time
import secrets

import torch
import random
import numpy as np
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LogitsProcessorList,
)
from .streamers import TextIteratorStreamer
from .generation_utils import StopWordsLogitsProcessor, ExtractivePenalty
from .conversation import get_conv_template


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class GenerationModel(object):
    def __init__(
        self,
        model_name,
        name_or_path,
        cache_dir=None,
        load_in_8bit=False,
        half_precision=False,
        device=None,
    ):
        """Load a hf text geneation model"""
        kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        self.model_name = model_name
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, truncation_side="left", padding_side="left", **kwargs
        )
        config = AutoConfig.from_pretrained(name_or_path, **kwargs)

        kwargs["torch_dtype"] = config.torch_dtype

        if torch.cuda.is_available():
            kwargs = kwargs.copy()
            kwargs["load_in_8bit"] = load_in_8bit
            if device is None:
                kwargs["device_map"] = "auto"

        if half_precision:
            kwargs["torch_dtype"] = torch.float16
        elif load_in_8bit:
            kwargs["load_in_8bit"] = True
        elif device == "cpu":
            # some operations are not supported in half-precision on CPU
            kwargs["torch_dtype"] = torch.float32
        else:
            kwargs["torch_dtype"] = config.torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
        if device is not None and isinstance(device, int):
            self.model.to(f"cuda:{device}")
            # self.model.to("mps")

        if "compile" in dir(torch):
            self.model = torch.compile(self.model, backend="inductor")
        self.model.eval()
        self.device = self.model.device

        if not self.model.can_generate():
            raise TypeError(f"{name_or_path} is not a text generation model")

    def compose_prompt_tokens(self, messages, max_len, max_output):
        conv = get_conv_template(self.model_name)
        for c in messages:
            if c["role"] == "system":
                conv.system = c["content"]
            elif c["role"] == "user":
                conv.append_message(conv.roles[0], c["content"])
            else:
                conv.append_message(conv.roles[1], c["content"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        max_input_len = max_len - max_output
        inputs = self.tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=max_input_len,
            truncation=True,
            return_tensors="pt",
        )
        return inputs, prompt

    def create_generation_params(
        self,
        input_ids,
        temperature,
        top_p,
        contrastive_alpha,
        repetition_penalty,
        no_repeat_ngram_size,
        max_tokens,
        num_return_sequences,
        stop_tokens,
        extractive_penalty,
        prefix_k,
        num_beams,
        do_sample,
    ):
        params = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "temperature": temperature,
        }
        prompt_len = len(input_ids)

        custom_logits_processors_list = LogitsProcessorList()
        if stop_tokens:
            stopwords_logits_processor = StopWordsLogitsProcessor(
                stop_tokens,
                self.tokenizer,
                self.tokenizer.eos_token_id,
                input_ids.device,
            )
            custom_logits_processors_list.append(stopwords_logits_processor)

        if extractive_penalty:
            prefix_ngram = prefix_k if isinstance(prefix_k, int) else 3
            extractive_penalty_logits = ExtractivePenalty(
                penalty=extractive_penalty,
                prefix_size=prefix_ngram,
                prompt_len=prompt_len,
                reference_tokens=input_ids,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            custom_logits_processors_list.append(extractive_penalty_logits)

        if contrastive_alpha:
            params.update({"penalty_alpha": contrastive_alpha, "top_k": 8})

        if do_sample:  # resolve conflicting options if any
            num_beams = 1

        params.update(
            {
                "do_sample": do_sample,
                "num_beams": num_beams,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "max_new_tokens": max_tokens,
                "num_return_sequences": num_return_sequences,
                "logits_processor": custom_logits_processors_list,
            }
        )

        return params

    def generate(
        self,
        messages,
        max_tokens=256,
        do_sample=True,
        num_beams=1,
        temperature=0.8,
        top_p=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        num_return_sequences=1,
        contrastive_alpha=None,
        extractive_penalty=None,
        prefix_k=None,
        stop_tokens=None,
        device=None,
        seed=0,
    ):
        """
        messages    : (list) message objects in the following form.
                {
                    'role': (str) [system|user|assistant],
                    'content': (str) message
                }
        """
        set_random_seed(seed)

        try:
            max_len = self.model.config.max_position_embeddings
        except AttributeError:
            max_len = 2048

        device = self.device if device is None else device

        inputs, prompt = self.compose_prompt_tokens(messages, max_len, max_tokens)
        inputs = inputs.to(device)
        prompt_len = len(inputs["input_ids"][0])

        gen_params = self.create_generation_params(
            inputs["input_ids"][0],
            temperature,
            top_p,
            contrastive_alpha,
            repetition_penalty,
            no_repeat_ngram_size,
            max_tokens,
            num_return_sequences,
            stop_tokens,
            extractive_penalty,
            prefix_k,
            num_beams,
            do_sample,
        )

        with torch.no_grad():
            torch.cuda.empty_cache()  # for memory safety
            outputs = self.model.generate(**inputs, **gen_params)

        choices = []
        n_tokens = 0
        for i, o in enumerate(outputs):
            n_tokens += len(o) - prompt_len
            finish = (
                "stop"
                if o[-1].item()
                in {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
                else "length"
            )

            response = self.tokenizer.decode(
                o[inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            )
            response = response.split(self.tokenizer.eos_token, 1)[0]
            choices.append(
                {
                    "message": {"role": "assistant", "content": response},
                    "index": i,
                    "finish_reason": finish,
                }
            )
        output_template = {
            "id": f"cmpl-{secrets.token_hex(12)}",
            "object": "chat_completion",
            "created": round(time.time()),
            "model": self.model.name_or_path,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": n_tokens,
                "total_tokens": prompt_len + n_tokens,
            },
        }
        return output_template

    # NOTE: For /v1/chat/stream API
    async def stream_generate(
        self,
        messages,
        max_tokens=256,
        do_sample=True,
        num_beams=1,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        num_return_sequences=1,
        contrastive_alpha=None,
        extractive_penalty=None,
        prefix_k=None,
        stop_tokens=None,
        device=None,
        seed=0,
    ):
        set_random_seed(seed)

        try:
            max_len = self.model.config.max_position_embeddings
        except AttributeError:
            max_len = 2048

        device = self.device if device is None else device

        inputs, prompt = self.compose_prompt_tokens(messages, max_len, max_tokens)
        inputs = inputs.to(device)

        gen_params = self.create_generation_params(
            inputs["input_ids"][0],
            temperature,
            top_p,
            contrastive_alpha,
            repetition_penalty,
            no_repeat_ngram_size,
            max_tokens,
            num_return_sequences,
            stop_tokens,
            extractive_penalty,
            prefix_k,
            num_beams,
            do_sample,
        )

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            **gen_params
        )
        with torch.no_grad():
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            for output in streamer:
                if not output:
                    continue
                yield output

    # NOTE: for gradio
    def generate_generator(
        self,
        messages,
        max_tokens=256,
        do_sample=True,
        num_beams=1,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        num_return_sequences=1,
        contrastive_alpha=None,
        extractive_penalty=None,
        prefix_k=None,
        stop_tokens=None,
        device=None,
        seed=0,
    ):
        set_random_seed(seed)

        try:
            max_len = self.model.config.max_position_embeddings
        except AttributeError:
            max_len = 2048

        device = self.device if device is None else device

        inputs, prompt = self.compose_prompt_tokens(messages, max_len, max_tokens)
        inputs = inputs.to(device)

        gen_params = self.create_generation_params(
            inputs["input_ids"][0],
            temperature,
            top_p,
            contrastive_alpha,
            repetition_penalty,
            no_repeat_ngram_size,
            max_tokens,
            num_return_sequences,
            stop_tokens,
            extractive_penalty,
            prefix_k,
            num_beams,
            do_sample,
        )

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            **gen_params
        )
        with torch.no_grad():
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            for output in streamer:
                if not output:
                    continue
                yield output
