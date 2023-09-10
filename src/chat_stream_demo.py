# coding:utf-8
"""
Simple Gradio chatbot demo with local API calls
__author__ = 'naubull2 (feasible@kakao.com)'
"""
import os
import gradio as gr
from pathlib import Path
from .model import GenerationModel

# device = os.getpid() % torch.cuda.device_count()
model = GenerationModel(
    os.environ.get("MODEL_NAME", "oasst_llama"),
    os.environ.get("MODEL", ""),
    half_precision=False,
    device=0
)


no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)


def user(message, history=[]):
    return history + [[message, None]]


def regenerate(history=[]):
    history[-1][1] = ""
    return history


def chatbot(
    system_prompt,
    message,
    history,
    temperature=0.7,
    top_p=1.0,
    max_tokens=256,
    seed=0,
    no_repeat_size=16,
    contrastive_penalty=None,
    extractive_penalty=None,
    stop_tokens=None,
):
    def flatten(lst):
        ret = []
        for tup in lst:
            ret.extend(tup)
        ret = [r.replace("<p>", "").replace("</p>", "").strip("\n") for r in ret]
        return ret

    def create_messages(system_prompt, lst):
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        for i, m in enumerate(lst):
            messages.append(
                {"role": ("user" if i % 2 == 0 else "assistant"), "content": m}
            )
        return messages

    stop_tokens = [p.strip(' ') for p in stop_tokens.split(",")]
    default_gen_opts = {
        "temperature": temperature,
        "top_p": top_p,
        "no_repeat_ngram_size": no_repeat_size,
        "seed": seed,
        "contrastive_alpha": contrastive_penalty,
        "extractive_penalty": extractive_penalty,
        "stop_tokens": stop_tokens,
    }
    try:
        history[-1][1] = ""
        q = flatten(history)
        if not q[-1]:
            q = q[:-1]

        cursor = "▌"
        stop_tokens.sort(key=lambda x: len(x), reverse=True)
        history[-1][1] += cursor
        for word in model.generate_generator(
            create_messages(system_prompt, q),
            max_tokens=max_tokens,
            **default_gen_opts,
        ):
            for stop_t in stop_tokens:
                if stop_t in word:
                    word = word.rstrip("\n").replace(stop_t, "")
            history[-1][1] = history[-1][1].rstrip(cursor)
            history[-1][1] += word + cursor
            yield ("", history) + (disable_btn,) * 2

        history[-1][1] = history[-1][1].rstrip(cursor)
        for stop_t in stop_tokens:
            history[-1][1] = history[-1][1].rstrip(stop_t)

        yield ("", history) + (enable_btn,) * 2

    except Exception:
        pass


with gr.Blocks() as demo:
    title = gr.Textbox(
        value=Path(os.environ.get("MODEL", "")).stem,
        label="Model name",
        interactive=False,
    )
    bot = gr.Chatbot(
        label="Scroll down and start chatting",
        visible=True,
        height=550,
    )

    with gr.Accordion("more options", open=False):
        contrastive_penalty = gr.Slider(
            minimium=0.0, value=0.0, maximum=0.9, step=0.05, label="Contrastive decoding alpha",
            info="Increase to create semantic coherency"
        )
        extractive_penalty = gr.Slider(
            minimum=-10.0, value=0.0, maximum=10.0, step=0.1, label="Token extractiveness penalty",
            info="Increase to be less extractive"
        )
        temperature = gr.Slider(
            minimum=0.05, value=0.05, maximum=1.0, step=0.05, label="Temperature"
        )
        top_p = gr.Slider(minimum=0.05, value=1.0, maximum=1.0, step=0.05, label="Top-P")
        max_tokens = gr.Slider(
            minimum=1, value=512, maximum=512, step=1, label="Max Tokens"
        )
        seed = gr.Slider(minimum=0, value=0, maximum=2030, step=1, label="Seed")
        no_repeat_size = gr.Slider(
            minimum=0, value=0, maximum=32, step=1, label="No repeat N-grams"
        )
        stop_tokens = gr.Textbox(
            value="</s>, <|endoftext|>",
            label="Stop tokens (separated with ',')",
            interactive=True,
        )

    system_prompt = gr.Textbox(
        value="",
        label="System prompt"
    )
    inp = gr.Textbox(placeholder="Input text and press enter", label="Input", autofocus=True)

    with gr.Row():
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button("🗑️  Clear", interactive=False)

    btn_list = [regenerate_btn, clear_btn]

    inp.submit(user, [inp, bot], [bot], queue=False).then(
        chatbot,
        [
            system_prompt,
            inp,
            bot,
            temperature,
            top_p,
            max_tokens,
            seed,
            no_repeat_size,
            contrastive_penalty,
            extractive_penalty,
            stop_tokens,
        ],
        [inp, bot] + btn_list,
    )

    def clear():
        return ([],) + (disable_btn,)*2

    # Register listeners
    regenerate_btn.click(regenerate, bot, [bot], queue=False).then(
        chatbot,
        [
            system_prompt,
            inp,
            bot,
            temperature,
            top_p,
            max_tokens,
            seed,
            no_repeat_size,
            contrastive_penalty,
            extractive_penalty,
            stop_tokens,
        ],
        [inp, bot] + btn_list,
    )
    clear_btn.click(clear, None, outputs=[bot] + btn_list, queue=False)

demo.queue()
