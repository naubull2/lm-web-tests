# lm-web-tests

A simple web demo, chat-completion API and streamed-chat-completion API to test various language model decoding methods and parameters.



For a better optimized generation, serving multiple workers and model comparisons in general I would suggest the following two repos.

- [lm-sys/FastChat](https://github.com/lm-sys/FastChat) 
- [NVIDIA/FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [vllm-project/vllm](https://github.com/vllm-project/vllm)

## Supported Methods

- Contrastive Decoding : 
  - Huggingface implementation of the paper "Contrastive Search Is What You Need For Neural Text Generation [Yixuan Su](https://arxiv.org/search/cs?searchtype=author&query=Su,+Y), [Nigel Collier](https://arxiv.org/search/cs?searchtype=author&query=Collier,+N)"
- Extractive Decoding :
  - The basic idea is from [Yoni Gottesman's method](https://yonigottesman.github.io/2023/08/10/extractive-generative.html) where I generalized a bit.
    - Strict masking of "non-extractive" tokens from the context, is generalized into adding a "penalty score". When the penalty score is large enough ( some float range of -5 to -inf), it will be the same as the originally suggested method.
    - Instead of matching entire prefix of the generated sequence, I used n-gram prefix instead. (set `n=3` as baseline)



# How-to-use

There's a `run.sh`included, where it's straight forward how to run your own model.

- `MODEL_NAME`: Chat conversation template format name. (`conversation.py`is from [lm-sys/fatchat](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py))
- `MODEL_PATH`: Directory of the model's binary and tokenizer files.

## Requirements

```text
accelerate==0.22.0
fastapi==0.81.0
gradio>=3.27.0
requests==2.28.1
uvicorn==0.18.3
transformers==4.32.1
```





# References

- Conversation template script from [lm-sys/FastChat](https://github.com/lm-sys/FastChat) 
- Extractive question answering from Yoni Gottesman's [blog](https://yonigottesman.github.io/2023/08/10/extractive-generative.html)



