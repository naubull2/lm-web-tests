import torch
from typing import List, Dict, Union
from transformers import LogitsProcessor
from functools import reduce


class StopWordsLogitsProcessor(LogitsProcessor):
    def __init__(self, stopwords, tokenizer, eos_token_id, device):
        self.stopwords = stopwords
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.device = device
        self.first = True

        self.stopwords_ids = [
            self.tokenizer.encode(stopword, return_tensors="pt").to(device)
            for stopword in stopwords
        ]

    def __call__(self, input_ids, scores):
        if self.first is False:
            for stopword_ids in self.stopwords_ids:
                exist_stopword = input_ids[
                    :, -stopword_ids.size(1) :
                ] == stopword_ids.view(1, -1)
                exist_stopword = exist_stopword.sum(-1) == stopword_ids.size(1)
                for i, stop in enumerate(exist_stopword):
                    if stop.item() is True:
                        mask = torch.ones_like(scores[i])
                        mask *= -float("inf")
                        mask[self.eos_token_id] = 0
                        scores[i] += mask
        else:
            self.first = False

        self.input_ids = input_ids
        self.scores = scores
        return scores


def create_trie(context_tokens: List[int]) -> Dict[int, dict]:
    trie = {}
    for suffix in [context_tokens[i:] for i in range(len(context_tokens))]:
        node = trie
        for token in suffix:
            if token not in node:
                node[token] = {}
            node = node[token]
    return trie


def valid_next_tokens(trie: Dict[int, dict], prefix: List[int]) -> List[int]:
    return list(reduce(lambda d, k: d.get(k, {}), prefix, trie).keys())


class ExtractivePenalty(LogitsProcessor):
    def __init__(
        self,
        penalty: float,
        prefix_size: int,
        prompt_len: int,
        reference_tokens: List[int],
        eos_token_id: Union[int, List[int]],
    ) -> None:
        self.trie = create_trie(reference_tokens)  # token to apply as allowed tokens, reference
        self.penalty = penalty
        self.k = prefix_size
        self.prompt_len = prompt_len  # prompt length
        self.eos_token_id = eos_token_id
        if not isinstance(self.eos_token_id, list):
            self.eos_token_id = [self.eos_token_id]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # we can focus on the generated tokens only
        beam_prefixes = input_ids[:, self.prompt_len :]
        for i, prefix in enumerate(beam_prefixes):
            options = valid_next_tokens(self.trie, prefix.tolist()[-self.k :])
            options.extend(self.eos_token_id)  # possible next token candidates
            options = torch.tensor(options, dtype=torch.int, device=input_ids.device)
            mask = torch.isin(
                torch.arange(scores[i].numel(), device=input_ids.device), options
            )
            # Instead of complete masking, add penalty value
            scores[i][~mask] += self.penalty
        return scores


if __name__ == "__main__":
    print("building a trie")
    trie = create_trie([1, 4, 3, 1, 4, 6])
    print("find possible next tokens")
    print(valid_next_tokens(trie, [4]))
