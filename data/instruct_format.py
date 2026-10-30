"""
The ONE place that turns (instruction, context, response) into tokens.
Instruction tuning and evaluation (Person C) both import from here, so every
variant is trained and tested on exactly the same prompt format.

    <|user|>
    {instruction}
    <|context|>            <- only when there is a context
    {context}
    <|assistant|>
    {response}<|endoftext|>

Prompt and response are tokenized SEPARATELY and concatenated. That is also
what happens at inference (encode the prompt, then generate), so the model is
trained on exactly the token boundary it will see when answering.
"""
from config import CONTEXT_BLOCK_TEMPLATE, EOT, PAD, PROMPT_TEMPLATE, RESPONSE_TEMPLATE


def build_prompt(instruction: str, context: str = "") -> str:
    context = (context or "").strip()
    block = CONTEXT_BLOCK_TEMPLATE.format(context=context) if context else ""
    return PROMPT_TEMPLATE.format(instruction=instruction.strip(), context_block=block)


def encode_prompt(tok, instruction: str, context: str = "") -> list:
    """Token ids to feed the model before generating an answer."""
    return tok.encode(build_prompt(instruction, context)).ids


def encode_example(tok, instruction: str, context: str, response: str):
    """Returns (ids, prompt_len). Loss is applied to ids[prompt_len:] only
    (the response and its closing <|endoftext|>)."""
    p = encode_prompt(tok, instruction, context)
    r = tok.encode(RESPONSE_TEMPLATE.format(response=response.strip())).ids
    return p + r, len(p)


def special_ids(tok) -> dict:
    return {"eot": tok.token_to_id(EOT), "pad": tok.token_to_id(PAD)}
