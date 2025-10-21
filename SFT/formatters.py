"""Prompt/response formatting utilities (6.1).
We keep a very simple template with clear separators.
This class is for feeding the inputs to the LLM for instruction finetuning.
The instructions and target response pairs are taken from the dataset.
"""
from dataclasses import dataclass

template = (
    "<s>\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}</s>"
)

@dataclass
class Example:
    instruction: str
    response: str

def format_example(ex: Example) -> str: # This is for training time
    return template.format(instruction=ex.instruction.strip(), response=ex.response.strip())


def format_prompt_only(instruction: str) -> str: # this is inference time
    return template.format(instruction=instruction.strip(), response="")