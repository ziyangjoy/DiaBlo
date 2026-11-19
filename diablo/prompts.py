"""Prompt templates shared by data processing and evaluation.

The training templates are copied verbatim (including the odd indentation) from the upstream
processors, which in turn copy LLM-Adapters (commonsense) and MetaMath. Evaluation must give
the model exactly the prefix it saw in training, up to the response. Upstream's eval scripts
did not: evaluate_commonsense.py sent only "<instruction>\\n", and evaluate_math.py indented
the template with 16 spaces where training used 12. `*_eval_prompt` fix that by cutting the
training template at the response slot. Pass `style="upstream"` to get the released eval
script's prompt instead.
"""

# ── Commonsense-170k (LLM-Adapters format) ───────────────────────────────────

def commonsense_train_prompt(instruction: str, input: str = "", output: str = "") -> str:
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                {output}"""
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Response:
                {output}"""


def commonsense_eval_prompt(instruction: str, style: str = "train") -> str:
    if style == "upstream":
        return f"{instruction}\n"
    return commonsense_train_prompt(instruction, "", "")


# ── MetaMathQA / GSM8K / MATH ────────────────────────────────────────────────

def metamath_train_prompt(query: str, response: str = "") -> str:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

            ### Instruction:
            {query}

            ### Response: Let's think step by step.
            {response}
            """


def metamath_eval_prompt(query: str, style: str = "train") -> str:
    if style == "upstream":
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {query}

                ### Response: Let's think step by step.
                """
    full = metamath_train_prompt(query, "\x00")
    return full[: full.index("\x00")]
