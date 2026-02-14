import time
import json
import re
from typing import Optional, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer


class QAgent(object):

    def __init__(self, **kwargs):
        model_name = "Qwen/Qwen3-4B"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    def clean_json(self, text: str):
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            return text

        json_str = text[start:end + 1]
        json_str = json_str.replace("\n", " ")

        try:
            parsed = json.loads(json_str)

            if "answer" in parsed and isinstance(parsed["answer"], str):
                parsed["answer"] = parsed["answer"].strip().upper()[0]

            if "choices" in parsed:
                cleaned_choices = []
                for choice in parsed["choices"]:
                    cleaned = re.sub(r'^([A-D]\))\s*\1', r'\1', choice)
                    cleaned_choices.append(cleaned)
                parsed["choices"] = cleaned_choices

            return parsed

        except:
            return text

    def generate_response(
        self,
        message: Union[str, List[str]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ):

        if system_prompt is None:
            system_prompt = (
                "You are a strict logical reasoning question generator. "
                "You must output ONLY valid JSON. "
                "Do not include markdown. "
                "Do not include extra text. "
                "Each choice must start exactly with A), B), C), or D). "
                "Do NOT repeat the option letter inside the option text. "
                "Answer must be exactly one of A, B, C, or D."
            )

        if isinstance(message, str):
            message = [message]

        texts = []

        for msg in message:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            texts.append(text)

        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        tgps_show_var = kwargs.get("tgps_show", False)

        if tgps_show_var:
            start_time = time.time()

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 256),
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        if tgps_show_var:
            generation_time = time.time() - start_time

        batch_outs = []

        if tgps_show_var:
            token_len = 0

        for input_ids, generated_sequence in zip(
            model_inputs.input_ids, generated_ids
        ):
            output_ids = generated_sequence[len(input_ids):]

            if tgps_show_var:
                token_len += len(output_ids)

            content = self.tokenizer.decode(
                output_ids,
                skip_special_tokens=True
            ).strip()

            parsed = self.clean_json(content)
            batch_outs.append(parsed)

        if tgps_show_var:
            return (
                batch_outs[0] if len(batch_outs) == 1 else batch_outs,
                token_len,
                generation_time,
            )

        return batch_outs[0] if len(batch_outs) == 1 else batch_outs, None, None
