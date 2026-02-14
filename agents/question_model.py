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

    # ---------------- JSON CLEANER ---------------- #
    def clean_json(self, text: str):
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            return None

        json_str = text[start:end + 1]
        json_str = json_str.replace("\n", " ")

        try:
            parsed = json.loads(json_str)

            required_keys = ["topic", "question", "choices", "answer", "explanation"]
            for key in required_keys:
                if key not in parsed:
                    return None

            if isinstance(parsed["answer"], str):
                parsed["answer"] = parsed["answer"].strip().upper()[0]

            if isinstance(parsed["choices"], list):
                cleaned_choices = []
                for choice in parsed["choices"]:
                    cleaned = re.sub(r'^([A-D]\))\s*\1', r'\1', choice)
                    cleaned_choices.append(cleaned)
                parsed["choices"] = cleaned_choices

            return parsed

        except:
            return None

    # ---------------- GENERATION ---------------- #
    def generate_response(
        self,
        message: Union[str, List[str]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ):

        if system_prompt is None:
            system_prompt = (
                "Generate ONE logical reasoning MCQ in strict JSON format. "
                "Return ONLY a valid JSON object with keys: "
                "topic, question, choices, answer, explanation. "
                "Each choice must start with A), B), C), or D). "
                "Answer must be A, B, C, or D."
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

        start_time = time.time()

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens= 200,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generation_time = time.time() - start_time

        batch_outs = []
        token_len = 0

        for input_ids, generated_sequence in zip(
            model_inputs.input_ids, generated_ids
        ):
            output_ids = generated_sequence[len(input_ids):]
            token_len += len(output_ids)

            content = self.tokenizer.decode(
                output_ids,
                skip_special_tokens=True
            ).strip()
            print("\nRAW OUTPUT:\n", content)

            parsed = self.clean_json(content)

            if parsed is not None:
                batch_outs.append(parsed)

        # -------- FIXED RETURN BLOCK -------- #
        if tgps_show_var:
            return (
                batch_outs[0] if len(batch_outs) == 1 else batch_outs,
                token_len,
                generation_time,
            )

        return (
            batch_outs[0] if len(batch_outs) == 1 else batch_outs,
            None,
            None,
        )
