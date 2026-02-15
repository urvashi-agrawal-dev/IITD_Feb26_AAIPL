import time
import json
import re
from typing import Optional, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class QAgent(object):

    def __init__(self, **kwargs):
        model_name = "Qwen/Qwen3-4B"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
        )

        self.seen_questions = set()

    # ---------------- STRICT JSON CLEANER ---------------- #
    def clean_json(self, text: str):

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            return None

        json_str = text[start:end + 1]

        try:
            parsed = json.loads(json_str)

            required_keys = ["topic", "question", "choices", "answer", "explanation"]
            for key in required_keys:
                if key not in parsed:
                    return None

            # Normalize answer
            parsed["answer"] = parsed["answer"].strip().upper()[0]

            if parsed["answer"] not in ["A", "B", "C", "D"]:
                return None

            # Remove duplicates globally
            if parsed["question"] in self.seen_questions:
                return None
            self.seen_questions.add(parsed["question"])

            # Validate choices
            if not isinstance(parsed["choices"], list) or len(parsed["choices"]) != 4:
                return None

            cleaned_choices = []
            for choice in parsed["choices"]:
                choice = choice.strip()
                choice = re.sub(r'^([A-D]\))\s*\1', r'\1', choice)
                cleaned_choices.append(choice)

            if len(set(cleaned_choices)) != 4:
                return None

            if not any(c.startswith(parsed["answer"] + ")") for c in cleaned_choices):
                return None

            parsed["choices"] = cleaned_choices

            # Explanation validation
            explanation = parsed["explanation"]
            if not explanation or len(explanation) < 20:
                return None

            # Token limit enforcement (150)
            question_tokens = len(self.tokenizer.encode(parsed["question"], add_special_tokens=False))
            choice_tokens = sum(len(self.tokenizer.encode(c, add_special_tokens=False)) for c in parsed["choices"])
            answer_tokens = len(self.tokenizer.encode(parsed["answer"], add_special_tokens=False))

            if question_tokens + choice_tokens + answer_tokens > 150:
                return None

            return parsed

        except Exception:
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
                "You are an expert exam setter. "
                "Output STRICTLY valid JSON only. "
                "Do not output reasoning steps. "
                "If logical inconsistency appears, silently regenerate."
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

        start_time = time.time()

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=300,
            temperature=0.1,
            top_p=0.8,
            repetition_penalty=1.2,
            do_sample=False,
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

            parsed = self.clean_json(content)

            if parsed is not None:
                batch_outs.append(parsed)

        return (
            batch_outs if len(batch_outs) > 1 else batch_outs[0] if batch_outs else [],
            token_len,
            generation_time,
        )
