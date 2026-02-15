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

            # ---- Normalize answer ----
            if isinstance(parsed["answer"], str):
                parsed["answer"] = parsed["answer"].strip().upper()[0]

            if parsed["answer"] not in ["A", "B", "C", "D"]:
                return None

            # ---- Clean choices ----
            if not isinstance(parsed["choices"], list) or len(parsed["choices"]) != 4:
                return None

            cleaned_choices = []
            for choice in parsed["choices"]:
                choice = choice.strip()
                choice = re.sub(r'^([A-D]\))\s*\1', r'\1', choice)
                cleaned_choices.append(choice)

            # Remove duplicate choices
            if len(set(cleaned_choices)) != 4:
                return None

            parsed["choices"] = cleaned_choices

            # Ensure correct option exists
            if not any(c.startswith(parsed["answer"] + ")") for c in cleaned_choices):
                return None

            # Explanation checks
            explanation = parsed["explanation"]
            if not explanation or len(explanation) < 20:
                return None

            # contradiction_words = ["however", "but", "although", "actually", "though"]
            # if any(word in explanation.lower() for word in contradiction_words):
            #     return None
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
                "Generate internally but DO NOT output reasoning steps. "
                "Output STRICTLY valid JSON only. "
                "Do not add commentary. "
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
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            temperature=kwargs.get("temperature", 0.3),
            top_p=kwargs.get("top_p", 0.9),
            repetition_penalty=kwargs.get("repetition_penalty", 1.2),
            do_sample=kwargs.get("do_sample", True),
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

        return (
            batch_outs if len(batch_outs) > 1 else batch_outs[0] if batch_outs else [],
            token_len,
            generation_time,
        )
