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
                    cleaned_choices.append(cleaned.strip())
                parsed["choices"] = cleaned_choices
            valid_options = ["A", "B", "C", "D"]
            if parsed["answer"] not in valid_options:
                return None
            answer_letter = parsed["answer"]
            if not any(choice.startswith(f"{answer_letter})") for choice in parsed["choices"]):
                return None
            explanation = parsed["explanation"]
            if not explanation or len(explanation) < 15:
                return None
            if answer_letter not in explanation:
                return None
            contradiction_words = ["however", "but", "although", "actually", "though"]
            if any(word in explanation.lower() for word in contradiction_words):
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
                "Generate internally but DO NOT output reasoning steps."
                "Output only valid JSON."
                "If any logical contradiction appears, regenerate internally before responding."

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
            max_new_tokens= 220,
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            repetition_penalty=kwargs.get("repetition_penalty", 1.1),
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
