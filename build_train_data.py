import json

# Load your clean questions
with open("outputs/questions.json", "r") as f:
    questions = json.load(f)

with open("data/train.jsonl", "w") as out:
    for q in questions:
        user_prompt = q["question"] + "\n\nChoices:\n" + "\n".join(q["choices"])
        
        assistant_response = json.dumps({
            "answer": q["answer"],
            "reasoning": q["explanation"]
        })

        example = {
            "messages": [
                {"role": "system", "content": "You are an expert reasoning assistant."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
        }

        out.write(json.dumps(example) + "\n")

print("Training file created at data/train.jsonl")

