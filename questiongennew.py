import os
import json
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

PROMPT_TEMPLATE = open("prompt_template.txt").read()  # the prompt you finalized

def build_prompt(act_name, jurisdiction, act_text, num_adv, num_unc, num_app):
    return (
        PROMPT_TEMPLATE
        .replace("{{ACT_NAME}}", act_name)
        .replace("{{JURISDICTION}}", jurisdiction)
        .replace("{{ACT_TEXT}}", act_text)
        .replace("{{NUM_ADVERSARIAL}}", str(num_adv))
        .replace("{{NUM_UNCERTAINTY}}", str(num_unc))
        .replace("{{NUM_APPLIED}}", str(num_app))
    )

def generate_questions(act_name, jurisdiction, act_text, num_adv=2, num_unc=2, num_app=2):
    prompt = build_prompt(act_name, jurisdiction, act_text, num_adv, num_unc, num_app)

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,              # low temp for consistency; not 0, some variety in scenarios is fine
            response_mime_type="application/json",  # forces valid JSON output
            # no `tools=` param at all -> no web search, no grounding
        ),
    )

    usage = response.usage_metadata
    print(
        f"Tokens: {usage.prompt_token_count} prompt + "
        f"{usage.candidates_token_count} output "
        f"{usage.total_token_count} total"
    )

    raw_text = response.text
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        print(raw_text)
        raise

    return data


if __name__ == "__main__":
    with open("EDA/statutes_FED.json") as f:
        acts = json.load(f)

    # smoke test on one substantive act first (skip trivial title-only ones
    # like "Canadian Environment Week Act" at n_chars_en=356)
    act = next(a for a in acts if a["n_chars_en"] > 700000)

    result = generate_questions(
        act_name=act["name_en"],
        jurisdiction="Canada",
        act_text=act["text_en"],
    )

    with open("output.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Act: {act['name_en']} ({act['n_chars_en']} chars)")
    if "skip_reason" in result:
        print(f"Skipped: {result['skip_reason']}")
    else:
        questions = result.get("questions", [])
        print(f"Generated {len(questions)} questions")
        summary = result.get("generation_summary", {})
        for key, value in summary.items():
            print(f"  {key}: {value}")