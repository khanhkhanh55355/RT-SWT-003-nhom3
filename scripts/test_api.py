#!/usr/bin/env python3
"""Gate E3: verify OpenAI API key and one real completion call."""
import os
import sys

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    print("FAILED: openai package not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-..."):
        print("FAILED: OPENAI_API_KEY not set.")
        print("Create a .env file in the repo root:")
        print("  OPENAI_API_KEY=sk-your-key-here")
        sys.exit(1)

    model = os.getenv("OPENAI_MODEL", "gpt-4o-2024-11-20")
    client = OpenAI(api_key=api_key)

    print(f"Testing model: {model}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a QA expert."},
                {
                    "role": "user",
                    "content": (
                        "Generate a one-line BDD scenario for: "
                        "As a User, I want to log in to the system."
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=128,
            timeout=30,
        )
    except Exception as exc:
        print(f"FAILED: API call error: {exc}")
        sys.exit(1)

    text = resp.choices[0].message.content.strip()
    usage = resp.usage
    print("OK: API responded successfully")
    print(f"  response_id: {resp.id}")
    print(f"  tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out")
    print(f"  preview: {text[:120]}...")


if __name__ == "__main__":
    main()
