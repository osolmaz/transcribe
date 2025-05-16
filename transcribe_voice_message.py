# To run this code you need to install the following dependencies:
# pip install google-genai

import argparse
import mimetypes
import os
from google import genai
from google.genai import types

PROJECT_ID = os.getenv("PROJECT_ID")


def generate(bytes_data: bytes, mime_type: str):
    # client = genai.Client(
    #     api_key=os.environ.get("GEMINI_API_KEY"),
    # )
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="global",
    )

    model = "gemini-2.5-pro-preview-05-06"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=bytes_data, mime_type=mime_type),
                types.Part.from_text(
                    text="Transcribe this voice message. "
                    "Output it as proper paragraphs. "
                    "Do not include filler sounds like 'um' or 'uh'."
                ),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe voice messages using Gemini API"
    )
    parser.add_argument(
        "audio_file", help="Path to the audio file (supports mp3, ogg, wav, etc.)"
    )
    args = parser.parse_args()

    try:
        # Detect MIME type from file extension
        mime_type, _ = mimetypes.guess_type(args.audio_file)
        if mime_type is None:
            # Default to audio/mpeg if detection fails
            mime_type = "audio/mpeg"

        print(f"Detected MIME type: {mime_type}")

        with open(args.audio_file, "rb") as f:
            audio_bytes = f.read()
        generate(audio_bytes, mime_type)
    except FileNotFoundError:
        print(f"Error: File '{args.audio_file}' not found")
    except Exception as e:
        print(f"Error processing file: {e}")
