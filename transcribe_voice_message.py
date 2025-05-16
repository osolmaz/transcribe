"""
Takes a voice message and transcribes it into text.

Why translate with Gemini 2.5 Pro and do the write-up with GPT-4.5?

Because IMO Gemini 2.5 Pro transcription is currently SOTA
and GPT-4.5 is the best model for doing write-ups.
"""
import argparse
import mimetypes
import os

from google import genai
from google.genai import types
from lingua import Language, LanguageDetectorBuilder
from openai import OpenAI

PROJECT_ID = os.getenv("PROJECT_ID")
GEMINI_MODEL = "gemini-2.5-pro-preview-05-06"
OPENAI_MODEL = "gpt-4.5-preview"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if OPENAI_API_KEY is None or OPENAI_API_KEY == "":
    raise ValueError("OPENAI_API_KEY is not set")

if GEMINI_API_KEY is None or GEMINI_API_KEY == "":
    # Use Vertex AI API
    gemini_client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="global",
    )
else:
    # Use Gemini API
    gemini_client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

openai_client = OpenAI(
    # This is the default and can be omitted
    api_key=OPENAI_API_KEY,
)


def generate_first_transcript(bytes_data: bytes, mime_type: str):
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=bytes_data, mime_type=mime_type),
                types.Part.from_text(
                    text="Transcribe this voice message. "
                    "Output it as proper paragraphs. "
                    "Do not include filler sounds like 'um' or 'uh'. "
                    "Output just the transcript, and no other commentary."
                ),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )
    ret = ""
    for chunk in gemini_client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")
        ret += chunk.text
    return ret


def detect_language_with_lingua(text: str):
    # Create a language detector with common languages
    languages = [
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.GERMAN,
        Language.ITALIAN,
        Language.PORTUGUESE,
        Language.RUSSIAN,
        Language.ARABIC,
        Language.CHINESE,
        Language.JAPANESE,
        Language.KOREAN,
        Language.TURKISH,
    ]

    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    # Detect the language
    detected_language = detector.detect_language_of(text)

    # If no language was detected, return None
    if detected_language is None:
        return "Unknown"

    return detected_language.name


def translate_transcript_into_english(transcript: str):
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=transcript),
                types.Part.from_text(
                    text="Translate this transcript into English. "
                    "Output just the transcript, and no other commentary."
                ),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )
    ret = ""
    for chunk in gemini_client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")
        ret += chunk.text
    return ret


def clean_transcript(transcript: str):
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=transcript
        + (
            "\n\n---\n\n"
            "- Above is a voice message transcription.\n"
            "- Restructure it better while keeping "
            "- ALL THE INFORMATION included in there.\n"
            "Start with a tl;dr bullet point summary that is not overly long.\n"
            "- The rest of the document can be any structure that makes sense, "
            "as long as it's succinct, easy to read and understand.\n"
            "- Output markdown, and start with a top-level heading.\n"
            "- Output just the transcript, and no other commentary."
        ),
    )

    print(response.output_text)
    return response.output_text


def main(audio_file: str, output_file: str | None = None):
    # If output_file is not provided, use the same name as the audio file
    if output_file is None:
        # Get the directory of the audio file
        directory = os.path.dirname(audio_file)
        # Get the base filename without extension
        base_name = os.path.basename(audio_file)
        base_without_ext = os.path.splitext(base_name)[0]
        # Create output file path in the same directory as source file
        output_file = os.path.join(directory, base_without_ext + ".md")

    try:
        # Detect MIME type from file extension
        mime_type, _ = mimetypes.guess_type(audio_file)
        if mime_type is None:
            # Default to audio/mpeg if detection fails
            mime_type = "audio/mpeg"

        print(f"Detected MIME type: {mime_type}")

        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
    except FileNotFoundError:
        print(f"Error: File '{audio_file}' not found")
        return
    except Exception as e:
        print(f"Error processing file: {e}")
        return

    print("Generating transcript...")
    first_transcript = generate_first_transcript(audio_bytes, mime_type)

    print("\nDetecting language using lingua-py...")
    detected_language = detect_language_with_lingua(first_transcript)
    print(f"Detected language: {detected_language}")

    translated_transcript = None

    if detected_language != "ENGLISH":
        print("\nTranslating transcript into English...")
        translated_transcript = translate_transcript_into_english(first_transcript)
        print("\nFinal transcript:")
        print(translated_transcript)
    else:
        print("\nTranscript is already in English. No translation needed.")

    print("\nCleaning transcript...")
    cleaned_transcript = clean_transcript(translated_transcript)
    print("\nFinal transcript:")
    print(cleaned_transcript)

    with open(output_file, "w") as f:
        f.write(cleaned_transcript)

        if translated_transcript is not None:
            f.write("\n\n---\n\n## Translated Transcript\n\n")
            f.write(translated_transcript)
            f.write(f"\n\n---\n\n## Original transcript in {detected_language}\n\n")
            f.write(first_transcript)
        else:
            f.write("\n\n---\n\n## Original transcript\n\n")
            f.write(first_transcript)

    print(f"\nTranscript saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe voice messages using Gemini API"
    )
    parser.add_argument(
        "audio_file",
        help="Path to the audio file (supports mp3, ogg, wav, etc.)",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        help="Path to the output file (supports md, txt, etc.)",
        default=None,
    )
    args = parser.parse_args()

    main(args.audio_file, output_file=args.output_file)
