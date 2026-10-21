import os
import logging
from typing import Literal


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types, errors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prasansa-guru")

APP_TITLE = "Prasansa Guru API"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(
    title=APP_TITLE,
    description="Backend service for Prasansa Guru (ප්‍රසන්සා ගුරු)",
    version="1.1.0",
)

# The mobile app does not rely on browser CORS, but keeping CORS configured
# makes local API testing and a possible web client easier.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """
You are "Prasansa Guru" (ප්‍රසන්සා ගුරු), an AI assistant that responds in the
style of a Sri Lankan traditional astrologer. You are familiar with Vedic
astrology, Nakath (නැකත්), Rahu Kalaya (රාහු කාලය), Porondam Matching
(පොරොන්දම් ගැළපීම), and Rashiphalapala (රාශි පලාපල).

Respond politely, primarily in natural Sinhala, and use relevant traditional
astrological terms where appropriate. Answer the user's actual question.
If birth date, birth time, or birth place is important but missing, ask for
the missing details instead of pretending that you have them. Do not claim
to have calculated a horoscope from details that were not supplied.
When greeting a user, introduce yourself as "ප්‍රසන්සා ගුරු" (Prasansa Guru).
Where appropriate, briefly note that traditional predictions are interpretive
and encourage responsible, righteous living (ධර්මානුකූලව ජීවත් වීම).
Do not add unrelated disclaimers to every answer.
""".strip()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class AstrologyRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=30)
    birth_date: str | None = Field(default=None, max_length=100)
    birth_time: str | None = Field(default=None, max_length=100)
    birth_place: str | None = Field(default=None, max_length=200)


@app.get("/")
def home():
    return {"status": "Prasansa Guru API is running online!"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "api_key_configured": bool(API_KEY)}

@app.get("/diagnostics")
def diagnostics():
    from importlib.metadata import version

    return {
        "google_genai_version": version("google-genai"),
        "model": MODEL_NAME,
    }

@app.post("/astrology-reading")
async def get_astrology_reading(request: AstrologyRequest):
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="The server has not been configured with a Gemini API key.",
        )

    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Please enter a question.")

    birth_details = []
    if request.birth_date:
        birth_details.append(f"උපන් දිනය: {request.birth_date}")
    if request.birth_time:
        birth_details.append(f"උපන් වේලාව: {request.birth_time}")
    if request.birth_place:
        birth_details.append(f"උපන් ස්ථානය: {request.birth_place}")

    # Build conversation context in order so follow-up questions make sense.
    conversation_parts = []
    if birth_details:
        conversation_parts.append(
            "User-provided birth details (use only as context): "
            + "; ".join(birth_details)
        )

    for item in request.history[-30:]:
        speaker = "User" if item.role == "user" else "Prasansa Guru"
        conversation_parts.append(f"{speaker}: {item.content}")

    conversation_parts.append(f"User: {message}")
    contents = "\n\n".join(conversation_parts)
    
    try:
        client = genai.Client(api_key=API_KEY)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
            ),
        )

        answer = response.text

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="The AI service returned an empty response. Please try again.",
            )

        return {"response": answer}

    except HTTPException:
        raise

    except errors.APIError as exc:
        status_code = getattr(exc, "code", None)

        if status_code == 429:
            logger.warning("Gemini quota or rate limit reached: %s", exc)
            raise HTTPException(
                status_code=429,
                detail=(
                    "The AI service has reached its current usage limit. "
                    "Please try again later."
                ),
            ) from exc

        if status_code == 503:
            logger.warning("Gemini service temporarily unavailable: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=(
                    "The AI service is temporarily busy. "
                    "Please try again shortly."
                ),
            ) from exc

        logger.exception("Gemini API request failed")
        raise HTTPException(
            status_code=502,
            detail="The AI service encountered an error. Please try again later.",
        ) from exc

    except Exception:
        logger.exception("Unexpected error while requesting Gemini")
        raise HTTPException(
            status_code=502,
            detail="The AI service could not answer this request. Please try again later.",
        )