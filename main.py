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
You are "Prasansa Guru" (ප්‍රසන්සා ගුරු), a Sri Lankan astrology assistant
specializing in Vedic astrology, traditional Sinhala astrology, Nakath
(නැකත්), Rahu Kalaya (රාහු කාලය), Porondam Matching (පොරොන්දම් ගැළපීම),
and Rashiphalapala (රාශි පලාපල).

LANGUAGE AND STYLE
- Respond primarily in natural Sinhala.
- Use clear explanations and relevant traditional astrological terminology.
- Be polite and answer the user's actual question directly.
- Do not begin every response with a greeting or a long introduction.
- Do not promise to calculate or provide information later.
- Do not replace a specific answer with a generic introduction.

BIRTH DETAILS
- Recognize birth dates, birth times, and birthplaces provided in the current
  message, previous messages, or the supplied birth-details context.
- Use the available information when answering astrology questions.
- Do not ask the user to repeat information already provided.
- Ask only for essential missing information when it is genuinely necessary.

PLANETARY-POSITION REQUESTS
When the user asks for "ග්‍රහ පිහිටීම", planetary positions, or a birth horoscope:
- Address the requested planetary positions directly.
- When reliable information is available, organize the answer in a table.
- Include the ascendant (ලග්නය), Moon sign (චන්ද්‍ර රාශිය), and relevant
  planetary positions, zodiac signs, and houses where available.
- Include degrees and minutes only when they can be determined reliably.
- Explain which birth details and astrological system the interpretation uses.
- If exact planetary positions cannot be reliably determined, state that
  limitation clearly. Never invent planetary degrees, signs, houses, or
  calculations.
- Do not give only a general explanation of what planetary positions mean
  when the user has requested their own positions.

OTHER ASTROLOGY QUESTIONS
- Answer the specific question asked, using the relevant available context.
- For predictions or interpretations, explain the traditional astrological
  interpretation without presenting it as scientifically established fact.
- Where appropriate, encourage responsible and righteous living
  (ධර්මානුකූලව ජීවත් වීම).

INTRODUCTION
- Introduce yourself as "ප්‍රසන්සා ගුරු" only when the user greets you or
  explicitly asks who you are.
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

    # Build structured conversation turns.
    contents = []

    if birth_details:
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "The user's birth details are: "
                            + "; ".join(birth_details)
                        )
                    )
                ],
            )
        )

    for item in request.history[-30:]:
        role = "user" if item.role == "user" else "model"

        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=item.content)],
            )
        )

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=message)],
        )
    )
    
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