import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai

app = FastAPI(
    title="Prashansa Guru API",
    description="Backend service for Prashansa Guru (ප්‍රශංසා ගුරු) - AI Sri Lankan Astrologer",
    version="1.0.0"
)

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# System prompt explicitly establishing the brand identity "Prashansa Guru"
system_prompt = """You are "Prashansa Guru" (ප්‍රශංසා ගුරු), an expert Sri Lankan traditional Astrologer (ජ්‍යෝතිෂවේදී) proficient in Vedic Astrology, Nakath (නැකත්), Rahu Kalaya (රාහු කාලය), Porondam Matching (පොරොන්දම් ගැළපීම), and Rashiphalapala (රාශි පලාපල).

Rules:
1. Always introduce yourself politely as "ප්‍රශංසා ගුරු" (Prashansa Guru) when greeting users.
2. Respond politely in natural Sinhala using authentic astrological terminology (ලග්නය, නැකත, ග්‍රහ පිහිටීම, දෝෂ).
3. Maintain a respectful, wise, and helpful tone. If essential birth details (date, time, or place) are missing for deep readings, ask for them politely.
4. Always include a brief cultural note that predictions depend on planetary positions and encourage righteous living (ධර්මානුකූලව ජීවත් වීම)."""

# Structure for incoming requests from the Mobile App
class ChatRequest(BaseModel):
    user_prompt: str
    birth_date: str | None = None
    birth_time: str | None = None
    birth_place: str | None = None

@app.get("/")
def home():
    return {"status": "Prashansa Guru API is running online!"}

@app.post("/astrology-reading")
async def get_astrology_reading(request: ChatRequest):
    try:
        # Build prompt with optional birth details
        context = request.user_prompt
        if request.birth_date or request.birth_time or request.birth_place:
            context += f"\n\n[ජන්ම තොරතුරු: දිනය: {request.birth_date}, වේලාව: {request.birth_time}, ස්ථානය: {request.birth_place}]"

        interaction = client.interactions.create(
            model='models/gemini-3.8-flash',
            input=context,
            system_instruction=system_prompt,
        )
        return {"response": interaction.output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))