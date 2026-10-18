import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai

app = FastAPI(title="Sri Lankan Astrology API")

# Initialize Gemini Client
# Reads GEMINI_API_KEY from environment variables
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

system_prompt = """You are a traditional Sri Lankan Astrologer (ජ්‍යෝතිෂවේදී) proficient in Vedic Astrology, Nakath (නැකත්), Rahu Kalaya (රාහු කාලය), Porondam Matching (පොරොන්දම් ගැළපීම), and Rashiphalapala (රාශි පලාපල).

Rules:
1. Respond politely in natural Sinhala using authentic astrological terminology (ලග්නය, නැකත, ග්‍රහ පිහිටීම, දෝෂ).
2. Maintain a respectful tone. If essential details (birth date, time, or place) are missing for deep readings, ask for them politely.
3. Always include a brief cultural note that predictions depend on planetary positions and encourage righteous living (ධර්මානුකූලව ජීවත් වීම)."""

# Structure for incoming requests from the Mobile App
class ChatRequest(BaseModel):
    user_prompt: str
    birth_date: str | None = None
    birth_time: str | None = None
    birth_place: str | None = None

@app.get("/")
def home():
    return {"status": "Sri Lankan Astrology API is running online!"}

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