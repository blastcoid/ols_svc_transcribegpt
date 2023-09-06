import openai, io, tiktoken
from fastapi import FastAPI, HTTPException, File
from pydantic_settings import BaseSettings
from fastapi.middleware.cors import CORSMiddleware

# from google.cloud import texttospeech


class Settings(BaseSettings):
    # OpenAI API Settings
    openai_api_key: str = "secret"
    chatcompletion_model: str = "gpt-3.5-turbo"
    whisper_model: str = "whisper-1"
    chatcompletion_temperature: float = 0.5
    max_response_tokens: int = 1000
    token_limit: int = 4096
    context_depth: int = 4096

    # Application Settings
    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000

    # Loading .env file if present
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class NamedBytesIO(io.BytesIO):
    def __init__(self, buffer, name=None):
        super().__init__(buffer)
        self.name = name


settings = Settings()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API key setup
openai.api_key = settings.openai_api_key

def countToken(str):
    encoding = tiktoken.encoding_for_model(settings.chatcompletion_model)
    encoded_text = encoding.encode(str)
    return len(encoded_text)


messages = [
    {
        "role": "system",
        "content": """
            Nama kamu adalah Stella. Kamu adalah ahli dokter kesehatan dan nutrisi. Kamu akan menerima pertanyaan menggunakan bahasa gaul pertanyaan dari pasien.
        """,
    }
]

token_data = []


async def chatGptResponse(message):
    try:
        messages.append({"role": "user", "content": message})
        token_data.append(countToken(message))
        response = openai.ChatCompletion.create(
            model=settings.chatcompletion_model,
            messages=messages,
            temperature=settings.chatcompletion_temperature,
            max_tokens=settings.max_response_tokens,
        )
        messages.append(
            {
                "role": "assistant",
                "content": response["choices"][0]["message"]["content"],
            },
        )
        total_tokens = response["usage"]["total_tokens"]
        # while 16000 - total_tokens < 2000:
        #     messages.pop(1)
        #     removed_token = token_data.pop(0)
        #     total_tokens -= removed_token
        while total_tokens > settings.context_depth:
            messages.pop(1)
            removed_token = token_data.pop(0)
            total_tokens -= removed_token
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return response


async def checkModeration(message):
    try:
        moderation_response = openai.Moderation.create(input=message)
    except:
        raise HTTPException(status_code=500, detail=str(e))

    return moderation_response["results"][0]["flagged"]


@app.post("/transcribe")
async def transcribe(audio_file: bytes = File(...)):
    try:
        # Get audio transcription
        audio_buffer = NamedBytesIO(audio_file, name="audio.wav")
        # transcribe audio to text
        transcript = openai.Audio.transcribe(settings.whisper_model, audio_buffer)
        user_message = transcript["text"]
        # Check moderation
        flagged = await checkModeration(user_message)
        if not flagged:
            # Get chatgpt completion
            completion = await chatGptResponse(user_message)
            content = {
                "message": messages,
                "prompt": transcript["text"],
                "completion": completion["choices"][0]["message"]["content"],
                "total_tokens": completion["usage"]["total_tokens"],
            }
        else:
            content = {
                "completion": "Maaf, pertanyaan atau statement kamu melanggar Moderation Policy kami",
            }
        return content
    except Exception as e:
        print(e)
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reset")
async def reset_context():
    try:
        global messages, token_data
        messages = [
            {
                "role": "system",
                "content": """
                    Nama kamu adalah Stella. Kamu adalah ahli dokter kesehatan dan nutrisi. Kamu akan menerima pertanyaan atau tanggapan dari pasien.
                    kamu harus menjawab dengan singkat, padat, jelas menggunakan bahasa gaul anak jaksel.

                """,
            }
        ]
        token_data = []
        return {"status": "success", "message": "Context has been reset."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Create health check endpoint
@app.get("/")
async def health_check():
    return {"status": "success", "message": "OK"}