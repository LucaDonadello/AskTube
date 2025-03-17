import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "your_default_secret_key")  # Security key for Flask sessions
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")  # YouTube Data API key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI key for Whisper/GPT
    GOOGLE_SPEECH_API_KEY = os.getenv("GOOGLE_SPEECH_API_KEY")  # Google Speech-to-Text API
    DEBUG = True  # Set False in production
