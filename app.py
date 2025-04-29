from flask import Flask, render_template, request, jsonify
import yt_dlp
import os
import whisper
import requests
import json
import re

app = Flask(__name__)

download_folder = "downloads"
transcriptions_folder = "transcriptions"
preprocessed_text_folder = "preprocessing"

os.makedirs(download_folder, exist_ok=True)
os.makedirs(transcriptions_folder, exist_ok=True)
os.makedirs(preprocessed_text_folder, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    try:
        data = request.get_json()
        youtube_link = data.get("youtubeLink")
        user_query = data.get("userQuery")

        if not youtube_link or not user_query:
            return jsonify({"error": "Please provide both a YouTube link and a query."}), 400

        info_dict, audio_filename = download_youtube_audio(youtube_link)
        video_title = info_dict.get('title', 'Unknown Title')

        transcription_text = extract_subtitles(info_dict)

        if transcription_text:
            save_text(transcription_text, os.path.join(transcriptions_folder, "subtitles.txt"))
            
            # Preprocess text
            preprocessed_text = preprocess_text(transcription_text)
            
            # Save preprocessed text
            save_preprocessed_text(transcription_text, "preprocessed_subtitles.txt")
            
            clear_download_folder()

            return jsonify({
                "message": f"Processed video with subtitles: {video_title}",
                "query": user_query,
                "transcription": transcription_text,
                "preprocessed_text": preprocessed_text
            }), 200

        # If no subtitles, use Whisper
        transcription_text = transcribe_with_whisper(audio_filename)
        save_text(transcription_text, os.path.join(transcriptions_folder, "transcriptSource.txt"))
        
        # Preprocess text (correct function call!)
        preprocessed_text = preprocess_text(transcription_text)
        
        save_preprocessed_text(transcription_text, "preprocessed_transcriptSource.txt")

        clear_download_folder()

        return jsonify({
            "message": f"Processed video with Whisper: {video_title}",
            "query": user_query,
            "transcription": transcription_text,
            "preprocessed_text": preprocessed_text
        }), 200

    except Exception as e:
        print(f"Error in /process route: {e}")
        return jsonify({"error": "Internal Server Error"}), 500



# -------------------------------
#       HELPER FUNCTIONS
# -------------------------------

def download_youtube_audio(youtube_link):
    ydl_opts = {
        'format': 'bestaudio/best',
        'writesubtitles': True,
        'writeautomaticsub': True,
        'noplaylist': True,
        'outtmpl': os.path.join(download_folder, 'audioSource.%(ext)s'),
        'quiet': False,
        'headers': {'User-Agent': 'Mozilla/5.0'},
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_link, download=True)
        audio_filename = os.path.join(download_folder, f"audioSource.{info_dict['ext']}")
        return info_dict, audio_filename

def extract_plaintext_subtitles(json_text):
    try:
        data = json.loads(json_text)
        events = data.get("events", [])
        lines = []
        for event in events:
            segs = event.get("segs")
            if segs:
                line = "".join(seg.get("utf8", "") for seg in segs)
                lines.append(line.strip())
        return "\n".join(lines).strip()
    except Exception as e:
        print(f"Error parsing subtitle JSON: {e}")
        return None
    
def fetch_subtitle_text(subs_dict):
    if 'en' in subs_dict:
        sub_formats = subs_dict['en']
        if isinstance(sub_formats, list) and sub_formats:
            subtitle_url = sub_formats[0].get('url')
            if subtitle_url:
                response = requests.get(subtitle_url)
                if response.ok:
                    return response.text
    return None

def extract_subtitles(info_dict):   
    subtitles = info_dict.get("subtitles", {})
    automatic_captions = info_dict.get("automatic_captions", {})

    raw_subtitles = fetch_subtitle_text(subtitles) or fetch_subtitle_text(automatic_captions)
    if raw_subtitles:
        return extract_plaintext_subtitles(raw_subtitles) or raw_subtitles
    return None


def transcribe_with_whisper(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    model = whisper.load_model("base")
    print(f"Transcribing with Whisper: {audio_path}")
    result = model.transcribe(audio_path)
    os.remove(audio_path)
    return result['text']


def save_text(text, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved file: {filepath}")


# preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

#function to save preprocessed text
def save_preprocessed_text(original_text, filename):
    preprocessed = preprocess_text(original_text)
    output_path = os.path.join(preprocessed_text_folder, filename)
    save_text(preprocessed, output_path)
    print(f"Saved preprocessed file: {output_path}")




def clear_download_folder():
    for filename in os.listdir(download_folder):
        file_path = os.path.join(download_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    app.run(debug=True)