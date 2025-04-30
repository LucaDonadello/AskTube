from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from collections import Counter
import torch
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
summaries_folder = "summaries"

os.makedirs(download_folder, exist_ok=True)
os.makedirs(transcriptions_folder, exist_ok=True)
os.makedirs(preprocessed_text_folder, exist_ok=True)
os.makedirs(summaries_folder, exist_ok=True)

# Load FLAN-T5 or any generative model
gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

#summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", device=0 if torch.cuda.is_available() else -1)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1) 

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
            preprocessed_filename = "preprocessed_subtitles.txt"
            save_text(preprocessed_text, os.path.join(preprocessed_text_folder, preprocessed_filename)) # Use preprocessed_text here
            print(f"Saved preprocessed file: {os.path.join(preprocessed_text_folder, preprocessed_filename)}")

            # Generate Summary
            summary_text = summarize_text(preprocessed_text)
            summary_filename = "summary_subtitles.txt"
            save_text(summary_text, os.path.join(summaries_folder, summary_filename))
            print(f"Saved summary: {os.path.join(summaries_folder, summary_filename)}")
            
            clear_download_folder()

            return jsonify({
                "message": f"Processed video with subtitles: {video_title}",
                "query": user_query,
                "transcription": transcription_text,
                "preprocessed_text": preprocessed_text,
                "summary": summary_text
            }), 200

        # If no subtitles, use Whisper
        transcription_text = transcribe_with_whisper(audio_filename)
        save_text(transcription_text, os.path.join(transcriptions_folder, "transcriptSource.txt"))
        
        # Preprocess text (correct function call!)
        preprocessed_text = preprocess_text(transcription_text)
        
        # Save preprocessed text
        preprocessed_filename = "preprocessed_transcriptSource.txt"
        save_text(preprocessed_text, os.path.join(preprocessed_text_folder, preprocessed_filename)) # Use preprocessed_text here
        print(f"Saved preprocessed file: {os.path.join(preprocessed_text_folder, preprocessed_filename)}")

        # Generate Summary
        summary_text = summarize_text(preprocessed_text)
        summary_filename = "summary_transcriptSource.txt"
        save_text(summary_text, os.path.join(summaries_folder, summary_filename))
        print(f"Saved summary: {os.path.join(summaries_folder, summary_filename)}")

        clear_download_folder()

        return jsonify({
            "message": f"Processed video with Whisper: {video_title}",
            "query": user_query,
            "transcription": transcription_text,
            "preprocessed_text": preprocessed_text,
            "summary": summary_text
        }), 200

    except Exception as e:
        print(f"Error in /process route: {e}")
        return jsonify({"error": "Internal Server Error"}), 500
    
# @app.route("/answer", methods=["POST"])
# def answer():
#     try:
#         data = request.get_json()
#         question = data.get("question")
#         context = data.get("context")

#         if not question or not context:
#             return jsonify({"error": "Missing question or context"}), 400

#         answer = generate_answer_with_window(question, context)
#         return jsonify({"answer": answer}), 200

#     except Exception as e:
#         print(f"Error in /answer route: {e}")
#         return jsonify({"error": "Internal Server Error"}), 500

# FOR TESTING PURPOSES ONLY 
# ---------------------------

@app.route("/answer", methods=["GET"])
def answer():
    question = request.args.get("question")
    context = request.args.get("context")

    if not question or not context:
        return jsonify({"error": "Missing question or context"}), 400

    answer = generate_answer_with_window(question, context)
    return jsonify({"answer": answer}), 200

# TEST: http://127.0.0.1:5000/answer?question=Who%20developed%20the%20theory%20of%20relativity%3F&context=The%20theory%20of%20relativity%2C%20developed%20by%20Albert%20Einstein%20in%20the%20early%2020th%20century%2C%20revolutionized%20our%20understanding%20of%20space%2C%20time%2C%20and%20gravity.%20It%20consists%20of%20two%20parts%3A%20special%20relativity%20and%20general%20relativity.%20Special%20relativity%20deals%20with%20the%20physics%20of%20objects%20moving%20at%20constant%20speeds%2C%20particularly%20those%20approaching%20the%20speed%20of%20light.%20General%20relativity%20extends%20this%20to%20include%20gravity%20and%20acceleration%2C%20proposing%20that%20massive%20objects%20cause%20a%20curvature%20in%20spacetime.%20These%20ideas%20have%20been%20confirmed%20by%20numerous%20experiments%20and%20have%20led%20to%20technologies%20like%20GPS%2C%20which%20rely%20on%20relativistic%20corrections%20to%20maintain%20accuracy.


# -------------------------------


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

# -------------------------------
#       NLP FUNCTIONS
# -------------------------------

def generate_answer_with_window(question, context, max_chunk_words=150, max_answer_tokens=64):
    # Break context into word chunks
    words = context.split()
    chunks = [
        " ".join(words[i:i + max_chunk_words])
        for i in range(0, len(words), max_chunk_words)
    ]
    
    # Generate answers per chunk
    answers = []
    for chunk in chunks:
        prompt = f"Question: {question}\nContext: {chunk}"
        inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = gen_model.generate(
            inputs["input_ids"],
            max_new_tokens=max_answer_tokens,
            num_beams=4,
            early_stopping=True
        )
        answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        answers.append(answer)

    return Counter(answers).most_common(1)[0][0]

def summarize_text(text_to_summarize, max_length=250, min_length=50):
    """
    Generates a summary for the given text using the loaded summarization pipeline.
    Handles length issues by chunking the text into fixed token sizes.
    """
    tokenizer = summarizer.tokenizer

    max_chunk_token_length = 800 # Size of chunks in tokens

    # Tokenize the entire input text
    print("Tokenizing the entire text...")
    input_ids = tokenizer.encode(text_to_summarize, return_tensors="pt")[0] # Get token IDs
    total_tokens = len(input_ids)
    print(f"Total tokens in input: {total_tokens}")

    chunks = []
    # Create chunks based on token count
    for i in range(0, total_tokens, max_chunk_token_length):
        chunk_ids = input_ids[i:i + max_chunk_token_length]
        # Decode the token IDs back to text for the summarizer
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        if chunk_text and not chunk_text.isspace(): # Avoid empty chunks
            chunks.append(chunk_text)

    if not chunks:
        return "Input text was too short or empty to summarize."

    print(f"Summarizing text split into {len(chunks)} chunk(s) of approx {max_chunk_token_length} tokens each...")

    summaries = []
    for i, chunk in enumerate(chunks):
         try:
             chunk_summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False, truncation=True)[0]['summary_text']
             summaries.append(chunk_summary)
             print(f"Summary for chunk {i+1}/{len(chunks)} generated.")
         except Exception as e:
             print(f"Error summarizing chunk {i+1}: {e}")
             print(f"Problematic chunk (first 500 chars): {chunk[:500]}...")
             summaries.append("[Error summarizing this chunk]")

    # Concatenate the summaries from all chunks
    final_summary = " ".join(summaries)
    return final_summary

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    gen_model.to(device)

    app.run(debug=True)