from flask import Flask, render_template, request, jsonify
import yt_dlp
import os
import whisper

app = Flask(__name__)

# Create a dedicated folder to save audio
download_folder = "downloads"
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

# Create a dedicated folder to save transcripts
transcriptions_folder = "transcriptions"
if not os.path.exists(transcriptions_folder):
    os.makedirs(transcriptions_folder)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    try:
        # Get data from the request
        data = request.get_json()
        print("Received Data:", data)

        youtube_link = data.get("youtubeLink")
        user_query = data.get("userQuery")

        if not youtube_link or not user_query:
            return jsonify({"error": "Please provide both a YouTube link and a query."}), 400

        print(f"Fetching video from: {youtube_link}")

        try:
            ydl_opts = {
                'format': 'bestaudio/best',  # Get the best audio format
                
                'noplaylist': True,          # Don't download the entire playlist if URL is a playlist
                'outtmpl': os.path.join(download_folder, 'audioSource.%(ext)s'),  # Save to the "downloads" folder
                'quiet': False,
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                },
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print("Starting download process...")
                info_dict = ydl.extract_info(youtube_link, download=True)
                video_title = info_dict.get('title', 'Unknown Title')
                audio_filename = os.path.join(download_folder, f"audioSource.{info_dict['ext']}")
                print(f"Video Title: {video_title}")
                print(f"Downloaded Audio File: {audio_filename}")

            # Ensure the audio file exists before proceeding
            if not os.path.exists(audio_filename):
                return jsonify({"error": f"Audio file not found after download: {audio_filename}"}), 500

            # Load Whisper model
            try:
                model = whisper.load_model("base")  # Load the Whisper model
                print(f"Transcribing audio file: {audio_filename}")
                result = model.transcribe(audio_filename)  # Transcribe the audio file

                print(f"Transcription: {result['text']}")

                os.remove(audio_filename)  # Clean up the audio file after processing
                
                print(f"Removed audio file: {audio_filename}")

            except Exception as whisper_error:
                print(f"Error with Whisper transcription: {whisper_error}")
                return jsonify({"error": "Failed to transcribe audio using Whisper."}), 500
            
            
            # Save the transcript to a file
            try:
                transcribed_text = result['text']

                transcription_filename = os.path.join(transcriptions_folder,"transcriptSource.txt")
                with open(transcription_filename, "w", encoding="utf-8") as f:
                    f.write(transcribed_text)
                    print(f"Saved transcription to: {transcription_filename}")

            except Exception as e:
                print(f"Error saving transcription: {e}")
                return jsonify({"error": "Failed to save transcription."}), 500
                

            return jsonify({
                    "message": f"Processing video: {video_title}",
                    "query": user_query,
                    "transcription": result['text']
                }), 200

        except Exception as yt_error:
            print(f"Error with yt-dlp: {yt_error}")
            return jsonify({"error": f"Failed to fetch YouTube video: {str(yt_error)}"}), 500

    except Exception as e:
        print(f"Error in /process route: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
