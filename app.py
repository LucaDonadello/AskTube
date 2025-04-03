from flask import Flask, render_template, request, jsonify
import yt_dlp
import os
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Create a dedicated folder to save audio if it doesn't exist
download_folder = "downloads"
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    try:
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
                'outtmpl': os.path.join(download_folder, '%(title)s.%(ext)s'),  # Save to the "downloads" folder
                'quiet': False,
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                },
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(youtube_link, download=True)
                video_title = info_dict.get('title', 'Unknown Title')
                audio_filename = os.path.join(download_folder, f"{video_title}.mp3")

            print(f"Video Title: {video_title}")
        except Exception as yt_error:
            print(f"Error with yt-dlp: {yt_error}")
            return jsonify({"error": f"Failed to fetch YouTube video: {str(yt_error)}"}), 500


        return jsonify({"message": f"Processing video: {video_title}", "query": user_query, "audio_filename": audio_filename})

    except Exception as e:
        print(f"Error in /process route: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
