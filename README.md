# AskTube Installation & Setup Guide

## Overview

AskTube is an intelligent tool that enables users to ask questions about YouTube videos and receive instant, context-aware answers. It analyzes the video's audio, subtitles, and contextual information to provide accurate responses. Built with Flask, AskTube integrates with Whisper for speech-to-text transcription and Hugging Face Transformers (such as RoBERTa or DistilBERT) for question answering.

---

## Prerequisites

- **FFmpeg:** Required for handling audio and video processing.
- **Python:** Ensure Python 3.7+ is installed.
- **Hugging Face Transformers:** For implementing the QA system with models like RoBERTa or DistilBERT.

---

## Steps to Install & Run AskTube

### 1. Install FFmpeg

FFmpeg is required for audio extraction from YouTube videos. Follow the instructions below based on your operating system:

- **Linux:**  

sudo apt install ffmpeg

- **Windows:**  
Download FFmpeg from the [official website](https://ffmpeg.org/download.html), extract the files, and add its `bin` directory to your system environment variables.

- **macOS:**  

brew install ffmpeg

---

### 2. Clone the Repository

Clone the AskTube repository to your local machine:

git clone https://github.com/yourusername/AskTube.git
cd AskTube

---

### 3. Install Dependencies

Install the required Python libraries:

pip install -r requirements.txt

---

### 4. Configure Script Execution (Windows) (Optional)

If you're on Windows and encounter an issue with script execution, you can temporarily allow it by running the following in PowerShell:

Set-ExecutionPolicy Unrestricted -Scope Process

---

### 5. Activate Virtual Environment (Optional)

If you're using a virtual environment, activate it using the following command:

- **Windows:**

AskTube\Scripts\Activate.ps1

- **macOS/Linux:**

source AskTube/bin/activate

---

### 6. Run the Application

Start the AskTube application by running:

python app.py

---

## Features

- **Audio & Subtitle Analysis:**  
  Extracts and processes audio and subtitles to understand video content.

- **Whisper Integration:**  
  Utilizes Whisper for high-quality speech-to-text transcription of video audio, ensuring accurate conversion of spoken content.

- **Context-Aware Question Answering:**  
  Implements Hugging Face Transformers (RoBERTa/DistilBERT) to provide intelligent, context-aware answers based on the video’s content.

- **YouTube Video Processing:**  
  Allows users to upload a YouTube URL, from which the audio and subtitles are extracted for analysis.

---

**Happy querying with AskTube!**

