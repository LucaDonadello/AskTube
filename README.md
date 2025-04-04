# AskTube Installation & Setup Guide

## Overview
AskTube is a smart tool that enables users to ask questions about YouTube videos and receive instant, accurate answers by analyzing audio, subtitles, and context.


## Prerequisites
- Ensure FFmpeg is installed on your system. If not, download and install it from the [official FFmpeg website](https://ffmpeg.org).


## Steps to Install & Run AskTube

### 1. Install FFmpeg
Follow the instructions for your operating system to install FFmpeg:
- **Linux**: Use `sudo apt install ffmpeg` (Debian/Ubuntu) or equivalent for your distribution.
- **Windows**: Download FFmpeg from the official website, extract it, and add its path to your system environment variables.
- **macOS**: Use Homebrew with `brew install ffmpeg`.

### 2. Clone the Repository
Run the following commands in your terminal:
- git clone https://github.com/yourusername/AskTube.git
- cd AskTube

### 3. Install Dependencies
Install the required Python packages:
- pip install -r requirements.txt

### 4. Configure Script Execution (Windows) -- OPTIONAL
Open PowerShell and allow script execution:
- Set-ExecutionPolicy Unrestricted -Scope Process

### 5. Activate Virtual Environment -- OPTIONAL
Activate the virtual environment:
- AskTube\Scripts\Activate.ps1

### 6. Run the Application
Start AskTube by running:
- python app.py

## Features

- **Audio & Subtitle Analysis**: Extracts and analyzes audio and subtitles for precise responses.
- **Whisper Integration**: Utilizes Whisper for accurate speech-to-text transcription.
- **Context-Aware Answers**: Provides meaningful responses by considering video context.
