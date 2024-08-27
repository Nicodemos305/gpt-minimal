# GPT Minimal API

This project is a study of the GPT-2 model, created to explore building a minimal API for text generation using Python, FastAPI, and Docker. It serves as a simple starting point for those who want to better understand how to use language models in a containerized environment.

## Overview

The API allows you to send a text prompt and receive a continuation generated by the GPT-2 model. It’s a lightweight implementation, easy to set up, and ideal for experimentation and learning about using GPT-2.

## Requirements

- Docker
- Python 3.10+

## Project Structure

```plaintext
gpt_minimal/
│
├── app/
│   ├── main.py            # Main API code
│   └── requirements.txt   # Project dependencies
│
└── Dockerfile             # Dockerfile for building the image
```

## How to Run

### 1. Clone the Repository

Clone the repository to your local environment:

```bash
git clone https://github.com/your-username/gpt_minimal.git
cd gpt_minimal
```

### 2. Build the Docker Image

docker build -t gpt-minimal .

### 3. Run the Container
docker run -d -p 8000:8000 gpt-minimal

### 4. GPT Minimal is Ready!

curl -X POST "http://localhost:8000/generate/" -H "Content-Type: application/json" -d '{"prompt": "Once upon a time"}'
