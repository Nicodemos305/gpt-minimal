from transformers import pipeline
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

generator = pipeline('text-generation', model='gpt2')

class TextGenerationRequest(BaseModel):
    prompt: str

@app.post("/generate/")
def generate_text(request: TextGenerationRequest):
    result = generator(request.prompt, max_length=50, num_return_sequences=1)
    return {"generated_text": result[0]['generated_text']}
