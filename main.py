import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = FastAPI(title="Korean Llama Instruct API")

# Retrieve your Hugging Face token from environment variable
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

def get_generator():
    # Load model and tokenizer only once
    model_name = "torchtorchkimtorch/Llama-3.2-Korean-GGACHI-1B-Instruct-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=HF_TOKEN
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# Instantiate the generator at startup
generator = get_generator()

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 150
    temperature: float = 0.7
    do_sample: bool = True

class GenerateResponse(BaseModel):
    generated_text: str

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Korean Llama Instruct API is running."}

@app.post("/generate", response_model=GenerateResponse, tags=["Generate"])
async def generate_text(request: GenerateRequest):
    try:
        output = generator(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=request.do_sample
        )
        return GenerateResponse(generated_text=output[0]["generated_text"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate", response_model=GenerateResponse, tags=["Generate"])
async def generate_text_get(
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    do_sample: bool = True
):
    try:
        output = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample
        )
        return GenerateResponse(generated_text=output[0]["generated_text"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
