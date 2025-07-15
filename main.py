import os
import base64
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import urllib.parse

app = FastAPI(title="Korean Llama Instruct API")

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

def get_generator():
    model_name = "torchtorchkimtorch/Llama-3.2-Korean-GGACHI-1B-Instruct-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=HF_TOKEN
    )
    return pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id
    )

generator = get_generator()

def decode_base64(encoded_text):
    """base64 인코딩된 텍스트를 디코딩합니다."""
    try:
        decoded_bytes = base64.b64decode(encoded_text)
        decoded_text = decoded_bytes.decode('utf-8')
        return decoded_text
    except Exception as e:
        print(f"Base64 decode error: {e}")
        return None

def encode_base64(text):
    """텍스트를 base64로 인코딩합니다."""
    try:
        encoded_bytes = text.encode('utf-8')
        encoded_text = base64.b64encode(encoded_bytes).decode('utf-8')
        return encoded_text
    except Exception as e:
        print(f"Base64 encode error: {e}")
        return None

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
            do_sample=request.do_sample,
            return_full_text=False
        )
        return GenerateResponse(generated_text=output[0]["generated_text"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/g", response_model=GenerateResponse, tags=["Generate"])
async def generate_g(
    prompt: str = Query(..., description="텍스트 생성을 위한 프롬프트"),
    max_new_tokens: int = Query(150, description="생성할 최대 토큰 수"),
    temperature: float = Query(0.7, description="생성 온도 (0.0-1.0)"),
    do_sample: bool = Query(True, description="샘플링 사용 여부"),
    base64: bool = Query(False, description="프롬프트가 base64 인코딩되었는지 여부")
):
    try:
        # base64 디코딩
        if base64:
            decoded_prompt = decode_base64(prompt)
            if decoded_prompt is None:
                raise HTTPException(status_code=400, detail="Invalid base64 encoding")
            final_prompt = decoded_prompt
        else:
            final_prompt = urllib.parse.unquote(prompt)
        
        print(f"Final prompt: {final_prompt}")
        
        output = generator(
            final_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            return_full_text=False
        )
        
        generated_text = output[0]["generated_text"]
        print(f"Generated: {generated_text}")
        
        return GenerateResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/b64", response_model=GenerateResponse, tags=["Generate"])
async def generate_base64(
    q: str = Query(...),
    max_new_tokens: int = Query(150),
    temperature: float = Query(0.7),
    do_sample: bool = Query(True)
):
    try:
        # base64
        decoded_prompt = decode_base64(q)
        if decoded_prompt is None:
            raise HTTPException(status_code=400, detail="Invalid base64 encoding")
        
        print(f"Base64 decoded: {decoded_prompt}")
        
        output = generator(
            decoded_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            return_full_text=False
        )
        
        generated_text = output[0]["generated_text"]
        print(f"Generated: {generated_text}")
        
        return GenerateResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/encode", tags=["Utility"])
async def encode_text(text: str = Query(..., description="인코딩할 텍스트")):
    try:
        decoded_text = urllib.parse.unquote(text)
        encoded = encode_base64(decoded_text)
        return {
            "original_text": decoded_text,
            "base64_encoded": encoded,
            "usage_example": f"http://localhost:8000/b64?q={encoded}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/decode", tags=["Utility"])
async def decode_text(encoded_text: str = Query(..., description="decoding")):
    try:
        decoded = decode_base64(encoded_text)
        if decoded is None:
            raise HTTPException(status_code=400, detail="Invalid base64 encoding")
        return {
            "base64_encoded": encoded_text,
            "decoded_text": decoded
        }
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