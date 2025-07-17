from fastapi import FastAPI, Query
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_huggingface import HuggingFacePipeline

app = FastAPI()

model_name = "torchtorchkimtorch/Llama-3.2-Korean-GGACHI-1B-Instruct-v1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=200,
    temperature=0.7
)

# LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

@app.get("/g")
async def generate_response(prompt: str = Query(..., description="생성할 프롬프트 입력")):
    result = llm(prompt)
    return {"answer": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
