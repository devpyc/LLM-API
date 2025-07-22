# LangChain-Test

```bash
docker build -t my-fastapi-mcp-gpu .
```

```bash
docker run -d --name langfastapi --gpus all -p 8000:8000 my-fastapi-mcp-gpu
```