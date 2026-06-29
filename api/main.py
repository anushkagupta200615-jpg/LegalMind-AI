import os
import sys
import uvicorn
import base64
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.orchestrator import run_legalmind

app = FastAPI(title='LegalMind AI API', version='1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    query: str
    file_base64: Optional[str] = None

class DraftRequest(BaseModel):
    draft_type: str
    context: dict

@app.on_event("startup")
async def startup_event():
    print("LegalMind AI API started")

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0"}

@app.post("/analyze")
def analyze_endpoint(req: AnalyzeRequest):
    file_path = None
    try:
        if req.file_base64:
            pdf_bytes = base64.b64decode(req.file_base64)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(pdf_bytes)
                file_path = tmp.name
                
        result = run_legalmind(req.query, file_path)
        
        return {"result": result, "agent_steps": []}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)

@app.post("/draft")
def draft_endpoint(req: DraftRequest):
    try:
        query = f"Drafting task: {req.draft_type}. Context: {req.context}"
        result = run_legalmind(query, None)
        return {"draft": result}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
