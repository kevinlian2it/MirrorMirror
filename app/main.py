from fastapi import FastAPI
from app.routers import classify, model_config, embeddings

app = FastAPI()

# Include routers
app.include_router(classify.router, prefix="/classify", tags=["Classification"])
app.include_router(model_config.router, prefix="/model", tags=["Model Configuration"])
app.include_router(embeddings.router, prefix="/embeddings", tags=["Embeddings"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Article Classification API"}
