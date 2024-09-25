from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Assuming you have already implemented the RAG pipeline in a function called run_rag_pipeline
# Let's assume it's in a module named `rag_pipeline`

from ncert_chatbot import ask_question
# Initialize FastAPI app
app = FastAPI()

# Create a request model to define the input for the RAG pipeline
class RAGRequest(BaseModel):
    query: str
    chat_history: list

# Create a response model to define the output of the RAG pipeline
class RAGResponse(BaseModel):
    answer: str
    chat_history: str

# Define a FastAPI endpoint for RAG pipeline
@app.post("/rag", response_model=RAGResponse)
async def get_rag_answer(request: RAGRequest):
    try:
        # Call the RAG pipeline function with the query
        result = ask_question(request.query, request.chat_history)
        
        # Assuming your RAG pipeline returns a dictionary with 'answer' and 'chat_history'
        return RAGResponse(
            answer=result["answer"],
            chat_history=result["chat_history"]
        )
    except Exception as e:
        # In case of an error, return an HTTPException with a 500 status code
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application (for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
