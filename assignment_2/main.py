import requests
import gradio as gr

# Define the API endpoint
API_URL = "http://127.0.0.1:8000/rag"

# Function to call the FastAPI backend
def predict(user_input, history):
    # Prepare the data to send to the FastAPI API
    payload = {"query": user_input, "chat_history": history}
    
    # Make a request to the FastAPI backend
    response = requests.post(API_URL, json=payload)
    
    # Get the response JSON
    result = response.json()
    print(result)
    # Extract the answer and the updated chat history
    answer = result["answer"]
    return answer

# Launch the Gradio interface
if __name__ == "__main__":
    gr.ChatInterface(predict).launch()
