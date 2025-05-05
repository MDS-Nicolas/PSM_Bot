import os
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration (Copy from ragnew.py or adjust) ---
load_dotenv() # Load .env file if your endpoint URL is stored there
GEMMA_ENDPOINT = os.getenv("GEMMA_ENDPOINT", "http://10.130.236.56:1235/v1") # Default if not in .env
GEMMA_MODEL_NAME = "gemma-3-27b-it"
# --- End Configuration ---

print(f"Attempting to connect to LLM endpoint: {GEMMA_ENDPOINT}")
print(f"Using model: {GEMMA_MODEL_NAME}")

try:
    # Initialize the OpenAI client pointed at the local endpoint
    client = OpenAI(
        api_key="EMPTY",  # Required by SDK, but often ignored by local servers
        base_url=GEMMA_ENDPOINT,
    )

    # Define the prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ]

    print("\nSending prompt to LLM...")

    # Send the request to the chat completions endpoint
    response = client.chat.completions.create(
        model=GEMMA_MODEL_NAME,
        messages=messages,
        max_tokens=150,  # Limit response length
        temperature=0.7,
        stream=False      # Keep it simple, no streaming for this test
    )

    # Extract and print the response content
    if response.choices:
        joke = response.choices[0].message.content.strip()
        print("\nLLM Response:")
        print(joke)
        print("\n--- Test Successful ---")
    else:
        print("\n--- Test Failed: No response choices received. ---")
        print("Full Response:", response)

except Exception as e:
    print(f"\n--- Test Failed: An error occurred ---")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Details: {e}")
    print("\nPlease check:")
    print("1. Is the vLLM server running?")
    print(f"2. Is the endpoint URL '{GEMMA_ENDPOINT}' correct?")
    print(f"3. Is the model '{GEMMA_MODEL_NAME}' loaded on the server?")
    print("4. Are there any network connectivity issues (firewalls, etc.)?") 