import requests
import json

# --- Local API Configuration (Copied from rag_original_gemma.py) ---
# Make sure this URL points to your running local model endpoint
# API_URL = "http://10.130.236.73:1234/v1/chat/completions" # Old endpoint
API_URL = "http://10.130.236.56:1235/v1/chat/completions" # New endpoint

def call_local_api(messages, max_tokens=2, temperature=0.2):
    """
    Sends a request to the local LLM API.
    """
    headers = {"Content-Type": "application/json"}
    data = {
        # Ensure this model name is compatible with the new endpoint
        "model": "llava-v1.5-7b:3", # You might need to change this model name too
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # You might need to adjust parameters based on your specific model
    }
    try:
        print(f"Sending request to {API_URL} with data: {json.dumps(data)}")
        response = requests.post(API_URL, json=data, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        # Check if the response is valid JSON
        try:
            response_json = response.json()
        except json.JSONDecodeError:
            print(f"Error: Response is not valid JSON. Response text: {response.text}")
            return "Error: Invalid JSON response from API."

        # Check the expected structure
        if "choices" in response_json and len(response_json["choices"]) > 0 and \
           "message" in response_json["choices"][0] and \
           "content" in response_json["choices"][0]["message"]:
            content = response_json["choices"][0]["message"]["content"]
            return content
        else:
            print(f"Error: Unexpected response structure. Full response: {response_json}")
            return "Error: Unexpected response structure from API."

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return f"Error: API request failed - {e}"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"Error: An unexpected error occurred - {e}"

# --- Main execution ---
if __name__ == "__main__":
    print("Asking the local model for a number 1-10...")

    # Construct the message payload
    joke_messages = [
        {"role": "system", "content": "You are a helpful assistant that generates numbers."},
        {"role": "user", "content": "Return a number between 1 and 10."}
    ]

    # Call the API
    joke_response = call_local_api(joke_messages)

    # Print the result
    print("\n--- Model Response ---")
    print(joke_response)
    print("----------------------") 