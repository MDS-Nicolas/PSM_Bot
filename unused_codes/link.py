import requests
import os
# from dotenv import load_dotenv # No longer needed
from requests_ntlm import HttpNtlmAuth
import argparse # Import the argparse library

# # Load environment variables from .env file # No longer needed
# # load_dotenv() # No longer needed

# Set up argument parser
parser = argparse.ArgumentParser(description="Connect to SharePoint URL using NTLM auth.")
parser.add_argument("--url", required=True, help="The SharePoint URL to connect to.")
parser.add_argument("--username", required=True, help="The SharePoint username (e.g., DOMAIN\\user).")
parser.add_argument("--password", required=True, help="The SharePoint password.")

# Parse arguments from command line
args = parser.parse_args()

# Get credentials and URL from arguments
url = args.url
username = args.username
password = args.password

# # Ensure credentials are loaded # Argparse handles required arguments
# if not all([url, username, password]):
#     print("Error: Ensure SHAREPOINT_URL, SHAREPOINT_USERNAME, and SHAREPOINT_PASSWORD are set in the .env file.")
# else: # No longer need the else block because argparse exits if required args are missing

# Wrap the main logic in a try block (moved indentation out one level)
try:
    # Make the request using NTLM authentication
    print(f"Attempting to connect to: {url}") # Optional: Add some feedback
    response = requests.get(url, auth=HttpNtlmAuth(username, password))

    # Raise an exception for bad status codes (4xx or 5xx)
    response.raise_for_status()

    # Print the HTML content of the page
    print("\nConnection successful! Page content:\n") # Optional: Add success message
    print(response.text)

except requests.exceptions.RequestException as e:
    print(f"An error occurred during the request: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")