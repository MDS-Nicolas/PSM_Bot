import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time
from collections import deque

# --- Configuration ---
# Directory to save the downloaded files
OUTPUT_DIR = "downloaded_site"
# Delay between requests in seconds to be polite to the server
REQUEST_DELAY = 1
# User agent to identify your bot
HEADERS = {
    'User-Agent': 'MySimpleWebCrawler/1.0 (https://example.com/botinfo)' # Optional: Replace with your info
}
# --- End Configuration ---

def is_valid_url(url):
    """Checks if a URL is valid and has an http/https scheme."""
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme) and parsed.scheme in ['http', 'https']

def get_domain(url):
    """Extracts the domain name (netloc) from a URL."""
    return urlparse(url).netloc

def download_and_save(url, base_domain, output_dir):
    """Downloads any file from the base domain and saves it, returning HTML content for parsing."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Check if the final URL after potential redirects is still on the same domain
        final_url_domain = get_domain(response.url)
        if final_url_domain != base_domain:
            print(f"Skipping redirect outside domain: {url} -> {response.url}")
            return None # Don't process links from pages outside the domain

        # Determine save path based on the final URL's path
        parsed_url = urlparse(response.url)
        path = parsed_url.path.lstrip('/')

        # If the path points to a directory, assume an index file
        # Use index.html as a standard default
        if not path or path.endswith('/'):
            path = os.path.join(path, "index.html")
        # Handle cases where the root itself is requested (e.g., http://example.com)
        elif '/' not in path and not os.path.splitext(path)[1]:
             # Check content type, if HTML, save as index.html in a folder named like the path
             content_type = response.headers.get('content-type', '').lower()
             if 'text/html' in content_type:
                 # Example: /somepage becomes /somepage/index.html
                 path = os.path.join(path, "index.html")
             # else: keep the path as is (e.g., 'robots.txt')


        save_path = os.path.join(output_dir, path)
        save_dir = os.path.dirname(save_path)

        # Create directories if they don't exist
        # Check if save_dir is not empty before creating
        if save_dir:
             os.makedirs(save_dir, exist_ok=True)

        # Save the file content (binary mode handles all file types)
        # Only save if the directory path could be determined/created
        if save_dir or not path: # Allow saving to root of output_dir if path is simple file
             # Handle root case where path might be just 'index.html'
             if not save_dir and path:
                 save_path = os.path.join(output_dir, path) # Ensure it's in output_dir

             with open(save_path, 'wb') as f:
                 f.write(response.content)
             print(f"Saved: {url} -> {save_path}")
        else:
             print(f"Skipping save for {url}, could not determine valid save directory for path: {path}")
             # Still potentially return HTML content if applicable
             pass # Continue to content type check


        # Check content type - return text only if it's HTML for link parsing
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' in content_type:
            # Use response.text which handles encoding, or fallback to content if needed
            try:
                return response.text
            except UnicodeDecodeError:
                 print(f"Warning: Could not decode HTML as text for {url}, using raw content for parsing attempt.")
                 # Fallback, BeautifulSoup might handle bytes
                 return response.content
        else:
            # It's not HTML, no links to extract from this file
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None
    except OSError as e:
        print(f"Error saving file for {url} (Path: {path}): {e}")
        return None
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def crawl_website(start_url, output_dir):
    """Crawls a website starting from start_url and downloads ALL linked files within the domain."""
    if not is_valid_url(start_url):
        print(f"Invalid starting URL: {start_url}")
        return

    base_domain = get_domain(start_url)
    if not base_domain:
        print(f"Could not determine domain for: {start_url}")
        return

    print(f"Starting crawl on domain: {base_domain}")
    print(f"Saving files to: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    queue = deque([start_url])
    visited = {start_url}

    while queue:
        current_url = queue.popleft()
        # Check if already visited *before* processing to handle redirects properly
        # However, the current logic adds to visited *before* queueing, which is generally fine.
        # Re-check here can be redundant but safe if redirects loop back in complex ways.
        # if current_url in visited_processing: continue # Optional stricter check
        # visited_processing.add(current_url) # Track what's being processed

        print(f"Processing: {current_url}")

        # Respect delay
        time.sleep(REQUEST_DELAY)

        # Download and save the resource. Get HTML content if applicable.
        content_to_parse = download_and_save(current_url, base_domain, output_dir)

        # Only try to parse and find links if the downloaded content was HTML
        if content_to_parse:
            try:
                # Use 'html.parser' or 'lxml' if installed (often faster)
                soup = BeautifulSoup(content_to_parse, 'html.parser')
                # Find all links (a href) and potentially other resources (img src, link href, script src)
                # For now, sticking to 'a' tags as per original goal, but could be expanded
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    # Join relative URLs with the current URL to get absolute URLs
                    absolute_url = urljoin(current_url, href)

                    # Clean up the URL (remove fragment identifiers like #section)
                    parsed_absolute_url = urlparse(absolute_url)
                    absolute_url = parsed_absolute_url._replace(fragment="").geturl()

                    # Check if it's a valid URL, within the same domain, and not visited
                    if (is_valid_url(absolute_url) and
                            get_domain(absolute_url) == base_domain and
                            absolute_url not in visited):
                        visited.add(absolute_url)
                        queue.append(absolute_url)
                        # print(f"  Found link: {absolute_url}") # Uncomment for debugging links
            except Exception as e:
                # Catch errors during parsing (e.g., malformed HTML)
                print(f"Error parsing HTML from {current_url}: {e}")

    print("\nCrawl finished.")
    print(f"Visited {len(visited)} unique URLs on domain {base_domain}.")

# --- Main Execution ---
if __name__ == "__main__":
    start_url = "http://techpubs.mdsaero.com/PSM5700EN_AFILeap/index.htm"
    #start_url = input("Enter the starting URL (e.g., http://example.com): ")
    crawl_website(start_url, OUTPUT_DIR)
