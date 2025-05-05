# app.py
import os
import uuid
import webbrowser
import uvicorn
import zipfile
import asyncio # Import asyncio for Queue
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request # Removed WebSocket
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse # Added StreamingResponse
import subprocess
import json# app.py
import os
import uuid
import webbrowser
import uvicorn
import zipfile
import asyncio # Import asyncio for Queue
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request # Removed WebSocket
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse # Added StreamingResponse
import subprocess
import json
from datetime import datetime
from typing import Dict, AsyncGenerator, Callable, Optional # Add Callable, Optional
from functools import partial
import shutil # Add this import for cleanup
 
# Import the backend functions for PDF to DOCX processing.
# Ensure app_testingfunction.py defines:
# def load_api_key()
# async def process_pdf(client, pdf_path, output_docx)
#from app_testingFunction import load_api_key, process_pdf
from matthew_best_final import load_api_key, process_pdf
 
app = FastAPI()
 
# --- Global state variables ---
processed_files = {}
log_queues: Dict[str, asyncio.Queue] = {}
job_start_times: Dict[str, datetime] = {}
cancelled_jobs: set = set()
# --- End Global state variables ---
 
# HTML content updated for persistent download, credits, and reset button
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Proposal PDF to DOCX Table Processor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"> <!-- Added viewport meta tag -->
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Changed font to Segoe UI */
      margin: 0; /* Removed default body margin */
      padding: 1rem 0.5rem; /* Reduced vertical padding */
      display: flex;
      flex-direction: column;
      min-height: 100vh; /* Changed to 100vh */
      background-color: #f4f7f9; /* Slightly different light grey */
      box-sizing: border-box; /* Include padding in height calculation */
      color: #333; /* Default text color */
    }
    .content-wrapper {
        flex-grow: 1; /* Allow main content to grow */
        max-width: 960px; /* Increased max-width slightly */
        width: 100%; /* Ensure it takes available width */
        margin: 0 auto; /* Center the wrapper */
        display: flex; /* Enable flexbox for side-by-side layout */
        gap: 1.5rem; /* Reduced gap */
        align-items: flex-start; /* Align panels to the top */
    }
    .instructions-panel, .main-panel {
        background-color: #ffffff; /* White background for panels */
        padding: 1.25rem; /* Reduced padding inside panels */
        border-radius: 8px; /* Rounded corners */
        box-shadow: 0 2px 10px rgba(0,0,0,0.05); /* Subtle shadow */
        display: flex; /* Enable flex for panel children */
        flex-direction: column; /* Stack children vertically */
    }
    .instructions-panel {
        flex: 0 0 280px; /* Slightly wider fixed width for the left panel */
        /* padding-top: 0.5rem; Removed, using uniform padding */
    }
    .instructions-panel h4 {
        margin-top: 0; /* Removed top margin */
        margin-bottom: 1rem; /* Increased bottom margin */
        font-weight: 600; /* Bolder heading */
        color: #1a1a1a;
    }
    .instructions-panel ol {
        padding-left: 1.5rem; /* Indent list */
        margin-bottom: 1.5rem; /* Increased margin */
    }
    .instructions-panel li {
        margin-bottom: 0.6rem; /* Increased spacing */
        line-height: 1.4;
    }
    .estimates {
        margin-top: 1.5rem; /* Add some space above the estimates section */
        font-size: 0.9em; /* Base font size for estimates */
        color: #388e3c; /* Slightly darker green */
    }
    .estimates-heading { /* Style like the h4 */
        display: block; /* Make it block level */
        font-weight: 600; /* Bolder heading */
        margin-bottom: 0.5rem; /* Space below heading */
        color: #1a1a1a; /* Match h4 color */
        font-size: 1.1em; /* Slightly larger than list items */
    }
    .estimates-details { /* Style like list items */
        padding-left: 0; /* No extra indent needed */
        line-height: 1.5; /* Spacing between lines */
    }
    .main-panel {
        flex-grow: 1; /* Allow main panel to take remaining space */
        /* display: flex; Already set above */
        /* flex-direction: column; Already set above */
    }
    h1 {
      text-align: center;
      margin-bottom: 1.5rem; /* Reduced margin below h1 */
      margin-top: 0; /* Remove default top margin */
      font-size: 1.8em; /* Slightly larger */
      font-weight: 600;
      color: #1a1a1a;
    }
    .drop-zone {
      border: 2px solid #e0e0e0; /* Lighter, solid border */
      border-radius: 8px; /* Match panel radius */
      padding: 2rem; /* Increased padding */
      text-align: center;
      color: #888; /* Lighter text color */
      transition: background-color 0.2s, border-color 0.2s, box-shadow 0.2s; /* Added box-shadow to transition */
      cursor: pointer;
      margin-top: 0.5rem; /* Reduced margin */
      /* box-shadow: 0 2px 5px rgba(0,0,0,0.05); Removed default shadow */
      background-color: #fdfdfd; /* Very light background */
      flex-shrink: 0; /* Prevent shrinking */
    }
    .drop-zone.dragover {
      background-color: #eef6ff; /* Light blue background on dragover */
      border-color: #64b5f6; /* Blue border */
      border-style: dashed; /* Dashed border on dragover */
      color: #333;
    }
    #file-input {
      display: none;
    }
    .button-container {
         text-align: center;
         margin-top: 1.25rem; /* Reduced margin */
         flex-shrink: 0; /* Prevent shrinking */
    }
    #process-button, #reset-button {
        display: inline-block;
        margin: 0 8px; /* Increased margin between buttons */
        padding: 0.8rem 1.8rem; /* Slightly larger padding */
        font-size: 1rem;
        font-weight: 500; /* Slightly bolder text */
        vertical-align: top;
        border: none;
        border-radius: 6px; /* Slightly more rounded */
        cursor: pointer;
        transition: background-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out; /* Smooth transition */
    }
    #process-button {
        background-color: #64b5f6; /* Brighter blue */
        color: white; /* White text */
    }
    #process-button:hover {
        background-color: #42a5f5; /* Darker blue on hover */
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    #reset-button {
      background-color: #ef5350; /* Slightly softer red */
      color: white;
    }
     #reset-button:hover {
        background-color: #e53935; /* Darker red on hover */
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    #output {
      text-align: center;
      margin-top: 0.75rem; /* Reduced margin-top from 1.5rem */
      font-size: 1.1rem;
      min-height: 40px; /* Reduced min-height */
      flex-shrink: 0; /* Prevent shrinking */
    }
    .download-btn {
      display: inline-block;
      padding: 15px 35px; /* Increased padding for height and width */
      background-color: #4CAF50;
      color: white;
      text-decoration: none;
      border-radius: 6px; /* Match buttons */
      margin-top: 8px; /* Keep top margin */
      margin-bottom: 0; /* Remove bottom margin */
      font-size: 1rem; /* Match buttons */
      font-weight: 500; /* Match buttons */
      transition: background-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
      flex-shrink: 0; /* Prevent shrinking */
    }
    .download-btn:hover {
        background-color: #43a047; /* Darker green on hover */
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    /* Style for the Processing Log heading */
    h2.log-heading {
        font-size: 1.1em; /* Adjusted font size to match other headings */
        font-weight: 600; /* Bolder */
        margin-top: 1.5rem; /* Reduced margin-top */
        margin-bottom: 0.75rem; /* Increased margin-bottom */
        flex-shrink: 0; /* Prevent shrinking */
        color: #1a1a1a;
    }
    #log-console {
      border: 1px solid #dcdcdc; /* Lighter border */
      background-color: #eeeeee; /* Light grey background */
      padding: 12px; /* Increased padding */
      font-family: 'Consolas', 'Monaco', monospace; /* Better monospace fonts */
      font-size: 0.9em; /* Slightly larger log font */
      white-space: pre-wrap;
      text-align: left;
      flex-grow: 1; /* Allow log to take available vertical space */
      min-height: 80px; /* Minimum height */
      max-height: 80px; /* Set max-height to match min-height */
      overflow-y: scroll; /* Changed from 'auto' to 'scroll' */
      border-radius: 6px; /* Rounded corners */
      margin-bottom: auto; /* Push footer down within the panel */
    }
    .log-entry {
      margin-bottom: 6px; /* Increased margin */
      line-height: 1.4;
    }
    /* Hide elements using a class */
    .hidden {
        display: none;
    }
    footer {
        text-align: center;
        margin-top: 1.5rem; /* Reduced margin */
        padding-top: 0.75rem; /* Reduced padding */
        border-top: 1px solid #e0e0e0; /* Lighter border */
        font-size: 0.9em;
        color: #888; /* Lighter footer text */
        width: 100%; /* Take width of parent panel */
        flex-shrink: 0; /* Prevent footer from shrinking */
        box-sizing: border-box; /* Include padding/border in width */
    }

    /* Responsive Design: Stack panels on smaller screens */
    @media (max-width: 768px) {
        body {
            padding: 1rem 0.5rem; /* Adjust padding for small screens */
        }
        .content-wrapper {
            flex-direction: column; /* Stack panels vertically */
            gap: 1.5rem; /* Adjust gap for vertical stacking */
            align-items: stretch; /* Stretch items to full width */
        }
        .instructions-panel {
            flex: 1 1 auto; /* Allow panel to grow/shrink */
            width: auto; /* Let flexbox handle width */
            /* text-align: center; Removed centering */
        }
        .instructions-panel ol {
             /* display: inline-block; Removed */
             /* text-align: left; Kept */
             padding-left: 1.5rem; /* Restore padding */
             /* margin-left: auto; Removed */
             /* margin-right: auto; Removed */
        }
        .main-panel {
            width: auto; /* Let flexbox handle width */
        }
        h1 {
            font-size: 1.6em; /* Adjust heading size */
            margin-bottom: 1.5rem;
        }
        .drop-zone {
            padding: 1.5rem; /* Adjust padding */
        }
        #process-button, #reset-button {
            padding: 0.7rem 1.5rem; /* Adjust button padding */
        }
        footer {
            margin-top: 1.5rem;
            padding-top: 0.75rem;
        }
        #log-console {
             max-height: 80px; /* Set max-height to match min-height on small screens */
        }
    }

  </style>
</head>
<body>
  <h1>AI RFQ PDF to Compliance Matrix</h1>

  <div class="content-wrapper">
    <!-- Left Panel for Instructions -->
    <div class="instructions-panel">
        <h4>Instructions:</h4> <!-- Removed underline -->
        <ol>
            <li>Drag and Drop ONE PDF into the drop box</li>
            <li>Click Process PDF</li>
            <li>Download the ZIP file</li>
        </ol>
        <!-- Updated Estimates Structure -->
        <div class="estimates">
            <span class="estimates-heading">Estimated time:</span>
            <div class="estimates-details">
                10 pages -&gt; ~10 minutes<br>
                50 pages -&gt; ~1 hour
            </div>
        </div>
        <!-- End Updated Estimates Structure -->
        <br>
        <!-- Added Controlled Goods Note -->
        <div style="color: red; font-weight: bold; margin-top: 1rem; margin-bottom: 1rem;">
            *** NOTE: Do not input Controlled Goods Documents at this time ***
        </div>
        <!-- End Added Note -->
    </div>

    <!-- Right Panel for Main Content -->
    <div class="main-panel">
      <!-- Removed Instruction Text from here -->

      <div id="drop-zone" class="drop-zone">
        Drag & Drop your PDF here or click to select.
      </div>
      <input type="file" id="file-input" accept="application/pdf">
            <!-- Button Container -->
            <div class="button-container">
                <button id="process-button">Process PDF</button> <!-- Initially visible -->
                <button id="reset-button">Reset</button> <!-- New Reset Button -->
            </div>
      <div id="output"></div>

            <!-- Removed file-status div, as it wasn't styled or used visibly -->

            <!-- Log Console -->
            <h2 class="log-heading">Processing Log:</h2> <!-- Added class -->
            <div id="log-console"></div>

            <!-- Footer moved inside main-panel -->
            <footer>
                Credits: Developed by the AI Team
            </footer>
    </div>
  </div>

  <script>
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const processButton = document.getElementById('process-button');
    const resetButton = document.getElementById('reset-button');
    const outputDiv = document.getElementById('output');
    const logConsole = document.getElementById('log-console');
   
    let selectedFile = null;
    let processingJobId = null;
    let checkStatusInterval = null;
    let eventSource = null;
    const LOCAL_STORAGE_KEY = 'pdfProcessorJobInfo';
 
    // --- Helper functions to show/hide elements (Keep current version) ---
    function showElement(element) {
        if (element) {
            element.style.display = 'inline-block'; // Use inline-block as defined in CSS
        }
    }
    function hideElement(element) {
        if (element) {
            element.style.display = 'none'; // Directly set display to none
        }
    }
    // --- End Helper functions ---
 
    // --- Helper function to format seconds ---
    function formatElapsedTime(totalSeconds) {
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;

        // Pad with leading zeros if necessary
        const paddedHours = String(hours).padStart(2, '0');
        const paddedMinutes = String(minutes).padStart(2, '0');
        const paddedSeconds = String(seconds).padStart(2, '0');

        return `${paddedHours}:${paddedMinutes}:${paddedSeconds}`;
    }
    // --- End Helper function ---
 
    // --- UI State Management Functions ---
    function showProcessingState(elapsedTime = 0) {
        hideElement(processButton);
        const formattedTime = formatElapsedTime(elapsedTime); // Format the time
        outputDiv.textContent = `Processing file... Elapsed time: ${formattedTime}`; // Use formatted time
    }
 
    // --- Modify showIdleState to be absolutely explicit ---
    function showIdleState() {
        // Explicitly set display style for processButton
        if (processButton) {
            processButton.style.display = 'inline-block';
        }
        // Original logic: showElement(processButton); // This is now redundant but fine
 
        // Clear other elements
        outputDiv.innerHTML = '';
        logConsole.innerHTML = '';
        dropZone.textContent = 'Drag & Drop your PDF here or click to select.';
        fileInput.value = '';
        selectedFile = null;
    }
    // --- End Modify showIdleState ---
 
    function showCompletedState(jobId) {
         displayDownloadButton(jobId); // Process button stays hidden until reset
    }
 
    function showErrorState(message) {
        outputDiv.innerHTML = `<div style="color: red">Error: ${message}</div>`; // Process button stays hidden until reset
    }
    // --- End UI State Management ---
 
    function addLogMessage(message) {
      const entry = document.createElement('div');
      entry.className = 'log-entry';
      if (!message.startsWith('[')) {
         message = `[${new Date().toLocaleTimeString()}] ${message}`;
      }
      entry.textContent = message;
      logConsole.appendChild(entry);
      logConsole.scrollTop = logConsole.scrollHeight;
    }
 
    function connectEventSource(jobId) {
       if (eventSource) { eventSource.close(); }
       const url = `/api/stream-logs/${jobId}`;
       addLogMessage(`Connecting to log stream: ${url}`);
       eventSource = new EventSource(url);
       eventSource.onmessage = function(event) { addLogMessage(`Server: ${event.data}`); };
       eventSource.onerror = function(event) {
         if (processingJobId) { addLogMessage("Log stream error or closed unexpectedly."); console.error("EventSource Error:", event); }
         if (eventSource) eventSource.close(); eventSource = null;
       };
    }
 
    function stopMonitoring() {
        if (checkStatusInterval) { clearInterval(checkStatusInterval); checkStatusInterval = null; console.log("Stopped status polling."); }
        if (eventSource) {
            if(processingJobId) { addLogMessage("Closing log stream connection."); }
            eventSource.close(); eventSource = null;
        }
    }
 
    function displayDownloadButton(jobId) {
        outputDiv.innerHTML = ''; // Clear previous content

        // Create the download link
        const downloadLink = document.createElement('a');
        downloadLink.href = `/api/download/${jobId}`;
        downloadLink.className = 'download-btn';
        downloadLink.textContent = 'Download Results (.zip)';
        // downloadLink.style.verticalAlign = 'middle'; // No longer needed

        // Append only the download link to the outputDiv
        outputDiv.appendChild(downloadLink);
    }
 
    // --- Event listeners for file selection (with logging) ---
    console.log("Attaching file selection listeners...");
    dropZone.addEventListener('click', () => {
        console.log("Drop zone clicked.");
        try { fileInput.click(); console.log("fileInput.click() called successfully."); }
        catch (e) { console.error("Error calling fileInput.click():", e); }
    });
    fileInput.addEventListener('change', (event) => {
        console.log("File input 'change' event fired.");
        if (event.target.files && event.target.files.length > 0) {
        selectedFile = event.target.files[0];
        dropZone.textContent = selectedFile.name;
            outputDiv.innerHTML = '';
            console.log("File selected via dialog:", selectedFile.name);
        } else { console.log("File input 'change' event fired, but no files selected."); }
    });
    dropZone.addEventListener('dragover', (event) => { event.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', (event) => { event.preventDefault(); dropZone.classList.remove('dragover'); });
    dropZone.addEventListener('drop', (event) => {
        console.log("Drop event fired.");
      event.preventDefault();
      dropZone.classList.remove('dragover');
        if (event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files.length > 0) {
        selectedFile = event.dataTransfer.files[0];
        dropZone.textContent = selectedFile.name;
            outputDiv.innerHTML = '';
            console.log("File dropped:", selectedFile.name);
            fileInput.files = event.dataTransfer.files;
        } else { console.log("Drop event fired, but no files found in dataTransfer."); }
    });
    console.log("File selection listeners attached.");
    // --- End File Selection Listeners ---
   
    async function checkProcessingStatus() {
      if (!processingJobId) { stopMonitoring(); return; } // Stop if job ID lost
     
      try {
        const response = await fetch(`/api/status/${processingJobId}`);
        if (response.ok) {
          const data = await response.json();
          console.log("Status check:", data);
 
          if (data.status === "cancelled") {
             addLogMessage("Job was cancelled on the server.");
             stopMonitoring();
             localStorage.removeItem(LOCAL_STORAGE_KEY); // Clear storage
             showErrorState("Task was cancelled.");
             processingJobId = null;
             return;
          }
         
          if (data.status === "completed") {
            stopMonitoring();
            // Store completed state with ID
            localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify({ id: processingJobId, status: 'completed' }));
            showCompletedState(processingJobId);
          } else if (data.status === "processing") {
            // Store processing state with ID
            localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify({ id: processingJobId, status: 'processing' }));
            showProcessingState(data.elapsed_time);
          } else if (data.status === "error") {
            stopMonitoring();
            localStorage.removeItem(LOCAL_STORAGE_KEY); // Clear storage
            showErrorState(data.message);
            processingJobId = null; // Clear job ID after error displayed
          }
        } else {
           const errorText = await response.text();
           console.error("Status check failed:", response.status, errorText);
           if (response.status === 404 || response.status === 500) {
               addLogMessage("Job status check failed or job not found.");
               stopMonitoring();
               localStorage.removeItem(LOCAL_STORAGE_KEY);
               showErrorState("Previous job status lost or unavailable. Please upload again.");
               processingJobId = null;
           } // Handle other non-OK statuses if needed
        }
      } catch (error) {
        console.error("Error checking status:", error);
        stopMonitoring();
        localStorage.removeItem(LOCAL_STORAGE_KEY);
        showErrorState(`Network error checking status: ${error.message}`);
        processingJobId = null;
      }
    }
   
    // --- processButton listener (Updated) ---
    processButton.addEventListener('click', async () => {
      if (!selectedFile) { alert('Please select a PDF file first.'); return; }
 
      // --- Move showProcessingState() to the top ---
      showProcessingState(); // Hide process button IMMEDIATELY
      // --- End move ---
 
      stopMonitoring(); // Stop any previous monitoring
      localStorage.removeItem(LOCAL_STORAGE_KEY); // Clear previous job info
      processingJobId = null;
      logConsole.innerHTML = ''; // Clear logs for new job
 
      const formData = new FormData();
      formData.append('file', selectedFile);
     
      // showProcessingState(); // Removed from here
      addLogMessage("Uploading file and initiating process...");
 
      try {
        const response = await fetch('/api/process-background', { method: 'POST', body: formData });
       
        if (response.ok) {
          const data = await response.json();
          processingJobId = data.job_id;
          // Store initial processing state
          localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify({ id: processingJobId, status: 'processing' }));
          addLogMessage(`Processing started with Job ID: ${processingJobId}`);
          connectEventSource(processingJobId);
          // Start polling after a short delay
          if (!checkStatusInterval) {
              setTimeout(() => { if (processingJobId) { checkStatusInterval = setInterval(checkProcessingStatus, 2000); } }, 500);
          }
        } else {
          const errorText = await response.text();
          stopMonitoring();
          localStorage.removeItem(LOCAL_STORAGE_KEY);
          showErrorState(errorText); // Show error and Process button
          addLogMessage(`Error starting process: ${errorText}`);
        }
      } catch (error) {
        console.error("Error:", error);
        stopMonitoring();
        localStorage.removeItem(LOCAL_STORAGE_KEY);
        showErrorState(error.message); // Show error and Process button
        addLogMessage(`Client-side error: ${error.message}`);
      }
    });
    // --- End processButton listener ---
 
    // --- Reset Button Event Listener ---
    resetButton.addEventListener('click', () => {
        addLogMessage("Reset button clicked - proceeding with reset."); // Updated log message
        const currentJobId = processingJobId; // Capture ID before clearing
 
        stopMonitoring(); // Stop polling and close SSE
        localStorage.removeItem(LOCAL_STORAGE_KEY); // Clear stored job state
        processingJobId = null; // Clear the current job ID variable
        showIdleState(); // Reset UI to initial state (calls the updated showIdleState)
 
        // If there was an active job, send a stop request to the backend
        if (currentJobId) {
            addLogMessage(`Attempting to stop backend job ID: ${currentJobId}`);
            fetch(`/api/stop-job/${currentJobId}`, { method: 'DELETE' })
                .then(response => {
                    if (response.ok) { return response.json(); }
                    console.error(`Backend stop request failed for ${currentJobId}: ${response.status}`);
                    addLogMessage(`Stop request sent, but backend responded with status ${response.status}. Task might continue.`);
                    return response.text().then(text => { throw new Error(`Backend Error: ${text}`) });
                })
                .then(data => { addLogMessage(`Backend confirmation: ${data.message}`); })
                .catch(error => { addLogMessage(`Error sending stop request to backend: ${error.message}`); console.error("Error stopping job on backend:", error); });
        } else {
             addLogMessage("No active job was running.");
        }
    });
    // --- End Reset Button Listener ---
 
    window.addEventListener('load', () => {
      const storedJobInfoString = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (storedJobInfoString) {
        try {
            const jobInfo = JSON.parse(storedJobInfoString);
            processingJobId = jobInfo.id;
 
            if (jobInfo.status === 'completed') {
                addLogMessage(`Restored completed job state for Job ID: ${processingJobId}.`);
                showCompletedState(processingJobId);
                hideElement(processButton); // Explicitly hide on completed restore
            } else if (jobInfo.status === 'processing') {
                addLogMessage(`Restoring processing job state for Job ID: ${processingJobId}...`);
                checkProcessingStatus(); // Calls showProcessingState which hides button
                connectEventSource(processingJobId);
                if (!checkStatusInterval) { checkStatusInterval = setInterval(checkProcessingStatus, 2000); }
            } else {
                 addLogMessage(`Stored job info has unexpected status '${jobInfo.status}'. Clearing and resetting.`);
                 localStorage.removeItem(LOCAL_STORAGE_KEY);
                 processingJobId = null;
                 showIdleState(); // Reset to initial state
            }
        } catch (e) {
            console.error("Error parsing stored job info:", e);
            localStorage.removeItem(LOCAL_STORAGE_KEY); // Clear corrupted data
            processingJobId = null;
            showIdleState(); // Reset to initial state
        }
      } else {
           showIdleState(); // Ensure clean state if nothing is stored
      }
    });
 
  </script>
</body>
</html>
"""
 
# --- Backend Python Code ---
 
# --- is_job_cancelled function ---
def is_job_cancelled(job_id: str) -> bool:
    return job_id in cancelled_jobs
 
# --- process_pdf_task (Add Zip Logging) ---
async def process_pdf_task(job_id, input_path):
    cancelled_jobs.discard(job_id)
    job_start_times[job_id] = datetime.now()
    # ... setup paths, log_queue, log_callback ...
    job_dir = os.path.dirname(input_path)
    output_normal_path = os.path.join(job_dir, "formatted_content_normal.docx")
    output_alltable_path = os.path.join(job_dir, "formatted_content_alltable.docx")
    output_zip_path = os.path.join(job_dir, f"formatted_content_{job_id[:8]}.zip")
    highlight_output_dir = os.path.join(job_dir, "highlighted_output")
    highlight_pdf_path = None
 
    log_queue = asyncio.Queue()
    log_queues[job_id] = log_queue
 
    async def log_callback(message: str):
        if is_job_cancelled(job_id): return
        print(f"Queueing log for {job_id}: {message}")
        await log_queue.put(message)
 
    try:
        if is_job_cancelled(job_id): raise asyncio.CancelledError("Job cancelled by user request.")
        # ... file cleanup with checks ...
        if os.path.exists(output_normal_path): os.remove(output_normal_path)
        if is_job_cancelled(job_id): raise asyncio.CancelledError("Job cancelled by user request.")
        if os.path.exists(output_alltable_path): os.remove(output_alltable_path)
        if is_job_cancelled(job_id): raise asyncio.CancelledError("Job cancelled by user request.")
        # ... etc ...
 
        await log_callback("Backend task started.")
        api_key = load_api_key()
        if is_job_cancelled(job_id): raise asyncio.CancelledError("Job cancelled by user request.")
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        await log_callback("OpenAI client initialized.")
 
        if is_job_cancelled(job_id): raise asyncio.CancelledError("Job cancelled by user request.")
        await log_callback("Calling main processing function (matthew_best.process_pdf)...")
        check_cancelled_func = partial(is_job_cancelled, job_id)
        highlight_pdf_path = await process_pdf(
            client, input_path, output_normal_path, output_alltable_path, job_dir,
            log_callback=log_callback,
            check_cancelled=check_cancelled_func
        )
        await log_callback(f"Main processing function finished. Highlight PDF path returned: {highlight_pdf_path}")
 
        if is_job_cancelled(job_id): raise asyncio.CancelledError("Job cancelled by user request.")
        # --- Check DOCX files exist ---
        normal_exists = os.path.exists(output_normal_path)
        alltable_exists = os.path.exists(output_alltable_path)
        highlight_exists = highlight_pdf_path and os.path.exists(highlight_pdf_path)
        await log_callback(f"Existence check before zipping: Normal DOCX: {normal_exists}, AllTable DOCX: {alltable_exists}, Highlight PDF: {highlight_exists}")
        # --- End Check DOCX files exist ---
 
        if is_job_cancelled(job_id): raise asyncio.CancelledError("Job cancelled by user request.")
        # --- Create zip with checks ---
        await log_callback(f"Attempting to create ZIP archive at: {output_zip_path}")
        files_added_to_zip = []
        try:
            with zipfile.ZipFile(output_zip_path, 'w') as zipf:
                if normal_exists:
                    zipf.write(output_normal_path, os.path.basename(output_normal_path))
                    files_added_to_zip.append(os.path.basename(output_normal_path))
                    await log_callback(f"Added normal DOCX to zip.")
                if alltable_exists:
                    zipf.write(output_alltable_path, os.path.basename(output_alltable_path))
                    files_added_to_zip.append(os.path.basename(output_alltable_path))
                    await log_callback(f"Added alltable DOCX to zip.")
                if highlight_exists:
                    zipf.write(highlight_pdf_path, os.path.basename(highlight_pdf_path))
                    files_added_to_zip.append(os.path.basename(highlight_pdf_path))
                    await log_callback(f"Added highlight PDF to zip.")
 
            if not files_added_to_zip:
                await log_callback("Warning: No files were found to add to the ZIP archive.")
            else:
                await log_callback(f"Successfully created ZIP with files: {', '.join(files_added_to_zip)}")
 
            zip_exists_after = os.path.exists(output_zip_path)
            await log_callback(f"ZIP file exists after creation? {zip_exists_after}")
 
        except Exception as zip_err:
            await log_callback(f"ERROR creating ZIP file: {zip_err}")
            raise # Re-raise the error to handle it below
        # --- End Create zip ---
 
        if is_job_cancelled(job_id): raise asyncio.CancelledError("Job cancelled by user request.")
        # ... set completed status ...
        processed_files[job_id] = {
            "status": "completed",
            "file_path": output_zip_path, # Store the path used for creation
            "file_id": job_id,
         }
        await log_callback("Processing complete.")
 
    except asyncio.CancelledError:
        # ... cancellation handling ...
        error_message = "Task cancelled by user request."
        print(f"Job {job_id}: {error_message}")
        processed_files[job_id] = { "status": "cancelled", "message": error_message }
        if not is_job_cancelled(job_id): # Check needed before logging
             await log_callback(error_message)
        # Cleanup
        if os.path.exists(output_normal_path): os.remove(output_normal_path)
        # ... other cleanup ...
       
    except Exception as e:
        # ... exception handling ...
        error_message = f"Error: {str(e)}"
        processed_files[job_id] = { "status": "error", "message": str(e) }
        print(f"Job {job_id}: Error processing file or creating ZIP: {error_message}")
        import traceback
        traceback.print_exc()
        if not is_job_cancelled(job_id):
             await log_callback(error_message)
             await log_callback("Processing failed.")
        # Cleanup
        if os.path.exists(output_normal_path): os.remove(output_normal_path)
        # ... other cleanup ...
 
    finally:
        # ... finally block cleanup ...
        if job_id in log_queues:
             if not is_job_cancelled(job_id): await log_queues[job_id].put(None)
             if job_id in log_queues: del log_queues[job_id] # Check again
             print(f"Removed log queue for job {job_id}")
        job_start_times.pop(job_id, None)
        if job_id in cancelled_jobs:
             processed_files[job_id] = {"status": "cancelled", "message": "Task cancelled by user."}
             cancelled_jobs.discard(job_id)
 
 
# --- Endpoints (log_stream_generator, stream_logs, index, process_file_background, check_status, stop_job, download_file) ---
# Ensure these are the same as the previous correct version, including the /api/stop-job endpoint.
 
async def log_stream_generator(job_id: str) -> AsyncGenerator[str, None]:
    # ... existing code ...
    queue = log_queues.get(job_id)
    if not queue: # ... handle missing queue ...
        print(f"Log queue not found for job {job_id}")
        yield f"data: Error: Log stream not found for job {job_id}. Task might be finished or failed.\n\n"
        return
    # ... rest of generator ...
    print(f"SSE Generator started for job {job_id}")
    yield f"data: Log stream connected for Job ID: {job_id}.\n\n"
    try:
        while True:
            message = await queue.get()
            if message is None: # ... handle end ...
                 break
            yield f"data: {message}\n\n"
            queue.task_done()
    # ... except/finally ...
    except asyncio.CancelledError: print(f"SSE Generator for job {job_id} cancelled.")
    except Exception as e: print(f"Error in SSE generator for job {job_id}: {e}"); yield f"data: Error in log stream: {e}\n\n"
    finally: print(f"SSE Generator finished for job {job_id}")
 
 
@app.get("/api/stream-logs/{job_id}")
async def stream_logs(request: Request, job_id: str):
    # ... existing code ...
    return StreamingResponse(log_stream_generator(job_id), media_type="text/event-stream")
 
@app.get("/", response_class=HTMLResponse)
async def index():
    # ... existing code ...
    return HTML_CONTENT
 
@app.post("/api/process-background")
async def process_file_background(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # ... existing code ...
    if file.content_type != "application/pdf": raise HTTPException(status_code=400, detail="Invalid file type.")
    job_id = str(uuid.uuid4())
    # ... setup dir/path ...
    temp_dir = os.path.join("temp", job_id)
    os.makedirs(temp_dir, exist_ok=True)
    input_file_path = os.path.join(temp_dir, file.filename or f"{job_id}.pdf")
    with open(input_file_path, "wb") as f: f.write(await file.read())
    # ... clear old state ...
    cancelled_jobs.discard(job_id)
    processed_files.pop(job_id, None)
    job_start_times.pop(job_id, None)
    # ... set initial state ...
    processed_files[job_id] = { "status": "processing" }
    job_start_times[job_id] = datetime.now()
    # ... add task ...
    background_tasks.add_task(process_pdf_task, job_id, input_file_path)
    return {"job_id": job_id}
 
 
@app.get("/api/status/{job_id}")
async def check_status(job_id: str):
    # ... existing code ...
    job_info = processed_files.get(job_id)
    if not job_info:
        if job_id in cancelled_jobs: return {"status": "cancelled", "message": "Cancellation requested."}
        raise HTTPException(status_code=404, detail="Job not found")
    if job_info.get("status") == "cancelled": return job_info
    status_data = job_info.copy()
    if status_data.get("status") == "processing":
        start_time = job_start_times.get(job_id)
        status_data["elapsed_time"] = (datetime.now() - start_time).seconds if start_time else 0
    status_data.pop("file_path", None)
    return status_data
 
@app.delete("/api/stop-job/{job_id}")
async def stop_job(job_id: str, request: Request):
    # ... existing code ...
    if job_id not in processed_files and job_id not in log_queues:
         print(f"Stop request for unknown or already finished job: {job_id}")
         cancelled_jobs.add(job_id)
         return {"message": f"Stop request acknowledged for job {job_id}. It might have already completed or failed."}
    print(f"Received stop request for job {job_id}")
    cancelled_jobs.add(job_id)
    log_queue = log_queues.get(job_id)
    if log_queue:
        try: log_queue.put_nowait("STOP_REQUESTED: Server received request to cancel this job.")
        except asyncio.QueueFull: print(f"Warning: Log queue full for job {job_id} during cancellation.")
        except Exception as e: print(f"Error putting cancel message in queue for job {job_id}: {e}")
    if job_id in processed_files:
        processed_files[job_id] = {"status": "cancelled", "message": "Cancellation requested by user."}
    return {"message": f"Cancellation requested for job {job_id}."}
 
@app.get("/api/download/{file_id}")
async def download_file(file_id: str):
    job_id = file_id
    print(f"Download request received for job_id: {job_id}") # Server log
 
    if job_id not in processed_files:
        print(f"Download failed: Job ID {job_id} not found in processed_files.") # Server log
        raise HTTPException(status_code=404, detail="Job not found")
 
    job_info = processed_files[job_id]
    print(f"Download request: Found job info: {job_info}") # Server log
 
    if job_info.get("status") == "cancelled":
        print(f"Download failed: Job {job_id} was cancelled.") # Server log
        raise HTTPException(status_code=400, detail="Job was cancelled.")
    if job_info.get("status") != "completed":
        print(f"Download failed: Job {job_id} status is {job_info.get('status')}, not 'completed'.") # Server log
        raise HTTPException(status_code=400, detail="File processing not completed or failed")
 
    zip_file_path = job_info.get("file_path")
    print(f"Download request: Path from job_info: {zip_file_path}") # Server log
 
    if not zip_file_path or not os.path.exists(zip_file_path):
        print(f"Download warning: Path '{zip_file_path}' from job_info not found or doesn't exist.") # Server log
        # --- Try fallback ---
        job_dir = os.path.join("temp", job_id)
        potential_zip_path = os.path.join(job_dir, f"formatted_content_{job_id[:8]}.zip")
        print(f"Download request: Checking fallback path: {potential_zip_path}") # Server log
        if os.path.exists(potential_zip_path):
            print(f"Download request: Fallback path exists. Using it.") # Server log
            zip_file_path = potential_zip_path
        else:
            print(f"Download failed: Fallback path '{potential_zip_path}' also not found.") # Server log
            raise HTTPException(status_code=404, detail="ZIP file not found")
            # --- End fallback ---
 
    download_filename = os.path.basename(zip_file_path)
    print(f"Download request: Preparing to send file: {zip_file_path} as {download_filename}") # Server log
    return FileResponse(path=zip_file_path, filename=download_filename, media_type="application/zip")
 
 
# --- Main Block ---
if __name__ == "__main__":
    # ... existing code ...
    port = 6000
    host = "0.0.0.0"
    print(f"Server running at: http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, timeout_keep_alive=120)
