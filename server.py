import os
import subprocess
import threading
import uuid
import time
import logging
import logging.config
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2

# Import and apply the logging configuration
from logging_config import LOGGING_CONFIG
logging.config.dictConfig(LOGGING_CONFIG)

# Get the logger for this module
logger = logging.getLogger(__name__)


# --- Configuration ---
TEMP_UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'zip'}
PROJECT_ROOT = Path(__file__).parent.resolve()
DEPTH_PRO_SUBMODULE_DIR = PROJECT_ROOT / 'ml-depth-pro'


# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = TEMP_UPLOAD_FOLDER
os.makedirs(TEMP_UPLOAD_FOLDER, exist_ok=True)

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_data_in_background(filepath, original_filename):
    """
    This function unzips the file and then runs the depth estimation process.
    """
    logger.info(f"Starting processing for '{original_filename}'.")
    try:
        # --- Unzipping ---
        output_parent_folder = Path.home() / "arkit_uploads"
        os.makedirs(output_parent_folder, exist_ok=True)

        logger.info(f"Unzipping '{original_filename}'...")
        subprocess.run(["unzip", "-o", filepath, "-d", str(output_parent_folder)], check=True, capture_output=True, text=True)
        logger.info("Unzip complete.")

        # --- Renaming and Organizing ---
        base_name = original_filename.rsplit('.', 1)[0]
        extracted_folder_path = output_parent_folder / base_name

        unique_folder_name = f"{base_name}_processed"
        final_destination_path = output_parent_folder / unique_folder_name
        
        counter = 1
        while final_destination_path.exists():
            unique_folder_name = f"{base_name}_processed_{counter}"
            final_destination_path = output_parent_folder / unique_folder_name
            counter += 1

        if extracted_folder_path.exists():
            os.rename(extracted_folder_path, final_destination_path)
            logger.info(f"Folder organized. Final destination: {final_destination_path}")

            # --- AUTOMATED DEPTH ESTIMATION ---
            color_folder = final_destination_path / "color"
            depth_output_folder = final_destination_path / "depth"

            try:
                image_files = os.listdir(color_folder)
                if image_files:
                    first_image_path = color_folder / image_files[0]
                    image = cv2.imread(str(first_image_path))
                    h, w, c = image.shape
                    logger.info(f"Verified image shape on server: W:{w}, H:{h}")
                else:
                    logger.warning("No images found in color folder to check shape.")
            except Exception as e:
                logger.error(f"Could not check image shape: {e}")
            
            if color_folder.is_dir():
                logger.info(f"Running Depth Estimation on '{color_folder}'.")
                
                cmd = [
                    "conda", "run", "-n", "depth-pro", "--cwd", str(DEPTH_PRO_SUBMODULE_DIR),
                    "depth-pro-run",
                    "-i", str(color_folder),
                    "-o", str(depth_output_folder),
                    "--skip-display" 
                ]
                
                logger.debug(f"Executing command: {' '.join(cmd)}")
                
                # --- Timing Logic ---
                start_time = time.time()
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                end_time = time.time()
                duration = end_time - start_time
                
                # Log the output from the ML script
                if result.stdout:
                    logger.debug(f"Depth Estimation STDOUT: {result.stdout.strip()}")
                if result.stderr:
                    logger.warning(f"Depth Estimation STDERR: {result.stderr.strip()}")
                
                logger.info(f"Depth Estimation complete. Time taken: {duration:.2f} seconds.")
            else:
                logger.error(f"'color' folder not found in {final_destination_path}.")
        else:
             logger.error(f"Expected folder '{extracted_folder_path}' not found after unzip.")

    except subprocess.CalledProcessError as e:
        # Using logger.exception automatically includes stack trace information
        logger.exception(f"A subprocess error occurred during processing.")
        logger.error(f"STDOUT: {e.stdout.strip()}")
        logger.error(f"STDERR: {e.stderr.strip()}")
    except Exception as e:
        logger.exception(f"A critical error occurred: {e}")
    finally:
        logger.info(f"Cleaning up temporary file: {filepath}")
        if os.path.exists(filepath):
            os.remove(filepath)

# --- API Endpoint ---
@app.route('/process-scene', methods=['POST'])
def upload_and_process_scene():
    if 'file' not in request.files:
        logger.warning("Upload attempt failed: No file part in request.")
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        logger.warning("Upload attempt failed: No file selected.")
        return jsonify({"error": "No file selected"}), 400
    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename or "")
        temp_filename = str(uuid.uuid4()) + ".zip"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_filepath)
        
        logger.info(f"File '{original_filename}' accepted. Starting background processing.")
        thread = threading.Thread(target=process_data_in_background, args=(temp_filepath, original_filename))
        thread.start()
        
        return jsonify({
            "message": "File upload accepted. Processing in the background.",
            "filename": original_filename,
        }), 200
    else:
        logger.warning(f"Upload attempt failed: File type not allowed for '{file.filename}'.")
        return jsonify({"error": "File type not allowed."}), 400

if __name__ == '__main__':
    logger.warning("This script should be run with a production server like Gunicorn, not directly.")