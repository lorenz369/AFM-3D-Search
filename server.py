import os
import subprocess
import threading
import uuid
import time
import logging
import logging.config
from pathlib import Path
import sys

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch

# --- Path Setup ---
# Get the root directory of your project (AFM-3D-Search)
PROJECT_ROOT = Path(__file__).parent.resolve()
# Add the 'src' directory from the 'ml-depth-pro' submodule to the Python path
SRC_PATH = PROJECT_ROOT / "ml-depth-pro" / "src"
sys.path.insert(0, str(SRC_PATH))

# --- Corrected Imports ---
# Now that the path is set correctly, we can import 'depth_pro' directly
try:
    from depth_pro.cli.run import run as run_depth_pro
    from depth_pro import create_model_and_transforms
except ImportError as e:
    # This will give a more helpful error if the path is still wrong
    raise ImportError(
        f"Could not import 'depth_pro'. "
        f"Please check that the path '{SRC_PATH}' is correct and contains the 'depth_pro' directory. Original error: {e}"
    )


# --- Logging Configuration ---
try:
    from logging_config import LOGGING_CONFIG
    logging.config.dictConfig(LOGGING_CONFIG)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


# --- Configuration ---
TEMP_UPLOAD_FOLDER = PROJECT_ROOT / 'temp_uploads'
ALLOWED_EXTENSIONS = {'zip'}
os.makedirs(TEMP_UPLOAD_FOLDER, exist_ok=True)


# --- Model Loading (Done ONCE per Gunicorn worker) ---
def get_torch_device() -> torch.device:
    """Get the Torch device and log it."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    logger.info(f"Using device: {device}")
    return device

logger.info("Initializing: Loading Depth Pro model...")
DEVICE = get_torch_device()
MODEL, TRANSFORM = create_model_and_transforms(
    device=DEVICE,
    precision=torch.half,
)
MODEL.eval()
logger.info("✅ Model loaded and ready.")


# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(TEMP_UPLOAD_FOLDER)

# --- Helper Class for Arguments ---
class RunArgs:
    def __init__(self, image_path, output_path):
        self.image_path = image_path
        self.output_path = output_path
        self.skip_display = True
        self.verbose = False

def allowed_file(filename):
    """Checks for allowed file extensions."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_data_in_background(filepath, original_filename):
    """
    Unzips the file and runs depth estimation using the pre-loaded model.
    """
    logger.info(f"Starting background processing for '{original_filename}'.")
    try:
        output_parent_folder = Path.home() / "arkit_uploads"
        os.makedirs(output_parent_folder, exist_ok=True)

        logger.info(f"Unzipping '{original_filename}'...")
        subprocess.run(["unzip", "-o", filepath, "-d", str(output_parent_folder)], check=True, capture_output=True, text=True)
        logger.info("Unzip complete.")

        base_name = original_filename.rsplit('.', 1)[0]
        extracted_folder_path = output_parent_folder / base_name
        
        final_destination_path = output_parent_folder / f"{base_name}_processed"
        counter = 1
        while final_destination_path.exists():
            final_destination_path = output_parent_folder / f"{base_name}_processed_{counter}"
            counter += 1
        
        if extracted_folder_path.exists():
            os.rename(extracted_folder_path, final_destination_path)
            logger.info(f"Folder organized. Final destination: {final_destination_path}")
        else:
            logger.error(f"Expected folder '{extracted_folder_path}' not found after unzip.")
            return

        color_folder = final_destination_path / "color"
        depth_output_folder = final_destination_path / "depth"

        if color_folder.is_dir():
            logger.info(f"Running Depth Estimation on '{color_folder}'...")
            args = RunArgs(image_path=color_folder, output_path=depth_output_folder)
            
            start_time = time.time()
            run_depth_pro(args, model=MODEL, transform=TRANSFORM)
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"✅ Depth Estimation complete. Time taken: {duration:.2f} seconds.")
        else:
            logger.error(f"'color' folder not found in {final_destination_path}.")

    except subprocess.CalledProcessError as e:
        logger.exception("A subprocess error occurred during unzip.")
        logger.error(f"STDERR: {e.stderr.strip()}")
    except Exception as e:
        logger.exception(f"A critical error occurred during background processing: {e}")
    finally:
        logger.info(f"Cleaning up temporary file: {filepath}")
        if os.path.exists(filepath):
            os.remove(filepath)

# --- API Endpoint ---
@app.route('/process-scene', methods=['POST'])
def upload_and_process_scene():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400
    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + ".zip")
        file.save(temp_filepath)
        
        thread = threading.Thread(target=process_data_in_background, args=(temp_filepath, original_filename))
        thread.start()
        
        return jsonify({
            "message": "File upload accepted. Processing in the background.",
            "filename": original_filename,
        }), 202
    else:
        return jsonify({"error": "File type not allowed."}), 400

# This block is not used when running with Gunicorn
if __name__ == '__main__':
    logger.warning("Running in development mode. Use a WSGI server like Gunicorn for production.")
    app.run(host='0.0.0.0', port=8080, debug=False)