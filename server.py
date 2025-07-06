import os
import subprocess
import threading
import uuid
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# --- Configuration ---
TEMP_UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'zip'}
DEPTH_PRO_DIR = Path(__file__).parent / 'ml-depth-pro'

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
    print(f"--- [Background Thread] Starting processing for {original_filename} ---")
    try:
        output_parent_folder = Path.home() / "arkit_uploads"
        os.makedirs(output_parent_folder, exist_ok=True)

        print(f"Unzipping {original_filename} into: {output_parent_folder}")
        subprocess.run(["unzip", "-o", filepath, "-d", str(output_parent_folder)], check=True, capture_output=True, text=True)

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
            print(f"--- ✅ [Background Thread] Success! Final folder: {final_destination_path} ---")

            color_folder = final_destination_path / "color"
            depth_output_folder = final_destination_path / "depth"
            
            if color_folder.is_dir():
                print(f"--- 🏃 [Background Thread] Running Depth Estimation on {color_folder} ---")
                
                cmd = [
                    "conda", "run", "-n", "depth-pro",
                    "depth-pro-run",
                    "-i", str(color_folder),
                    "-o", str(depth_output_folder),
                    "--skip-display" 
                ]
                
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(result.stdout) 
                
                print(f"--- ✅ [Background Thread] Depth Estimation complete. Output in: {depth_output_folder} ---")
            else:
                print(f"--- ❌ [Background Thread] Error: 'color' folder not found in {final_destination_path}. ---")

        else:
            print(f"--- ❌ [Background Thread] Error: Expected folder '{extracted_folder_path}' not found after unzip. ---")

    except subprocess.CalledProcessError as e:
        print(f"--- ❌ [Background Thread] An error occurred during processing: ---")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
    except Exception as e:
        print(f"--- ❌ [Background Thread] A critical error occurred: {e} ---")
    finally:
        print(f"--- [Background Thread] Cleaning up temporary file: {filepath} ---")
        if os.path.exists(filepath):
            os.remove(filepath)

# --- API Endpoint ---
@app.route('/process-scene', methods=['POST'])
def upload_and_process_scene():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename or "")
        temp_filename = str(uuid.uuid4()) + ".zip"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_filepath)
        thread = threading.Thread(target=process_data_in_background, args=(temp_filepath, original_filename))
        thread.start()
        return jsonify({
            "message": "File upload accepted. Processing in the background.",
            "filename": original_filename,
        }), 200
    else:
        return jsonify({"error": "File type not allowed."}), 400

# --- Running the App ---
if __name__ == '__main__':
    print("This script should be run with Gunicorn, not directly.")