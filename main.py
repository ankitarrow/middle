from flask import Flask, request, jsonify
import requests
from azure.storage.blob import BlobServiceClient

app = Flask(__name__)

def get_azure_url(storage_account_name, container_name, file_name):
    base_url = f"https://{storage_account_name}.blob.core.windows.net"
    return f"{base_url}/{container_name}/{file_name}"

def azure_upload(file_url, new_file_name, container_name, storage_account_name, sas_token, retries=2):
    while retries > 0:
        try:
            if not file_url or not new_file_name:
                raise ValueError("Missing required parameters")
            response = requests.get(file_url)
            response.raise_for_status()
            file_bytes = response.content

            blob_service_client = BlobServiceClient(
                account_url=f"https://{storage_account_name}.blob.core.windows.net",
                credential=sas_token
            )

            container_client = blob_service_client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(new_file_name)

            blob_client.upload_blob(file_bytes, overwrite=True)
            azure_url = get_azure_url(storage_account_name, container_name, new_file_name)
            return azure_url

        except Exception as error:
            retries -= 1
            if retries == 0:
                raise Exception("Failed to upload file to Azure Blob Storage after retries")

@app.route('/upload-to-azure', methods=['POST'])
def upload_to_azure():
    try:
        data = request.json
        file_url = data.get('file_url')
        file_name = data.get('file_name')
        container_name = "sound"
        storage_account_name = data.get('storage_account_name')
        sas_token = data.get('sas_token')

        if not all([file_url, file_name, storage_account_name, sas_token]):
            return jsonify({"error": "Missing required parameters"}), 400

        azure_url = azure_upload(file_url, file_name, container_name, storage_account_name, sas_token)
        return jsonify({"azure_url": azure_url}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-audio', methods=['POST'])
def generate_audio():
    try:
        data = request.json
        api_key = data.get('api_key')
        model_name = data.get('model_name')
        prompt = data.get('prompt')
        duration = data.get('duration')

        if not api_key or not model_name or not prompt or not duration:
            return jsonify({"error": "Missing required parameters"}), 400
        
        if not isinstance(duration, (int, float)) or duration <= 0:
            return jsonify({"error": "Invalid duration. It must be a positive number"}), 400

        payload = {
            "input": {
                "top_k": 250,
                "top_p": 0,
                "prompt": prompt,
                "duration": duration,
                "temperature": 1,
                "continuation": False,
                "model_version": "stereo-large",
                "output_format": "wav",
                "continuation_start": 0,
                "multi_band_diffusion": False,
                "normalization_strategy": "peak",
                "classifier_free_guidance": 3,
            },
            "version": model_name,
        }

        url = "https://api.replicate.com/v1/predictions"
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            try:
                error_details = response.json()
            except ValueError:
                error_details = {"message": response.text}
            return jsonify({"error": "Failed to generate audio", "details": error_details}), response.status_code

    except Exception as e:
        return jsonify({"error": "An error occurred", "details": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
