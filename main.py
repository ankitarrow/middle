from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

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
