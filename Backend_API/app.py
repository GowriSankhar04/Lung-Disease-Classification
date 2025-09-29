from flask import Flask, request, jsonify

from pp import extract_all_features, predict_audio
import os
import traceback
import sys
sys.stdout.reconfigure(line_buffering=True)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return {"status": "API is running. Use POST /extract to send audio."}

@app.route('/extract', methods=['POST'])
def extract():
    print("✅ Received a POST request to /extract")

    if 'audio' not in request.files:
        return jsonify({"status": "error", "message": "No audio file provided"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    try:
        # Save audio
        # Save audio
        os.makedirs("/tmp", exist_ok=True)   # <-- create folder if not exists
        audio_path = os.path.join("/tmp", "input.wav")
        file.save(audio_path)
        print(f"✅ Audio file saved at: {audio_path}")



        # Get prediction
        feature_result = predict_audio(audio_path)

        if feature_result is None:
            return jsonify({"status": "error", "message": "Feature extraction failed"}), 500

        # Wrap response with status
        if isinstance(feature_result, dict):
            return jsonify({
            "status": "success",
            **feature_result
                 })
        else:
            return jsonify({
           "status":"Error",
             "message": feature_result
                    })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ✅ Start server
#if __name__ == "__main__":
#    port = int(os.environ.get("PORT", 10000))
#    print(f"✅ Flask app running on port {port}")
#    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))  # Hugging Face requires 7860
    print(f"✅ Flask app running on port {port}")
    app.run(host="0.0.0.0", port=port)
