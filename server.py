import os

from flask import Flask, request, send_file
from synthesise import generate_voice

app = Flask(__name__)

@app.post("/generate")
def generate_voice_request():
    text = request.get_json()["text"]
    print(text)
    voice = request.get_json()["voice"]
    print(voice)
    result = generate_voice(text, voice)
    if result == 0:
        return {
            "message": "success",
            "path": "test_generated.wav"
        }
    else:
        return {"message": "error"}

@app.route("/dl/<string:filename>")
def download(filename):
    result = send_file(
        os.path.join("generated_files", filename),
        mimetype="audio/wav",
        as_attachment=True,
    )
    return result

port = 6969
print("hello")
if __name__ == "__main__":
    app.run(port=port, host="0.0.0.0")


