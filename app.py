# app.py
from flask import Flask, jsonify

app = Flask(__name__)

# Routes
@app.route('/')
def index():
    return jsonify({"message": "Your deployment is successful"})

if __name__ == '__main__':
    app.run(debug=True)