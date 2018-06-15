from flask import Flask, jsonify
import process_query
from process_query import predict_IOB_labels

app = Flask(__name__)

@app.route('/')
def index():
    return "hello world!"

@app.route('/flightreservation/api/get_entities/<query>', methods = ['GET'])
def extract_entities(query):
    labels = predict_IOB_labels(query)
    return jsonify(labels)

if __name__ == '__main__':
    app.run(debug=True)

