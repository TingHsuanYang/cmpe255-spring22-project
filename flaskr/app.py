import flask
from flask import request, jsonify
from flaskr.q1model import Q1Model

app = flask.Flask(__name__)
app.config["DEBUG"] = True

q1 = Q1Model()

@app.route('/Q1', methods=['POST'])
def q1_predict():
    body = request.json
    return jsonify(q1.predict(genres=body['genres'], rating=body['rating']))

def run():
    app.run()

if __name__ == '__main__':
    app.run()