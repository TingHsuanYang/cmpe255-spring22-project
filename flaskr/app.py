import flask
from flask import request, jsonify
from flaskr.q1model import Q1Model
from flaskr.q3model import Q3Model
from flaskr.q4model import Q4Model

app = flask.Flask(__name__)
app.config["DEBUG"] = True

q1 = Q1Model()
q3 = Q3Model()
q4 = Q4Model()

@app.route('/Q1', methods=['POST'])
def q1_predict():
    body = request.json
    return jsonify(q1.predict(genres=body['genres'], rating=body['rating']))
    
@app.route('/Q3', methods=['POST'])
def q3_predict():
    body = request.json
    return jsonify(q3.predict(platform=body['platform'], developer=body['developer']))

@app.route('/Q4', methods=['POST'])
def q4_predict():
    body = request.json
    return jsonify(q4.predict(genres=body['genres'], type=body['type'], rating=body['rating']))

def run():
    app.run()

if __name__ == '__main__':
    app.run()
