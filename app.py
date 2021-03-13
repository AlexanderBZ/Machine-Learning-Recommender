
from flask import Flask,request,jsonify
from flask_cors import CORS
import beatly_recommend
import pandas as pd

app = Flask(__name__)
CORS(app)


@app.route('/album', methods=['GET'])
def recommend_movies():
    res = beatly_recommend.content_based_recommender(request.args.get('album_name'))
    res_formatted = res.to_json()
    return res_formatted


if __name__ == '__main__':
    app.run(port=5000, debug=True)