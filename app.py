from flask import Flask, request, render_template, json, jsonify

from review.vo import Review
from review.service import ReviewService

s = ReviewService()
app = Flask(__name__)

@app.route('/')
def root():
    return render_template('index.html')

@app.route('/fit')  
def fit():
    s.review_fit()
    score = s.review_test()
    return render_template('index.html', score=score)

@app.route('/review_action', methods=['POST'])
def review_action():
    data = json.loads(request.data)
    search =  data.get('search')
    df = s.getMovieReviews(search)
    res = []
    data_list = df.values.tolist()
    for d in data_list:
        res.append(Review(date=d[0], writer=d[1], review=d[2], rating=d[3], type=d[5]))
    return jsonify(result=json.dumps(res, default=str))

if __name__ == "__main__": 
    app.run(host = "127.0.0.1", port = 4400, debug = True)