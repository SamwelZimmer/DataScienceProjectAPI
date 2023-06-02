import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from search import process_term
from face_search import get_similar_faces

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'https://data-science-project-frontend.vercel.app'])

@app.route('/')
def hello_world():
    return 'Hello from Flask!'


@app.route('/time')
def get_current_time():
    return {'time': time.time()}


@app.route('/money')
def money():
    return {"Money": ["£100", "£420", "£211"]}


@app.route('/maths')
def maths():
    term = request.args.get('term')
    # Process the search term and get the results
    results = process_maths(term)
    return jsonify(results)


@app.route('/search')
def search():
    term = request.args.get('term')
    # Process the search term and get the results
    results = process_term(term)
    print(results)
    return jsonify(results)


@app.route('/faces')
def facial_similarity():
    url = request.args.get('img')
    print("url", url)
    try:
        results = get_similar_faces(url)
    except:
        results = None
    return {'term': url, 'result': results}


def process_maths(term):
    if term.isdigit():
        return {'term': term, 'result': float(term) * 10}
    else:
        return {'term': term, 'result': "not a number - from dev"}


if __name__ == '__main__':
    app.run(debug=True)

