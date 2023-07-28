import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from search import process_term
from face_search import get_similar_faces
from scripts.neuron_signal_generator import generate_signal
from scripts.noise_and_filtering import generate_electrode_signal

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://localhost:5000', 'https://data-science-project-frontend.vercel.app'])

@app.route('/')
def hello_world():
    return 'Hello from Flask!'

@app.route('/generate_neuron_signal', methods=['POST'])
def neuron_signal():
    data = request.get_json()
    neuron_type, lmbda, v_rest, v_thres, t_ref, fix_random_seed = data["neuron_type"], data["lambda"], data["v_rest"], data["v_thres"], data["t_ref"], data["fix_random_seed"]

    signal_data = generate_signal(
        neuron_type=neuron_type, 
        lmbda=lmbda, 
        v_rest=v_rest, 
        v_thres=v_thres, 
        t_ref=t_ref,
        fix_random_seed=fix_random_seed
    )
    
    return signal_data

@app.route('/process_signal', methods=['POST'])
def process_signal():
    data = request.get_json()
    param_data = data["processingParams"]
    signal = data["neuronSignal"]
    decay_type, decay_rate, noise_type, noise_std, filter_type, low, high = param_data["decay_type"], param_data["decay_rate"], param_data["noise_type"], param_data["noise_std"], param_data["filter_type"], param_data["low"], param_data["high"]

    signal_data = generate_electrode_signal(
        signal=signal,
        decay_type=decay_type, 
        decay_rate=decay_rate, 
        noise_type=noise_type, 
        noise_std=noise_std, 
        filter_type=filter_type, 
        low=low, 
        high=high
    )
    
    return signal_data


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

