import time
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import ast
from search import process_term
from face_search import get_similar_faces
from scripts.neuron_signal_generator import generate_signal
from scripts.noise_and_filtering import generate_electrode_signal
from scripts.simulator import simulate_recording
from scripts.spike_extraction import get_threshold_value, get_waveform_data
from scripts.reduce import dimensional_reduction
from scripts.clustering import get_clusters
from scripts.triangulate_neurons import triangulate_neurons

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://localhost:5000', 'https://data-science-project-frontend.vercel.app'])

@app.route('/')
def hello_world():
    return 'Hello from Flask!'

def jsonify_dict(d):
    json_dict = {}
    for key, value in d.items():
        if isinstance(key, tuple):
            key = str(key)
        if isinstance(value, dict):
            value = jsonify_dict(value)
        json_dict[key] = value
    return json_dict

def stringify_keys(d):
    return {str(k): stringify_keys(v) if isinstance(v, dict) else v for k, v in d.items()}


@app.route('/triangulate', methods=['POST'])
def triangulate():
    data = request.get_json()

    signals, placements, labels, waveforms, decay_type = ast.literal_eval(data["recordedSignals"]), ast.literal_eval(data["placements"]), ast.literal_eval(data["predictedLabels"]), ast.literal_eval(data["waveforms"]), data["decay_type"]
    signals = signals["signals"]
    print("waveforms", type(waveforms), len(waveforms), len(waveforms[0]), len(waveforms[0][0]), type(waveforms[0][0][0]))
    print("signals", type(signals), len(signals), len(signals[0]), type(signals[0][0]))
    print("labels", type(labels), len(labels), type(labels[0]))
    print("placements", type(placements), len(placements), type(placements[0]))

    construction_dict = triangulate_neurons(signals=signals, placements=placements, labels=labels, waveforms=waveforms, decay_type=decay_type)

    construction_dict = stringify_keys(construction_dict)

    return construction_dict


@app.route('/cluster', methods=['POST'])
def cluster():
    data = request.get_json()
    params, reduced_data = data["clusteringParams"], data["reductionData"]
    cluster_type, k_type, k = params["cluster_type"], params["k_type"], params["k"]

    reduced_data = ast.literal_eval(reduced_data)
    predicted_labels = get_clusters(cluster_type, k_type, k, reduced_data)

    return { "predicted_labels": predicted_labels }


@app.route('/reduce', methods=['POST'])
def reduce():
    data = request.get_json()
    params, waveforms = data["featuresParams"], data["waveforms"]
    model, n_components = params["reduction_type"], params["n_components"]

    # convert from string to a list
    waveforms = ast.literal_eval(waveforms)

    reduced_data = dimensional_reduction(model=model, n_components=n_components, waveforms=waveforms)
    return reduced_data


@app.route('/waveforms', methods=['POST'])
def waveforms():
    data = request.get_json()
    signals = data["signals"]
    extraction_data = data["extractionParams"]
    multiplier, waveform_duration = extraction_data["thresholdMultiplier"], extraction_data["waveformDuration"]

    waveforms, waveform_info = get_waveform_data(signals, multiplier, waveform_duration)

    return { "waveforms": waveforms.tolist(), "waveform_info": waveform_info.tolist() }


@app.route('/threshold_value', methods=['POST'])
def threshold_value():
    data = request.get_json()
    signal = data["signal"]
    multiplier = data["thresholdMultiplier"]

    threshold = get_threshold_value(signal, multiplier)

    return jsonify({"threshold": threshold})


@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.get_json()
    placements, neuron_params, processing_params = data["placements"], data["neuronParams"], data["processingParams"]

    time, filtered_signals = simulate_recording(placements, neuron_params, processing_params)

    return { "time": time, "signals": filtered_signals }


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

