from flask import Flask, request, jsonify, send_file
from io import BytesIO
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
from emg_pytorch_cnn import EMG_CNN
from emg_image_utils import create_emg_image

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Configure production settings
app.config.update(
    DEBUG=False,
    PROPAGATE_EXCEPTIONS=True
)

# Load model
try:
    model = EMG_CNN((100, 1), 8)
    model.load_state_dict(torch.load("models/emg_cnn_model.pth", map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Muscle mapping
muscles = ['Biceps', 'Triceps', 'Deltoids', 'Pectorals', 'Quadriceps', 'Hamstrings', 'Abdominals', 'Trapezius']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Process EMG data file
        df = pd.read_csv(file)
        data = df.values.astype(np.float32)
        
        # Prepare tensor input
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)
        data = torch.tensor(data).unsqueeze(0).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(data)
            prediction_probs = torch.nn.functional.softmax(output, dim=1).numpy()[0]

        # Format results
        results = {
            'prediction': prediction_probs.tolist(),
            'muscle_activation': {
                muscle: float(prediction_probs[i])
                for i, muscle in enumerate(muscles)
            }
        }

        # Generate EMG visualization image
        emg_data_flat = data.numpy().flatten()
        emg_img = create_emg_image(emg_data_flat)
        img_buffer = BytesIO()
        emg_img.save(img_buffer, format='PNG')
        results['emg_image'] = img_buffer.getvalue().hex()

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=5000)
    except ImportError:
        app.run(host="0.0.0.0", port=5000)
