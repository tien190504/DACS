"""
Flask Web Application for S&P 500 Prediction Dashboard
Serves the web interface and handles routing
"""

import os
from flask import Flask, render_template, send_from_directory, jsonify, request
from datetime import datetime
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__,
            template_folder='app/views',
            static_folder='static')

# Configuration
API_BASE_URL = 'http://localhost:5000'  # API server URL


@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    try:
        return render_template('dashboard.html')
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        return f"Error loading dashboard: {e}", 500


@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        # Check if API server is running
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        api_status = response.status_code == 200
    except:
        api_status = False

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'api_connected': api_status
    })


@app.route('/api/dashboard_data')
def dashboard_data():
    """Proxy endpoint for dashboard data"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/realtime_predict", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return jsonify({'error': 'Failed to fetch prediction data'}), 500
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching dashboard data: {e}")
        return jsonify({'error': 'API server unavailable'}), 503


@app.route('/api/model_status')
def model_status():
    """Get model status information"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/model_info", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return jsonify({'error': 'Failed to fetch model info'}), 500
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching model status: {e}")
        return jsonify({'error': 'API server unavailable', 'connected': False}), 503


@app.route('/api/predictions')
def get_predictions():
    """Proxy endpoint for predictions"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/predict", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return jsonify({'error': 'Failed to fetch predictions'}), 500
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching predictions: {e}")
        return jsonify({'error': 'API server unavailable'}), 503


@app.route('/api/historical')
def get_historical():
    """Proxy endpoint for historical data"""
    try:
        days = request.args.get('days', 30)
        response = requests.get(f"{API_BASE_URL}/api/historical_predictions?days={days}", timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            return jsonify({'error': 'Failed to fetch historical data'}), 500
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching historical data: {e}")
        return jsonify({'error': 'API server unavailable'}), 503


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500


# Create templates directory if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

if not os.path.exists('static'):
    os.makedirs('static')

if __name__ == '__main__':
    print("=" * 60)
    print("🌐 Starting S&P 500 Prediction Web Application")
    print("=" * 60)
    print(f"📊 Dashboard URL: http://localhost:8080")
    print(f"🔗 API Server: {API_BASE_URL}")
    print(f"📁 Templates: {app.template_folder}")
    print(f"📁 Static: {app.static_folder}")
    print("=" * 60)
    print("⚠️  Make sure to run api.py first on port 5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=8080)