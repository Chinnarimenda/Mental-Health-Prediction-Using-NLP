from flask import Flask, render_template, request
from model import MentalHealthModel, model_exists

app = Flask(__name__)

# Initialize the model
mental_health_model = MentalHealthModel()

# Load or train the model on startup
@app.before_request
def initialize_model():
    """Initialize model before first request"""
    if model_exists():
        print("Loading existing model...")
        mental_health_model.load_model()
    else:
        print("No trained model found. Please run model.py first to train the model.")
        print("You can train the model by running: python model.py")


@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get text from form data
        text_input = request.form.get('text', '').strip()
        
        # Validate input
        if not text_input:
            return render_template('index.html', 
                                 error='Please provide some text to analyze')
        
        # Check if model is loaded
        if mental_health_model.model is None:
            return render_template('index.html',
                                 error='Model not loaded. Please train the model first by running: python model.py')
        
        # Get prediction
        prediction_score = mental_health_model.predict(text_input)
        
        # Interpret the result
        result = mental_health_model.interpret_prediction(prediction_score)
        
        # Render template with results
        return render_template('index.html',
                             prediction_score=result['prediction_score'],
                             risk_level=result['risk_level'],
                             message=result['message'],
                             input_text=text_input)
    
    except Exception as e:
        return render_template('index.html', 
                             error=f'An error occurred: {str(e)}',
                             input_text=request.form.get('text', ''))


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)