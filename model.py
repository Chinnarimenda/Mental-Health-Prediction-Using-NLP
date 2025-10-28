import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GRU, Dense, SpatialDropout1D, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import pickle
import os

class MentalHealthModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_length = 100
        self.vocab_size = 10000
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        text = re.sub(r"http\S+|www\S+|https\S+", "", str(text))
        text = re.sub(r"[^A-Za-z\s]", "", str(text))
        text = text.lower().strip()
        #remove stopwords like feel,one,life
        stopwords = set({   
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
            "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
            "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
            "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", 
            "are", "was", "were", "be", "been", "being",  "have",  "has",  "had",
            "having","do","does","did","doing","a","an","the","and","but","if","or","because",
            "as","until","while","of","at","by","for","with","about","against","between",
            "into","through","during","before","after","above","below","to","from","up",
            "down","in","out","on","off","over","under","again","further","then","once",
            "here","there","when","where","why","how","all","any","both","each","few",
            "more","most","other","some","such","no","nor","not","only","own","same",
            "so","than","too","very","s","one","life","feel","think"} )
        text = ' '.join([word for word in text.split() if word not in stopwords])
        return text
    
    def train_model(self, csv_file_path):
        """Train the Bidirectional GRU model"""
        print("Loading dataset...")
        df = pd.read_csv(csv_file_path)
        
        # Clean text
        print("Cleaning text data...")
        df["clean_text"] = df["text"].apply(self.clean_text)
        
        text = df["clean_text"].values
        labels = df["label"].values
        
        # Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            text, labels, test_size=0.2, random_state=42
        )
        
        # Tokenize
        print("Tokenizing text...")
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(X_train)
        
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_length, padding='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_length, padding='post')
        
        # Calculate class weights
        print("Calculating class weights...")
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights = dict(enumerate(class_weights))
        print("Class weights:", class_weights)
        
        # Build model with Bidirectional GRU
        print("Building model...")
        self.model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=128, input_length=self.max_length),
            SpatialDropout1D(0.3),
            Bidirectional(GRU(128, dropout=0.3, recurrent_dropout=0.3)),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.build(input_shape=(None, 100))
        self.model.summary()
        
        # Train model
        print("Training model...")
        history = self.model.fit(
            X_train_pad,
            y_train,
            epochs=10,
            batch_size=64,
            validation_data=(X_test_pad, y_test),
            class_weight=class_weights,
            verbose=2
        )
        
        # Save model and tokenizer
        self.save_model()
        
        print("Training completed!")
        return history
    
    def save_model(self):
        """Save model and tokenizer"""
        print("Saving model and tokenizer...")
        self.model.save('mental_health_model.h5')
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model and tokenizer saved successfully!")
    
    def load_model(self):
        """Load pre-trained model and tokenizer"""
        try:
            print("Loading model and tokenizer...")
            self.model = load_model('mental_health_model.h5')
            with open('tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            print("Model and tokenizer loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, text_input):
        """Predict mental health risk from text"""
        if self.model is None or self.tokenizer is None:
            raise Exception("Model not loaded. Please train or load a model first.")
        
        # Clean the input text
        cleaned = self.clean_text(text_input)
        
        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=self.max_length, padding='post')
        
        # Predict
        prediction = self.model.predict(padded, verbose=0)[0][0]
        
        return float(prediction)
    
    def interpret_prediction(self, prediction_score):
        """Interpret prediction score into risk level and message"""
        if prediction_score >= 0.6:
            risk_level = "Concern"
            message = "Take care of Your mental health."
        else :
            risk_level = "Normal"
            message = "Take care of your mental wellbeing."
        # else:
        #     risk_level = "Low"
        #     message = "The text shows  no indicators of mental health concerns. Continue taking care of your mental wellbeing."
        
        return {
            'risk_level': risk_level,
            'message': message,
            'prediction_score': round(prediction_score * 100, 2)
        }


# Function to check if model exists
def model_exists():
    """Check if trained model files exist"""
    return os.path.exists('mental_health_model.h5') and os.path.exists('tokenizer.pickle')


# Main execution for training
if __name__ == "__main__":
    mental_health_model = MentalHealthModel()
    
    # Train the model
    csv_path = "mental_health (2).csv"
    mental_health_model.train_model(csv_path)
    
    # Test prediction
    test_text = "I feel really sad and hopeless today"
    prediction = mental_health_model.predict(test_text)
    result = mental_health_model.interpret_prediction(prediction)
    
    print("\n--- Test Prediction ---")
    print(f"Text: {test_text}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Score: {result['prediction_score']}%")
    print(f"Message: {result['message']}")