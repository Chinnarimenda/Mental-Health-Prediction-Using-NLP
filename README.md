# 🧠 Mental-Health-Prediction-Using-NLP

This project uses Natural Language Processing (NLP) and Deep Learning (LSTM) to predict whether a text indicates signs of a mental disorder or distress.

The model analyzes user-generated text (such as posts, comments, or messages) and classifies it into categories like Normal, Distressed, or At-risk, helping support early detection of mental health conditions.

# 🎯 Objective

To develop a text classification model that can identify potential mental distress based on linguistic patterns in written text, supporting early mental-health awareness and intervention.

# 📊 Dataset Overview

The dataset consists of text samples labeled according to mental health condition or emotional state.

COLUMNS:

TEXT :- The user-generated text input

LABEL :- The target class (e.g., Normal, Distressed, Depression, Anxiety, etc.) IN 0 and 1

# ⚙️ Project Workflow

1. Text Preprocessing

    Cleaning text (removing punctuation, stopwords, special characters)
   
    Tokenization and padding
   
    Label encoding of target classes
   
    Model Architecture

3. Embedding Layer for word representation

    SpatialDropout1D to prevent overfitting
   
    LSTM (Long Short-Term Memory) layers for sequence learning
   
    Dense output layer with activation (sigmoid or softmax)
   
5. Model Training

    Optimizer: Adam
   
    Loss: Binary or Categorical Cross-Entropy (depending on number of classes)
   
    Metrics: Accuracy, Precision, Recall, F1-Score
   
7. Evaluation

    Tested on unseen data
   
    Achieved strong accuracy and stable validation performance
   
    Visualized loss/accuracy curves and confusion matrix

 #  🧠 Key Insights

 LSTM effectively captures the emotional tone and context in user text.
 
 Proper preprocessing (tokenization, padding, stemming) greatly improves performance.
 
 Balanced data and word embeddings lead to higher model generalization.

# 🛠️ Tech Stack

 Language: Python

 Libraries: TensorFlow, Keras, NumPy, Pandas, Scikit-learn, NLTK, Matplotlib
 
 Tools: Jupyter Notebook / VS Code
 
 Deployment (Optional): Streamlit or Flask web app for live text prediction

# ✅ Results

 Model accuracy: ~87–90% on test data
 
 Robust detection of distress vs. normal text
 
 Deployed successfully as an interactive web app (Streamlit)

# 🚀 Future Improvements

 Expand dataset for multi-class classification (depression, anxiety, stress, etc.)

 Incorporate transformer-based models (BERT / RoBERTa) for better contextual understanding

 Deploy to cloud (e.g., Hugging Face Spaces / Render / Streamlit Cloud)


















