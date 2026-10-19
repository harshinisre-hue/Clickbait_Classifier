# Clickbait Classifier

A machine learning web application that distinguishes between Legitimate News headlines and Clickbait using NLP (Natural Language Processing) and a Support Vector Machine.

*User can input a headline in the streamlit app, and the classifier will label it as either Legitimatee News or Clickbait.

<img width="883" height="422" alt="image" src="https://github.com/user-attachments/assets/546d9868-a80d-4c70-86bd-5e53c438a795" />

## Structure
1. **Text Cleaner (preprocessor.py)**
- Takes a news headline from dataset and removes punctuation, lowercase, and strips out stop words. (Tokenization)

2. **AI Classifier (model.py)**
- Trains a Support Vector Machine (SVM) on over dataset. (96% Accuracy on Tests)
- Clickbait Dataset from Kaggle contains over 32,000 headlines. (80% for training, 20% for testing)

3. **Performance Graph (confusion_matrix.png)**
- Generates a confusion matrix showing exactly how many headlines the model got right and wrong.

4. **Web App (app.py)**
- Streamlit page where you can type any headline into a box and recieve a success/error banner (Legitimate/Clickbait)
