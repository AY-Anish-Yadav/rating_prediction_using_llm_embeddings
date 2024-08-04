import joblib
from angle_emb import AnglE
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, request, jsonify

app = Flask(__name__)

def load_model():
    model = joblib.load('D:/projects_llm/review_prediction_using_llm_embeddings/utils/random_forest_regressor_best_model.joblib')
    model_embed = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
    return model,model_embed

def preprocessing(review):
    pass

@app.route('/')
def home():
    return '''
        <form action="/predict" method="post">
            <label for="review">Review:</label><br>
            <textarea id="review" name="review" rows="4" cols="50"></textarea><br><br>
            <input type="submit" value="Submit">
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    # Preprocess the review if necessary
    doc_vecs = model_embed.encode(review, normalize_embedding=True)
    print(doc_vecs)
    prediction = model.predict(doc_vecs)
    return jsonify({'rating': prediction[0]})

if __name__ == '__main__':
    model,model_embed=load_model()
    app.run(debug=True)
