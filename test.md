cd .\Products-Recommender-System\

python -m venv venv

venv\Scripts\activate

pip install flask pandas numpy scikit-learn gunicorn spacy>=3.0.0
python -m spacy download en_core_web_sm

python app.py


testing items
nail, skin, men, lipstick, oil, shampoo, brush
