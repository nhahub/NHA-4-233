# NHA-4-233
Auto generated repo 233
-----------------------------------------
AI Pharaoh — Smart Egyptian Tourist Guide
-----------------------------------------
AI Pharaoh is a graduation project developed for the DEPI (Digital Egypt Pioneers Initiative) — Microsoft Machine Learning Engineer track. It's an AI-powered tourist guide that helps visitors identify Egyptian monuments and understand hieroglyphic inscriptions using computer vision.
-----------------------------------------
🔗 Live Demo:
Try it here NOTE:OPERATE IT AT DARK MODE: https://ai-pharaoh.streamlit.app/


✨ Features

🏛️ Monument Recognition

Upload a photo of an Egyptian monument or artifact and instantly get:


The artifact's name
Historical overview
Dynasty and time period
Location
An audio guide (text-to-speech)
A link to the location on Google Maps
An interactive VR-style exploration mode


📜 Hieroglyph Translator

Upload a photo containing hieroglyphic inscriptions, crop a single symbol, and get:


The symbol's Gardiner sign classification code
Its meaning and, where available, its phonetic value
The top 5 most likely matches with confidence scores



🧠 How It Works

Monument Recognition Pipeline

Image → Resize (224×224) → EfficientNet preprocessing → CNN Classifier → Result + JSON lookup

Hieroglyph Translation Pipeline

Image → User crops a single symbol → Resize (299×299) → InceptionV3 feature extraction
      → SVM/Logistic Regression classifier → Gardiner code → Meaning lookup
---------------------------------

🛠️ Tech Stack

ComponentTechnologyUI / Web appStreamlitMonument classifierTensorFlow / Keras (EfficientNet)Hieroglyph feature extractorInceptionV3 (ImageNet weights)Hieroglyph classifierscikit-learn (Logistic Regression, GridSearchCV-tuned)Cropping toolstreamlit-cropperHistorical dataCustom JSON knowledge baseHieroglyph meaningsGardiner Sign List reference (custom JSON)


📂 Repository Structure

├── app2.py                     # Main Streamlit application
├── featureExtractor.py         # InceptionV3-based feature extractor for hieroglyphs
├── requirements.txt            # Python dependencies
├── artifact_classifier.keras   # Trained monument recognition model
├── labels.json                 # Monument class labels
├── artifacts.json              # Historical data for each monument
├── real_svm_tuned.pkl          # Trained hieroglyph classifier
├── gardiner_signs.json         # Gardiner code → meaning dictionary
└── README.md


🚀 Running Locally

bashgit clone https://github.com/nhahub/NHA-4-233.git
cd NHA-4-233
pip install -r requirements.txt
streamlit run app.py


⚠️ TensorFlow requires Python 3.9–3.12. If you're on a newer Python version and installation fails, use a virtual environment with a compatible version.




📊 Datasets


Hieroglyph classifier trained on the GlyphDataset , labeled using Gardiner's sign list classification system — 161 classes.
Monument classifier trained on a custom-curated dataset of Egyptian monuments and artifacts.




👥 Team
Mostafa wael 
Youssef emad 
Ahmed wael
Mohammed ashraf


