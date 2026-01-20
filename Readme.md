🫁 Lung Disease Classification – Mobile App  
  
📌 Problem

Classify lung diseases from audio recordings of respiratory sounds using AI to assist in early diagnosis.

🔧 Methodology

Collected respiratory sound datasets

Extracted 146 audio features using librosa & scikit-learn

Built an ensemble model (MLP + Random Forest)

Deployed models on Hugging Face for real-time inference

Developed a Kotlin Android App with backend integration  

launched the app on play store 

Focus on educational and research applications

📊 Results

High classification accuracy across multiple lung disease categories

Real-time prediction and feature extraction

Android app availabe on Google Play Store

Link: https://play.google.com/store/apps/details?id=com.gs.safebreath

🚀 How to Run  
pip install -r requirements.txt  
python predict.py --audio sample_lung.wav
