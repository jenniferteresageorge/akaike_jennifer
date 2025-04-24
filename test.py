from models import EmailClassifier

# Load the trained model
classifier = EmailClassifier()
classifier.load_model("email_classifier.joblib")

# Test prediction
sample_email = "John Doe needs help resetting my password"
prediction = classifier.predict(sample_email)

print(f"Predicted category: {prediction}")
