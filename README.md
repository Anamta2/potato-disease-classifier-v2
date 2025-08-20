# Potato Disease Classification

A machine learning project for classifying potato plant diseases using deep learning techniques. This system can identify healthy potato plants and detect common diseases including Early Blight and Late Blight with high accuracy.

## 🎯 Project Overview

This project implements a convolutional neural network (CNN) to classify potato plant diseases from leaf images. The model achieves over 95% accuracy in distinguishing between healthy plants and those affected by Early Blight or Late Blight diseases.

## 🔍 Disease Classes

The model classifies potato plants into three categories:
- **Healthy**: Normal, disease-free potato plants
- **Early Blight**: Caused by *Alternaria solani*
- **Late Blight**: Caused by *Phytophthora infestans*

## 📊 Model Performance

- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~97%
- **Model Confidence**: Up to 90% for disease predictions
- **Training Loss**: Converged to ~0.2
- **Validation Loss**: Converged to ~0.1

## 🏗️ Project Structure

```
potato-disease-classification/
│
├── potato_diseases_clean/
│   ├── api/
│   │   ├── main.py              # FastAPI application
│   │   └── requirements.txt     # API dependencies
│   │
│   └── training_model/
│       └── training.ipynb       # Model training notebook
│
└── README.md
```

## 🚀 Features

- **Deep Learning Model**: CNN architecture optimized for plant disease detection
- **REST API**: FastAPI-based web service for real-time predictions
- **High Accuracy**: Achieves >95% accuracy on validation data
- **Multi-class Classification**: Distinguishes between 3 different plant conditions
- **Confidence Scoring**: Provides prediction confidence for each classification

## 🛠️ Technologies Used

- **Python**: Primary programming language
- **TensorFlow/Keras**: Deep learning framework
- **FastAPI**: Web framework for API development
- **Jupyter Notebook**: Model development and training
- **NumPy**: Numerical computations
- **PIL/OpenCV**: Image processing
- **Uvicorn**: ASGI server for API deployment

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/potato-disease-classification.git
   cd potato-disease-classification
   ```

2. **Install API dependencies**
   ```bash
   cd potato_diseases_clean/api
   pip install -r requirements.txt
   ```

3. **Install training dependencies** (for Jupyter notebook)
   ```bash
   pip install tensorflow jupyter matplotlib numpy pandas
   ```

## 🔧 Usage

### Training the Model

1. Open the training notebook:
   ```bash
   cd potato_diseases_clean/training_model
   jupyter notebook training.ipynb
   ```

2. Run all cells to train the model from scratch
3. The trained model will be saved for use with the API

### Running the API

1. Start the FastAPI server:
   ```bash
   cd potato_diseases_clean/api
   python main.py
   ```

2. The API will be available at `http://localhost:8000`

3. Access the interactive API documentation at `http://localhost:8000/docs`

### Making Predictions

Send a POST request to the `/predict` endpoint with an image file:

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("potato_leaf.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Example Response:**
```json
{
  "class": "late_blight",
  "confidence": 0.9033572673797607,
  "all_predictions": {
    "Early blight": 0.09663786739110947,
    "late blight": 0.9033572673797607,
    "healthy": 4.856416580878431e-06
  }
}
```

## 📈 Model Training Details

The model training process includes:
- **Data Preprocessing**: Image normalization and augmentation
- **Architecture**: Convolutional Neural Network with multiple layers
- **Optimization**: Adam optimizer with learning rate scheduling
- **Validation**: 20% of data reserved for validation
- **Early Stopping**: Prevents overfitting during training

## 🎯 Use Cases

- **Agricultural Monitoring**: Early detection of potato diseases
- **Crop Management**: Helping farmers make informed decisions
- **Research**: Plant pathology studies and analysis
- **Mobile Applications**: Integration into farming apps
- **Educational Tools**: Teaching plant disease identification

## 🔮 Future Improvements

- [ ] Add more disease classes
- [ ] Implement real-time video processing
- [ ] Mobile app development
- [ ] Integration with IoT sensors
- [ ] Batch processing capabilities
- [ ] Model quantization for edge deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- Dataset providers for potato plant images
- TensorFlow team for the deep learning framework
- FastAPI developers for the excellent web framework
- Agricultural research community for disease classification insights

---

⭐ **If you found this project helpful, please give it a star!** ⭐
