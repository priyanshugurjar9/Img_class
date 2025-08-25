Image Classification (CNN)


Project Overview

This project builds a Convolutional Neural Network (CNN) to classify images of dogs and cats using the Kaggle Dogs vs Cats dataset (link - /kaggle/input/dogs-vs-cats).

The model is trained from scratch with TensorFlow/Keras and achieves reliable accuracy in distinguishing between the two classes.

⸻

📂 Dataset
	•	Dataset Source: Dogs vs Cats Dataset – Kaggle
 
	•	Contents:
 
	•	Training images: 25,000 (12,500 dogs / 12,500 cats)
 
	•	Testing images: 12,500 unlabeled images


⸻

⚙️ Tech Stack

	•	Programming Language: Python
 
	•	Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn
 
	•	Tools: Jupyter Notebook / Kaggle Notebooks, GitHub
 

⸻


🧠 Model Architecture

	•	Input Layer: 256x256x3 RGB images
 
	•	3 Convolutional Layers (Conv2D + ReLU + MaxPooling)
 
	•	Flatten Layer
 
	•	Fully Connected Dense Layers with Dropout
 
	•	Output Layer (Sigmoid for binary classification)
 

⸻

🚀 Project Workflow

	1.	Data Preprocessing
 
	•	Image resizing (256x256)
 
	•	Normalization & batch loading using image_dataset_from_directory()
 
	2.	Model Training
 
	•	CNN built with TensorFlow/Keras
 
	•	Optimizer: Adam
 
	•	Loss Function: Binary Crossentropy
 
	•	Metrics: Accuracy
 
	3.	Model Evaluation
 
	•	Training vs Validation accuracy plots
 
	•	Test set evaluation
 
	4.	Prediction
 
	•	Single image prediction (dog or cat)
 

⸻

📊 Results

	•	Achieved ~85–90% accuracy on validation set.
 
	•	CNN generalizes well to unseen test images.
 

⸻

🔮 Future Improvements

	•	Use Data Augmentation for improved generalization.
 
	•	Implement Transfer Learning (e.g., VGG16, ResNet50).
 
	•	Deploy the model with Streamlit/Flask + Docker.

