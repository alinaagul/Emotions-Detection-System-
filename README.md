# Emotion Detection with MobileNet

This repository contains code for training and deploying an emotion detection model using the MobileNet architecture. The model is trained on a dataset containing images of human faces labeled with different emotions.

## Dataset

The dataset used for training and testing the model can be downloaded from the following link:
[Dataset](https://www.dropbox.com/s/nilt43hyl1dx82k/dataset.zip?dl=0)

## Usage

### Training

To train the model, follow these steps:

1. Download the dataset and unzip it.
2. Execute the provided Python script to train the model.

```bash
python train_model.py
```

3. Once the training is complete, the best performing model will be saved as `best_model.h5`.

### Prediction

To make predictions using the trained model, you can use the provided script or integrate the model into your own applications.

```bash
python predict_emotion.py
```

This script prompts you to enter the path to an image containing a human face. It then uses the trained model to predict the emotion expressed in the image.

## Model Architecture

The model architecture is based on MobileNet with custom modifications for the final layers for emotion classification.

## Requirements

- Python 3.x
- Keras
- TensorFlow
- OpenCV
- NumPy
- Matplotlib

## Acknowledgements

- The dataset used for training the model was obtained from [source](insert_source_link).
- The MobileNet architecture was implemented using the Keras library.

## License

This project is licensed under the [MIT License](LICENSE).

