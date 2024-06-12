# [American Sign Language Recognition Model](https://american-sign-language-detection-4fcwdnqtjszseqw6ydrtnb.streamlit.app/)

## Overview
This project focuses on building a sign language recognition model using deep learning techniques. The model is trained to classify hand gestures representing different letters or words in sign language.

## Dataset
The dataset used for training and testing the model is the American Sign Language (ASL) dataset, consisting of images of hand gestures representing letters from A to Z.
(https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

## Model Architecture
The sign language recognition model is built using a convolutional neural network (CNN) architecture, specifically designed to analyze spatial patterns in images. The architecture comprises multiple convolutional layers followed by pooling layers for feature extraction and downsampling. The final layers consist of fully connected layers and softmax activation for classification.

## Training Details
- **Dataset Split**: 80% training, 10% validation, 10% testing
- **Optimizer**: Adam optimizer
- **Loss Function**: Categorical cross-entropy
- **Metrics**: Accuracy
- **Training Epochs**: 50
- **Batch Size**: 32

## Performance Evaluation
- **Test Accuracy**: 92.5%
- **Traning Accuracy and Loss**:
- ![image](https://github.com/Apurv090405/american-sign-language-detection/assets/120238040/24dfa610-a276-4298-bb3c-178bcee3f686)


## Model Deployment
The trained model is deployed as a web application using Flask, allowing users to upload images of hand gestures and receive real-time predictions of the corresponding sign language letters or words.

## Future Improvements
- Incorporating real-time video processing for sign language recognition.
- Enhancing model robustness to variations in lighting conditions and hand orientations.
- Expanding the dataset to include more diverse hand gestures and sign language variations.

## Google Drive Link
https://drive.google.com/drive/folders/15rCGeyGzloFVrhWqnKCmCA3VaYe58TMh?usp=sharing

## Author
Apurv Chudasama

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
