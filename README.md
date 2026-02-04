# LSTM Gesture Recognition

A deep learning project that uses Long Short-Term Memory (LSTM) networks to recognize and classify hand gestures into one of five gesture commands for smart device control.

## Project Overview

This project implements a gesture recognition system capable of understanding and categorizing hand gestures in real-time. The model can identify 5 different gestures that serve as intuitive commands for controlling smart devices, TVs, or other interactive systems without the need for physical remotes or touch interfaces.

## Problem Statement

In an era of smart homes and touchless interfaces, gesture-based control systems offer a natural and hygienic way to interact with devices. This project aims to build a robust deep learning model that can accurately recognize hand gestures captured through video sequences and translate them into device commands.

## Model Architecture

The project leverages **LSTM (Long Short-Term Memory)** neural networks, a type of Recurrent Neural Network (RNN) particularly well-suited for:
- Sequential data processing
- Temporal pattern recognition
- Learning long-term dependencies in video frames

### Why LSTM?

LSTMs are ideal for gesture recognition because:
- **Temporal Understanding**: Gestures are sequences of movements over time
- **Context Retention**: The model can remember earlier frames while processing later ones
- **Pattern Detection**: Capable of identifying complex motion patterns

## Gesture Classes

The model recognizes 5 distinct gesture commands:
1. **Thumbs Up** - Approval/Select/Increase
2. **Thumbs Down** - Disapproval/Deselect/Decrease
3. **Left Swipe** - Navigate Left/Previous
4. **Right Swipe** - Navigate Right/Next
5. **Stop** - Stop/Pause/Standby

## Technical Implementation

### Dataset
- Video sequences of hand gestures
- Multiple samples per gesture class
- Preprocessed frames for model input

### Preprocessing Pipeline
- Frame extraction from video sequences
- Image normalization and resizing
- Temporal sequence creation
- Data augmentation techniques

### Model Training
- Training-validation split for model evaluation
- Categorical cross-entropy loss function
- Adam optimizer for gradient descent
- Metrics: Accuracy, precision, recall, F1-score

## Repository Structure

```
LSTM-Gesture-Recognition/
│
├── Mohini_Aggarwal_Gesture_Recognition.ipynb  # Main notebook with implementation
├── Gesture_Recognition_Results.docx            # Detailed results and analysis
└── README.md                                    # Project documentation
```

## Getting Started

### Prerequisites

```bash
Python 3.7+
TensorFlow / Keras
NumPy
Pandas
OpenCV
Matplotlib
Scikit-learn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mohinia09/LSTM-Gesture-Recognition.git
cd LSTM-Gesture-Recognition
```

2. Install required dependencies:
```bash
pip install tensorflow numpy pandas opencv-python matplotlib scikit-learn
```

3. Open the Jupyter notebook:
```bash
jupyter notebook Mohini_Aggarwal_Gesture_Recognition.ipynb
```

## Results

Detailed experimental results, model performance metrics, and analysis can be found in the `Gesture_Recognition_Results.docx` file. The model achieves competitive accuracy in recognizing the 5 gesture classes across various test scenarios.

## Use Cases

- **Smart TV Control**: Change channels, adjust volume, play/pause
- **Smart Home Automation**: Control lights, thermostats, appliances
- **Accessibility**: Hands-free device control for users with limited mobility
- **Gaming Interfaces**: Gesture-based game controls
- **Presentation Tools**: Navigate slides without physical controllers
- **Automotive Interfaces**: In-car entertainment and climate control

## Future Enhancements

- [ ] Expand gesture vocabulary to 10+ gestures
- [ ] Real-time gesture recognition with webcam integration
- [ ] Model optimization for edge devices (Raspberry Pi, mobile)
- [ ] Multi-hand gesture support
- [ ] 3D gesture recognition using depth sensors
- [ ] Transfer learning with pre-trained models
- [ ] Integration with IoT device APIs

## Technologies Used

- **Deep Learning Framework**: TensorFlow/Keras
- **Neural Network**: LSTM (Long Short-Term Memory)
- **Data Processing**: NumPy, Pandas
- **Computer Vision**: OpenCV
- **Visualization**: Matplotlib, Seaborn
- **Development Environment**: Jupyter Notebook

## Project Workflow

1. **Data Collection**: Gather video sequences of gestures
2. **Preprocessing**: Extract frames, normalize, and prepare sequences
3. **Model Design**: Build LSTM architecture with appropriate layers
4. **Training**: Train model on prepared dataset with validation
5. **Evaluation**: Test model performance on unseen data
6. **Optimization**: Fine-tune hyperparameters for better accuracy
7. **Deployment**: Prepare model for real-world application

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.

## Author

**Mohini Aggarwal**
- GitHub: [@mohinia09](https://github.com/mohinia09)

## License

This project is available for educational and research purposes.


---

 If you found this project helpful, please consider giving it a star!
