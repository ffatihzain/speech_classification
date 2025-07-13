# 🎤 Animal Sound Classification using CNN

## 📖 Project Overview
This project implements a **Convolutional Neural Network (CNN)** to classify animal sounds imitated by humans into 5 categories. The model is trained to recognize these sound classes through audio signal processing and deep learning techniques.

## 🎧 Sound Classes (Labels)
The model classifies the following animal sounds:

| Animal | Imitated Sound | Frequency Range |
|--------|---------------|----------------|
| 🐄 **Cow** | "moo" | 80-300 Hz |
| 🐱 **Cat** | "meow" | 100-2000 Hz |
| 🐶 **Dog** | "woof" | 100-3000 Hz |
| 🐐 **Goat** | "mbee" | 200-1500 Hz |
| 🐦 **Bird** | "tweet" | 1000-8000 Hz |

## 🏗️ Project Structure
```
speech_classification/
├── vocal.ipynb          # Main notebook with complete implementation
├── README.md            # This file
├── data/                # Dataset directory (created during execution)
│   ├── COW/            # Cow sound samples
│   ├── CAT/            # Cat sound samples
│   ├── DOG/            # Dog sound samples
│   ├── GOAT/           # Goat sound samples
│   └── BIRD/           # Bird sound samples
├── animal_sound_cnn.pth # Trained model weights
└── model_info.pkl      # Model configuration and metadata
```

## 🔧 Technical Implementation

### 📊 Dataset
- **Total Samples**: 75 audio files (15 per animal class)
- **Train/Test Split**: 10 training samples, 5 test samples per class
- **Audio Format**: WAV files, 2 seconds duration, 32kHz sample rate
- **Data Source**: **Real animal sound recordings** with advanced audio augmentation

### 🎵 Real Animal Audio Dataset
This project uses **authentic animal sound recordings** as the foundation, providing superior quality and realism:

#### � **Source Files in `animal_dataset/`:**
- **cow.mp3** - Authentic cow lowing sound
- **Cat.mp3** - Real cat meowing recording
- **dog.mp3** - Genuine dog barking sound
- **goat.mp3** - Natural goat bleating audio
- **Bird.mp3** - Real bird chirping recording

#### 🔧 **Advanced Audio Augmentation:**
1. **Time Stretching**: Changes tempo while preserving pitch (0.8x - 1.2x speed)
2. **Pitch Shifting**: Alters pitch without changing duration (±2 semitones)
3. **Noise Injection**: Adds controlled background noise for robustness
4. **Volume Variation**: Dynamic range modification (0.7x - 1.3x amplitude)
5. **Spectral Filtering**: Low-pass filtering for timbre variation
6. **Dynamic Processing**: Compression/expansion for natural dynamics
7. **Segmentation**: Random segments from original recordings

#### ✅ **Advantages of Real Audio Approach:**
1. **Authenticity**: Uses genuine animal acoustic characteristics
2. **Natural Variations**: Preserves real-world sound properties
3. **High Quality**: Professional-grade source recordings
4. **Better Generalization**: More realistic features for CNN training
5. **Acoustic Fidelity**: Real formants, harmonics, and spectral patterns

### 🎵 Audio Processing Pipeline
1. **Voice Activity Detection (VAD)**: Remove silent segments
2. **Normalization**: Amplitude normalization for consistent input
3. **Padding/Cropping**: Ensure uniform duration (2 seconds)
4. **Feature Extraction**: Extract 40 MFCC coefficients
5. **Standardization**: Normalize MFCC features (mean=0, std=1)

### 🧠 Model Architecture
```python
AnimalSoundCNN(
  (conv): Sequential(
    (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0)
    (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0)
    (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU()
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0)
  )
  (fc): Sequential(
    (0): Linear(in_features=X, out_features=128)
    (1): ReLU()
    (2): Dropout(p=0.5)
    (3): Linear(in_features=128, out_features=5)
  )
)
```

### 🏋️ Training Configuration
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 150
- **Batch Size**: 16
- **Regularization**: Dropout (0.5) and Weight Decay (1e-4)
- **Device**: CUDA if available, otherwise CPU

## 📈 Performance Metrics
The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## 🚀 Features

### 1. **Real-time Prediction**
- Live audio recording and classification
- Confidence scoring for all classes
- User-friendly interface with countdown timer

### 2. **Comprehensive Analysis**
- Waveform visualization (original vs preprocessed)
- MFCC feature visualization
- Spectral analysis with frequency characteristics
- Training metrics visualization

### 3. **Model Persistence**
- Save/load trained model weights
- Model configuration and metadata storage
- Reproducible results

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install librosa sounddevice soundfile
pip install matplotlib seaborn
pip install scikit-learn numpy pandas
```

### Running the Project
1. Clone or download the project
2. Ensure you have the `animal_dataset/` folder with MP3 files:
   - cow.mp3, Cat.mp3, dog.mp3, goat.mp3, Bird.mp3
3. Open `vocal.ipynb` in Jupyter Notebook or VS Code
4. Run all cells sequentially - the dataset will be generated from real audio automatically
5. The model will train on the augmented real animal sound data
6. Test with real-time predictions (record actual animal sounds or imitations)

## 🎯 Usage Instructions

### 1. Data Generation (From Real Audio)
- The notebook automatically processes real animal MP3 files
- Each original recording is augmented to create 15 variations
- Advanced audio processing preserves natural characteristics

### 2. Model Training
- The notebook will automatically preprocess the real audio data
- Train the CNN model on authentic animal sound features
- Monitor training progress through loss and accuracy plots

### 3. Testing & Prediction
- Use the real-time prediction function
- Record a 2-second animal sound (real recordings work best)
- Get confidence scores for all animal classes

## 📊 Results Visualization
The notebook provides comprehensive visualizations:
- **Waveform Analysis**: Original vs preprocessed audio
- **MFCC Features**: Visual representation of extracted features
- **Spectral Analysis**: Frequency characteristics per animal
- **Training Metrics**: Loss and accuracy curves
- **Confusion Matrix**: Model performance breakdown

## 🔬 Technical Details

### Signal Processing
- **Sample Rate**: 32,000 Hz
- **Window Size**: 2 seconds (64,000 samples)
- **MFCC Coefficients**: 40 features
- **Hop Length**: 512 samples
- **Window Function**: Hamming window

### Deep Learning
- **Framework**: PyTorch
- **Input Shape**: (batch_size, 1, 40, time_steps)
- **Output Classes**: 5 (one for each animal)
- **Activation**: ReLU for hidden layers, Softmax for output

## 🚧 Limitations & Future Work

### Current Limitations
- Synthetic data (not real animal recordings)
- Limited to 5 animal classes
- Single acoustic model per animal
- Test predictions require manual recording

### Future Enhancements
- **Real Animal Recordings**: Integrate actual animal sound datasets
- **Larger Dataset**: Generate more samples with greater variation
- **More Animal Classes**: Expand to more animal sounds
- **Advanced Synthesis**: Implement more sophisticated audio synthesis techniques
- **Model Optimization**: Experiment with RNN, Transformer architectures
- **Mobile Deployment**: Create mobile app for real-time classification
- **Noise Robustness**: Improve performance in noisy environments
- **Multi-modal**: Combine audio with visual features
