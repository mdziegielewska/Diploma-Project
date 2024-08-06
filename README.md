# Diploma-Project

## Analysis of Movement Measurement in ICSI Procedure using Machine Learning

### Introduction

This project aims to investigate the feasibility of measuring the movement of cellular fluids and needle content during the Intracytoplasmic Sperm Injection (ICSI) procedure by analyzing video recordings. The goal is to develop a tool capable of measuring motion and events within the needle containing sperm, medium, and the egg cell.


### Project Structure

- Demo-App: Houses the Flask application showcasing the project's outcomes.
- Event-Boundary-Detection: Contains code, models, and results related to scene detection and event boundary identification using SceneDetect and TransNet.
- Semantic-Segmentation: Encompasses training data, models, and outputs for segmenting relevant elements within ICSI procedure videos.


### Installation

To set up the project environment, run the following command in your terminal:
```
pip install -r requirements.txt
```

### Future Work

- **Improve models**: Explore advanced architectures (transformers, attention), optimize hyperparameters.
- **Enhance data**: Expand dataset, apply augmentation, refine labels.
- **Develop techniques**: Focus on trajectory analysis, event detection, anomaly detection.
- **Broaden impact**: Seek biomedical applications, real-time processing, and model interpretability.
