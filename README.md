# Radiologist Assistant System

## Overview
The Radiologist Assistant System is an AI-powered tool designed to assist radiologists by automating medical image segmentation and prioritizing critical cases. It focuses on brain tumor detection and stroke case classification using deep learning models.

## Features
- **Brain Tumor Segmentation**: Utilizes U-Net for accurate tumor segmentation in MRI scans.
- **Integration with Radiology Workflow**: Supports DICOM format and PACS integration.
- **Cloud Storage Support**: Enables remote access to medical reports.
- **Enhanced Communication System**: Facilitates seamless communication between radiologists and physicians for better diagnosis and treatment coordination.

## Technologies Used
- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Image Processing**: OpenCV
- **Medical Imaging Formats**: DICOM, NIfTI, TIFF
- **Frontend**: Streamlit
- **Backend**: Streamlit
- **Database**: 

## Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/radiologist-assistant.git

# Navigate to the project directory
cd radiologist-assistant

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Start the backend API
uvicorn app.main:app --reload

# Start the frontend
cd frontend
npm install
npm start
```

## Model Training
```bash
python train.py --dataset /path/to/dataset --epochs 50 --batch_size 16
```

## API Endpoints
| Method | Endpoint | Description |
|--------|---------|-------------|
| POST | `/upload` | Uploads MRI/CT images |
| GET | `/analyze/{patient_id}` | Returns AI-generated segmentation results |
| GET | `/report/{patient_id}` | Fetches the AI-generated report |

## Future Development
- Expansion to support segmentation and analysis for all types of scans and body parts.
- Improved real-time segmentation and classification.
- Federated learning for multi-institutional training.
- Integration with a robust radiologist-to-physician communication system.
- **Automated Report Generation**: AI-powered system for generating comprehensive radiology reports.

## Contributors
- Shree Charan
- Yaazhini
- Yaazhini A


