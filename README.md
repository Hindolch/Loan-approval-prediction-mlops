# Loan Approval Prediction System

## Project Overview
This project implements an automated loan approval prediction system using modern MLOps practices. The system uses machine learning to predict loan approval decisions based on applicant information such as Income, credit histoy, education level etc, with a complete pipeline from data processing to deployment.


### Key Components
- **Pipeline Orchestration**: Prefect
- **Experiment Tracking**: MLflow
- **Containerization**: Docker
- **Deployment**: Render
- **Model Serving**: The application exposes an intuitive user interface and REST API for predictions using **Streamlit** and **FastAPI**.

### Machine Learning Model: RandomForest Classifier
The loan approval prediction system is built using a **RandomForest Classifier**, a robust ensemble learning method known for its accuracy and interpretability. The model is trained to classify loan applications as approved or denied based on applicant features.

#### Evaluation Metrics:
- **Accuracy**: Measures the proportion of correct predictions among total predictions.  
  - Ensures the model performs well across all classes.

- **Precision Score**: Evaluates the correctness of positive predictions by assessing the ratio of true positives to the total predicted positives.  
  - Ensures the model minimizes false positives, which is critical for loan approval scenarios.


## Features
- End-to-end ML pipeline orchestration
- Automated model training and validation
- Real-time prediction API
- Experiment tracking and model versioning
- Containerized deployment
- CI/CD pipeline with GitHub Actions

## Prerequisites
- Python 3.9+
- Docker
- Git

## Installation & Setup

1. Clone the repository
```bash
git clone https://github.com/Hindolch/Loan-approval-prediction-mlops.git
cd Loan-approval-prediction-mlops
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Running Locally

1. Start the Prefect server:
```bash
prefect server start
```

2. Run the training pipeline:
```bash
python3 run_pipeline.py
```

3. Start the prediction service:
```bash
streamlit run demo-app.py
```

### Using Docker

1. Build the Docker image:
```bash
docker build -t loan-mlops .
```

2. Run the container:
```bash
docker run -p 8501:8501 loan-mlops
```


## Model Training and Tracking

The project uses MLflow to track experiments and model versions. You can access the MLflow UI by running:
```bash
mlflow ui
```

Navigate to `http://localhost:5000` to view experiment results and model metrics.

## Deployment

The model is deployed on Render and can be accessed at:
[https://loan-approval-prediction-docker-image.onrender.com](https://loan-approval-prediction-docker-image.onrender.com)

## CI/CD Pipeline

The project includes a GitHub Actions workflow that:
1. Runs tests
2. Builds Docker image
3. Pushes to Docker Hub
4. Deploys to Render

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request


![Screenshot from 2024-12-19 22-48-55](https://github.com/user-attachments/assets/014f0136-28d9-458b-8163-9d829dd102d9)

![Screenshot from 2024-12-19 22-50-08](https://github.com/user-attachments/assets/a7a0c09c-2e81-424f-926d-e03e2ff31323)

![Screenshot from 2024-12-20 20-39-54](https://github.com/user-attachments/assets/ae600771-881b-4cf0-b7aa-7de187c8856c)

![Screenshot from 2024-12-20 14-46-05](https://github.com/user-attachments/assets/a79a6f41-9267-4931-aa6d-9f7ba5b6ba78)

![Screenshot from 2024-12-20 14-46-45](https://github.com/user-attachments/assets/4874adfb-2f7a-4980-bdab-a387b39eaff2)

![Screenshot from 2024-12-20 17-46-06](https://github.com/user-attachments/assets/0c7c619d-44b2-4578-8998-838297b75cd7)



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
- **Author**: [Hindol R. Choudhury]
- **Linkedin**: [Hindol Choudhury](https://www.linkedin.com/in/hindol-choudhury-5ab5a5271/)
- **GitHub**: [@Hindolch](https://github.com/Hindolch)

