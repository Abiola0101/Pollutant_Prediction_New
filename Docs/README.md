#  Project Title
Production Deployment of NPRI Machine Learning Project
 
## Team members
 
Abi Afolabi
 
Abiola Bakare  
 
Ruth Olasupo
 
 
## Project Description:
This assignment is to showcase steps to containerize pollution prediction app using Docker desktop. Containerization offers several key advantages for deploying machine learning models in that it allows for:

- **Reproducibility**: Ensures a consistent environment across development, testing, and production phases.
- **Portability**: Allows your application to run reliably on any platform that supports Docker.
- **Isolation**: Keeps dependencies and configurations separate from the underlying host system.
- **Scalability**: Simplifies the process of scaling your application in production environments.
- **Versioning**: Enables version control for the entire application environment, beyond just the code. 



1. Create Dockerfile for ML application

- Dockerfile.mlapp 

2. Create Dockerfile for MLflow

- Dockerfile.mlflow 

3. Create Docker Compose file (.yml)

- docker-compose.yml

The docker compose file is necessary for:
1. **Service Orchestration**: Manages startup order (ensuring MLflow is running before your ML app)
2. **Networking**: Creates an internal network allowing containers to communicate
3. **Configuration Management**: Centralizes environment variables and port mappings
4. **Volume Management**: Simplifies mounting directories for data persistence



4. Employ MLflow to track experiments, record model parameters, and monitor performance metrics, facilitating continuous evaluation and improvement of the deployed model
 
## To implement the project:
 
## Setup Instructions
 
1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
 
2. **Create a virtual environment**:
python3 -m venv venv
 
3. **Activate virual environment**:
For Mac/Linux: source venv/bin/activate
For Windows: .\venv\Scripts\activate
 
4. **Install the required packages**
Ensure you have a requirements.txt file in your repository. If not, create one with the necessary dependencies:
pandas
scikit-learn
category_encoders
numpy
pickle-mixin
mlflow
pyyaml
 
5. **Install dependencies**
pip install -r requirements.txt
 
6. **Build the Docker image**

 
7. 
 
8. 

 
 