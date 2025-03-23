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






4. Employ MLflow to track experiments, record model parameters, and monitor performance metrics, facilitating continuous evaluation and improvement of the deployed model
 
## To implement the project:
 
### Setup Instructions
 
1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
 
2. **Create a virtual environment**:
make init-cpu
 
3. **Activate virual environment**:
For Mac/Linux: source venv/bin/activate
For Windows: .\venv\Scripts\activate
 
4. **Install the required packages**

Ensure you have a requirements.txt file in your repository. If not, create one with the necessary dependencies:
- category_encoders==2.6.2
- joblib==1.4.2
- numpy==1.26.0
- pandas==2.2.3
- scikit-learn==1.3.1
- dvc==3.59.0
- PyYAML==6.0.2
- Cython==3.0.12
- Flask==3.1.0
- mlflow==2.20.2
 
5. **Install dependencies**
- pip install -r requirements.txt
 
6. **Build the Docker image**
- Create Dockerfile for ML application
-- Dockerfile.mlapp 

- Create Dockerfile for MLflow
-- Dockerfile.mlflow 

- Create Docker Compose file (.yml)
-- docker-compose.yml

The docker compose file is necessary for:

i)   **Service Orchestration**: Manages startup order (ensuring MLflow is running before your ML app)

ii)  **Networking**: Creates an internal network allowing containers to communicate

iii) **Configuration Management**: Centralizes environment variables and port mappings

iv)  **Volume Management**: Simplifies mounting directories for data persistence

 
7. Configure training script to connect to the MLflow tracking server
**Get the tracking URI from environment variable or use default**
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:9000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
 
8. Build and run contianers

**run the code below:**

docker-compose up --build

9. Publish Docker Images and view them on Docker desktop
- ML Application API: http://localhost:(your-api-port)
- MLflow UI: http://localhost:(your-mlflow-port)

 
 