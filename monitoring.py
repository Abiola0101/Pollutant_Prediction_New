import sys
print("Python Path:", sys.path)  # Debug statement to check import paths

from prometheus_client import Gauge, start_http_server, Histogram, Counter

# Step 1: Define TrainingMonitor class
class TrainingMonitor:
    def __init__(self, port=8000):
        """
        Initialize the training monitor and start the HTTP server for metrics.
        
        Args:
            port (int): Port on which the Prometheus HTTP server will run.
        """
        self.port = port
        self.start_server()

    def start_server(self):
        """
        Start the Prometheus HTTP server on the specified port.
        """
        try:
            start_http_server(self.port)
            print(f"Prometheus server started on port {self.port}")
        except Exception as e:
            print(f"Error starting Prometheus server: {e}")

    def log_metric(self, gauge, value):
        """
        Log a specific metric to Prometheus.
        
        Args:
            gauge (Gauge): The Prometheus gauge object.
            value (float): The value to set for the gauge.
        """
        try:
            gauge.set(value)
            print(f"Logged metric: {gauge} with value {value}")
        except Exception as e:
            print(f"Error logging metric: {e}")

# Step 2: Define RegressionMonitor class
class RegressionMonitor(TrainingMonitor):
    def __init__(self, port=8002):
        super().__init__(port)
        
        # Regression-specific metrics
        self.mse = Gauge('regression_mean_squared_error', 'Mean Squared Error')
        self.rmse = Gauge('regression_root_mean_squared_error', 'Root Mean Squared Error')
        self.mae = Gauge('regression_mean_absolute_error', 'Mean Absolute Error')
        self.r_squared = Gauge('regression_r_squared', 'R-squared coefficient')
        
        # Feature importance tracking (top 5 features)
        self.feature_importance = Gauge('feature_importance', 'Feature importance value', ['feature_name'])
        
    def record_metrics(self, mse=None, rmse=None, mae=None, r_squared=None, feature_importance=None):
        """Record regression metrics"""
        if mse is not None:
            self.mse.set(mse)
        if rmse is not None:
            self.rmse.set(rmse)
        if mae is not None:
            self.mae.set(mae)
        if r_squared is not None:
            self.r_squared.set(r_squared)
            
        # Update feature importance for top features
        if feature_importance is not None:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature_name, importance in sorted_features:
                self.feature_importance.labels(feature_name=feature_name).set(importance)

# Step 3: Run an example instance if this script is executed
if __name__ == "__main__":
    monitor = RegressionMonitor(port=8002)
    print("Monitoring started for regression metrics.")
