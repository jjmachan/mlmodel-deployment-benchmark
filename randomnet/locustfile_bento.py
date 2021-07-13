import numpy as np
from locust import HttpUser, TaskSet, task, between

INPUT_WIDTH = 10


class PredictLocust(HttpUser):
    wait_time = between(1, 3)

    @task()
    def predict(self):
        payload = {
            'input': np.random.random((INPUT_WIDTH)).tolist()
        }
        self.client.post('/predict', json=payload)
