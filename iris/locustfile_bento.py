# flask_test.py
from locust import HttpUser, between, task


class WebsiteUser(HttpUser):
    # wait_time = between(1, 3)

    @task
    def ping_index(self):
        self.client.post("/predict", json=[1, 2, 2, 3])
