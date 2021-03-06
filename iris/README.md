## Sklearn - Iris Classifier

This is a simple model build with sklearn. Very fast inference times (~=5ms).

Frameworks used

  1. Flask
  2. Bentoml

## Running the bechmarks

First we start the deployment framework that sets up the APIs for testing. They
we use locust framework to load test the API

### Start deployment server

1. Flask
`gunicorn flask_wsgi:app -b localhost:5000 -w <the num of workers>`

2. Bentoml
`bentoml serve-gunicorn -w <num of workers> iris_bundle`


### Load test with locust

Run the locust server and then go to `localhost:8089` to set the number of users
to mock.
`locust --locustfile locustfile_<framework you want to test>.py`

The results should be available. Play around with these settings to find
something before deciding what works for you.


## My Take

So this is a really light model compared to other deep learning based models so
a simple flask api or fastapi endpoint is more than enough. Also using other
options might bring the performance down too.
