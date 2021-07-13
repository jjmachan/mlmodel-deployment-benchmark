## Tensorflow - A fully connected network

This is a simple Fully Connected Network build with TF. Avg inference times (~=100ms).

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
`bentoml serve-gunicorn -w <num of workers> tf_bundle`


### Load test with locust

Run the locust server and then go to `localhost:8089` to set the number of users
to mock.
`locust --locustfile locustfile_<framework you want to test>.py`

The results should be available. Play around with these settings to find
something before deciding what works for you.


## My Take

This is where Bentoml and other advanced deployment options shine. We are able
to save inference time and increase throughput with an idea called
micro-batching. This batches together the requests in a specific time frame and
hence you don't have to process it individually.
