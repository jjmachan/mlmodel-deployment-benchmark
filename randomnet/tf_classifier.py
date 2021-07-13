import tensorflow as tf
import numpy as np
import bentoml
from bentoml.adapters import JsonInput
from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact


@bentoml.env(pip_packages=["tensorflow"])
@bentoml.artifacts([TensorflowSavedModelArtifact('model')])
class TFmodelService(bentoml.BentoService):

    @bentoml.api(input=JsonInput(), batch=True, mb_max_latency=200)
    def predict(self, json):
        input_np = np.array([j['input'] for j in json])
        print(input_np, input_np.shape)
        prediction = self.artifacts.model(input_np)
        return prediction
