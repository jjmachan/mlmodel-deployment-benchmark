from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, Input


def build_big_model(
    # small input and output in order to avoid latency due to data transfer through the internet during our tests
    input_width=10,
    output_width=10,
    hidden_layers_width=10**3,
    nb_hidden_layers=200,
):
    input_layer = Input((input_width,))
    first_output = Dense(hidden_layers_width)(input_layer)
    output = first_output
    # sharing the weights between all hidden layers in order not to run out of gpu memory
    layer_to_duplicate = Dense(hidden_layers_width)
    for _ in range(nb_hidden_layers):
        # adding first_output residual prevents TensorFlow of using cached computed values to speed up inference
        output = layer_to_duplicate(output + first_output)
    output = Dense(output_width)(output)

    return Model(inputs=input_layer, outputs=output)


if __name__ == '__main__':
    model = build_big_model()
