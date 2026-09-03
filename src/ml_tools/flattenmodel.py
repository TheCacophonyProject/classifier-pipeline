
# ==========================================
# STEP 1: READ THE WEIGHT ARRAYS NATIVELY IN KERAS 3
# ==========================================
import tensorflow as tf
import numpy as np



# The saved model was written with a newer/older Keras than the one
# installed here, so some layer configs carry kwargs this Keras's layer
# classes don't recognize (e.g. BatchNormalization's old `renorm*` args,
# Dense's newer `quantization_config`). Every plain layer class resolves
# `from_config` to `Operation.from_config` (none of them override it), which
# just does `cls(**config)` -- so patch that one choke point to drop known
# forward/backward-compat keys before reconstructing the layer, instead of
# chasing every affected layer class individually. `custom_objects` can't be
# used for this: it's keyed off `registered_name`, which is `None` in this
# saved model's config, so it's never consulted during deserialization.
from keras.src.ops.operation import Operation

_orig_from_config = Operation.from_config.__func__
_DROP_CONFIG_KEYS = {"renorm", "renorm_clipping", "renorm_momentum", "quantization_config"}


@classmethod
def _patched_from_config(cls, config):
    if isinstance(config, dict):
        config = {k: v for k, v in config.items() if k not in _DROP_CONFIG_KEYS}
    return _orig_from_config(cls, config)


Operation.from_config = _patched_from_config



def flatten_model(k3_model):
    print("Flatting model")
    k3_model.get_layer("channel_aligner").trainable = False
    flat_input = tf.keras.layers.Input(shape=k3_model.input_shape[1:], name="flat_input_image")

    tensor_map = {}
    # Seed the mapping using our fresh standalone input tracer
    for original_inp in k3_model.inputs:
        tensor_map[original_inp] = flat_input

    # Recursive function to unpack inner layers safely 
    def unroll_layers(layers_list):
        for layer in layers_list:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
                
            # If it's a sub-model container, dive deep and unroll its children
            if isinstance(layer, tf.keras.Model) and hasattr(layer, "layers"):
                print(f"-> Unnesting container: '{layer.name}'")
                # `.input`/`.output` on a nested Functional model return its own
                # *structural* definition tensors (from when it was built as a
                # standalone model) -- not the tensors of the node created when
                # it's called as a sub-layer in the outer graph. The real outer
                # connection points live on its inbound node instead.
                outer_node = layer._inbound_nodes[0]
                outer_inbound = outer_node.input_tensors
                if isinstance(outer_inbound, list):
                    outer_inbound = outer_inbound[0]
                outer_output = outer_node.output_tensors
                if isinstance(outer_output, list):
                    outer_output = outer_output[0]

                # Map the sub-model's internal structural input to its current outer tracer
                tensor_map[layer.inputs[0]] = tensor_map[outer_inbound]

                # Recursively unroll the inner layers
                unroll_layers(layer.layers)

                # Link the container's outer output tensor to the terminal internal layer trace
                tensor_map[outer_output] = tensor_map[layer.outputs[0]]
                continue

            # --- PROCESS REGULAR LAYERS ---
            # if layer.name == "channel_aligner":
            #     layer.trainable = False

            # Extract current mapped inputs
            original_inputs = layer.input
            if isinstance(original_inputs, list):
                mapped_inputs = [tensor_map.get(t, t) for t in original_inputs]
            else:
                mapped_inputs = tensor_map.get(original_inputs, original_inputs)

            # Call the layer instance explicitly at the root level!
            # This strips away the sub-model visual nesting properties in Keras 3.
            x = layer(mapped_inputs)
            
            # Save output tracers safely
            if isinstance(layer.output, list):
                for out_t, new_t in zip(layer.output, x):
                    tensor_map[out_t] = new_t
            else:
                tensor_map[layer.output] = x

    # Run the global flattening trace
    unroll_layers(k3_model.layers)

    # 3. Create the final flattened architecture
    # The terminal value of 'x' tracks perfectly back to 'flat_input'
    flat_model = tf.keras.Model(inputs=flat_input, outputs=tensor_map[k3_model.output])
    return flat_model


import sys
def main():
    model_file = sys.argv[1]
    k3_model = tf.keras.models.load_model(model_file, compile=False)
    print("Loaded")
    k3_model.summary()
    flat_model = flatten_model(k3_model)
    print("\nSuccess! Fully unnested and flattened model summary:")
    flat_model.summary()
    flat_model.save("flattened_model.keras")



if __name__ == "__main__":
    main()
