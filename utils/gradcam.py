import tensorflow as tf
import numpy as np
import cv2

def get_gradcam_overlay(sequential_model, image_tensor, conv_layer_name=None):
    """
    Wraps a Sequential model into a Functional model and generates Grad-CAM heatmap.
    """

    # Ensure input is a tf.Tensor
    if isinstance(image_tensor, np.ndarray):
        image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)

    # Wrap Sequential model into a Functional model with new input
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = sequential_model(inputs)
    wrapped_model = tf.keras.Model(inputs=inputs, outputs=x)

    # Automatically find last Conv2D layer if not specified
    if conv_layer_name is None:
        conv_layer_name = None
        for layer in reversed(wrapped_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layer_name = layer.name
                break
        if conv_layer_name is None:
            print("🧠 DEBUG: No Conv2D layers found in model. Here's the layer list:")
            for layer in wrapped_model.layers:
                print(f"- {layer.name}: {type(layer)}")
            raise ValueError("No Conv2D layer found for Grad-CAM.")

    # Build Grad-CAM model
    grad_model = tf.keras.models.Model(
        [wrapped_model.input],
        [wrapped_model.get_layer(conv_layer_name).output, wrapped_model.output]
    )

    # Run Grad-CAM
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    # Safe reduction based on shape
    if len(grads.shape) == 4:
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    elif len(grads.shape) == 2:
        pooled_grads = tf.reduce_mean(grads, axis=0)
    else:
        raise ValueError("Unexpected gradient shape: " + str(grads.shape))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (224, 224))
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    original = tf.squeeze(image_tensor).numpy() * 255
    original = np.uint8(original)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    return overlay
