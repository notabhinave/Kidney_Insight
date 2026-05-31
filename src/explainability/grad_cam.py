import cv2
import numpy as np
import tensorflow as tf

# We removed: from tensorflow.keras.models import Model (This fixes the warning)

def make_gradcam_heatmap(image, model, last_conv_layer_name="conv5_block3_out"):
    # Create a sub-model that outputs the last conv layer + the final predictions
    # We use "tf.keras.models.Model" directly here
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # 1. Forward pass: Run the image through the model
        # Keras models can expect a list/dict of inputs; wrap single tensor if needed
        inp = image
        if not isinstance(image, (list, tuple, dict)):
            inp = [image]
        conv_outputs, predictions = grad_model(inp)

        # 2. Handle the prediction output (fix for List vs Tensor issue)
        if isinstance(predictions, list):
            predictions = predictions[0]

        # 3. Define the "Loss" we want to explain
        if predictions.shape[-1] == 1:
            # Binary classification (Sigmoid)
            loss = predictions[0]
        else:
            # Multi-class -> look at the highest score
            score_index = tf.argmax(predictions[0])
            loss = predictions[:, score_index]

    # 4. Calculate gradients of the Loss with respect to the Conv Layer output
    grads = tape.gradient(loss, conv_outputs)

    # 5. Global Average Pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 6. Multiply feature maps by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 7. Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def overlay_heatmap(image, heatmap, alpha=0.4):
    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Rescale heatmap to 0-255 RGB
    heatmap = np.uint8(255 * heatmap)
    
    # Colorize
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)