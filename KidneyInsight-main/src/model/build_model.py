import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from src.utils.config import IMG_SIZE, LEARNING_RATE


def build_model():
    # Load pretrained ResNet50 (exclude top classifier)
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    # classification head (existing)
    class_out = Dense(1, activation="sigmoid", name="class_output")(x)
    # bounding-box regression head (xmin,ymin,xmax,ymax)
    bbox_out = Dense(4, activation="linear", name="bbox_output")(x)

    model = Model(inputs=base_model.input, outputs=[class_out, bbox_out])

    # Compile model with two losses; bbox loss weight small to not hurt classification
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            "class_output": "binary_crossentropy",
            "bbox_output": "mse",
        },
        loss_weights={"class_output": 1.0, "bbox_output": 0.1},
        metrics={"class_output": "accuracy"}
    )

    return model
