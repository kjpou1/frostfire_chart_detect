import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    DenseNet121,
    EfficientNetB0,
    MobileNetV3Large,
    MobileNetV3Small,
    ResNet50,
)


def create_model(model_type, input_shape=(224, 224, 3)):
    """
    Create a model based on the specified model type.

    Args:
        model_type (str): The type of model to create. Options: 'mobile', 'custom-1', 'efficientnet', 'resnet', 'densenet'.
        input_shape (tuple): The input shape of the model.

    Returns:
        tf.keras.Model: The created model.
    """
    if model_type == "mobile":
        # Load the MobileNetV3 model
        base_model = MobileNetV3Small(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False
        model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        file_name = "../models/mobilenetv3_classifier.keras"

    if model_type == "mobile_large":
        # Load the MobileNetV3 model
        base_model = MobileNetV3Large(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False
        model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        file_name = "../models/mobilenetv3large_classifier.keras"

    elif model_type == "custom-1":
        model = models.Sequential(
            [
                tf.keras.Input(shape=input_shape),
                layers.Rescaling(1.0 / 255),
                layers.Conv2D(16, (3, 3), activation="relu"),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D(2, 2),
                layers.Flatten(),
                layers.Dense(512, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        file_name = "../models/custom1_classifier.keras"

    elif model_type == "custom-mobile":
        # Load the MobileNetV3 model
        base_model = MobileNetV3Small(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False
        model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.9,
            staircase=True,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        file_name = "../models/custom_mobilenetv3_classifier.keras"

    elif model_type == "efficientnet":
        base_model = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False
        model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        file_name = "../models/efficientnet_classifier.keras"

    elif model_type == "resnet":
        base_model = ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False
        model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        file_name = "../models/resnet_classifier.keras"
    elif model_type == "densenet":
        base_model = DenseNet121(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False
        model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        file_name = "../models/densenet_classifier.keras"
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported types are 'mobile', 'custom-1', 'efficientnet', 'resnet', and 'densenet'."
        )

    return model, file_name
