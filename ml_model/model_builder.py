import tensorflow as tf
from tensorflow.keras import layers, Model

def build_model(num_classes=102, base='resnet50'):
    """
    Builds the CNN model with a pre-trained backbone and custom head.
    """
    if base == 'resnet50':
        backbone = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
    elif base == 'efficientnetb3':
        backbone = tf.keras.applications.EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=(300, 300, 3)
        )
    else:
        raise ValueError("Unsupported base model")

    # Freeze backbone
    backbone.trainable = False

    # Custom classification head
    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=backbone.input, outputs=outputs)
    return model, backbone
