import os
import sys
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model_builder import build_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_DIR = r"c:\Users\ABC\Desktop\anigrevity\Flower_System\data\flowers\train"
MODEL_SAVE_DIR = r"c:\Users\ABC\Desktop\anigrevity\Flower_System\models"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS_STAGE_1 = 5
EPOCHS_STAGE_2 = 10 
USE_ADVANCED_AUG = False # Set to True later if accuracy < 85% or overfitting is detected

# --- Advanced Augmentation Handlers (Phase 2) ---
try:
    import albumentations as A
    ADVANCED_AUGMENTATIONS = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.3),
    ])
    
    def _albumentations_fn(image):
        # image shapes are [224, 224, 3], scale [0..255]
        img_np = image.numpy().astype(np.uint8)
        aug_img = ADVANCED_AUGMENTATIONS(image=img_np)['image']
        return aug_img.astype(np.float32)
        
    def process_advanced_aug(image, label):
        aug_image = tf.py_function(func=_albumentations_fn, inp=[image], Tout=tf.float32)
        aug_image.set_shape(image.get_shape())
        return aug_image, label
except ImportError:
    logger.warning("Albumentations not found! Run 'pip install albumentations' before enabling USE_ADVANCED_AUG.")
    USE_ADVANCED_AUG = False

def apply_mixup_tf(images, labels, alpha=0.2):
    """
    TF-Native MixUp implementation. Runs extremely fast on GPUs.
    Shuffles the batch and mixes images and labels against themselves.
    """
    batch_size = tf.shape(images)[0]
    
    # Sample from Beta distribution using Gamma distribution
    gamma1 = tf.random.gamma(shape=[batch_size, 1, 1, 1], alpha=alpha)
    gamma2 = tf.random.gamma(shape=[batch_size, 1, 1, 1], alpha=alpha)
    lam = gamma1 / (gamma1 + gamma2)
    
    # Shuffle batch to mix with different images
    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = tf.gather(images, indices)
    mixed_labels = tf.gather(labels, indices)
    
    # Mix
    images = lam * images + (1.0 - lam) * mixed_images
    
    lam_label = tf.reshape(lam, [batch_size, 1])
    labels = lam_label * labels + (1.0 - lam_label) * mixed_labels
    return images, labels


def create_dataset_pipeline(data_dir, img_size, batch_size):
    logger.info("Loading training and validation datasets...")
    
    # 80% for training
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )
    
    # 20% for validation - keep shuffle False for easier evaluation tracking if needed 
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )

    class_names = train_ds.class_names
    logger.info(f"Found {len(class_names)} classes: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    
    # Apply ResNet50 preprocessing
    def preprocess(image, label):
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image, label
        
    # Handle Augmentations
    if USE_ADVANCED_AUG:
        logger.info("Using ADVANCED augmentations (MixUp, CutOut, ColorJitter) - Phase 2.")
        # We need to unbatch, apply albumentations, then batch again for Mixup
        train_ds = train_ds.unbatch().map(process_advanced_aug, num_parallel_calls=AUTOTUNE).batch(batch_size)
        
        # Then Apply ResNet50 preprocess
        train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
        
        # Apply MixUp on the Batched dataset
        train_ds = train_ds.map(lambda x, y: apply_mixup_tf(x, y, alpha=0.2), num_parallel_calls=AUTOTUNE)
    else:
        logger.info("Using BASIC augmentations - Phase 1.")
        train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
        
        # Simple data augmentation for training only
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ])
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    train_ds, val_ds, class_names = create_dataset_pipeline(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    num_classes = len(class_names)
    
    class_names_path = os.path.join(MODEL_SAVE_DIR, 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    logger.info(f"Saved class names to {class_names_path}")

    model, backbone = build_model(num_classes=num_classes, base='resnet50')
    logger.info("Model built. Backbone frozen for Stage 1.")

    logger.info("--- STAGE 1: Training Custom Head ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    stage1_cp = os.path.join(MODEL_SAVE_DIR, 'stage1_best.keras')
    callbacks_s1 = [
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        ModelCheckpoint(stage1_cp, monitor='val_accuracy', save_best_only=True, mode='max'),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2)
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE_1,
        callbacks=callbacks_s1
    )

    logger.info("--- STAGE 2: Fine-Tuning Top Backbone Layers ---")
    backbone.trainable = True
    
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    final_model_path = os.path.join(MODEL_SAVE_DIR, 'flower_model_final.keras')
    legacy_path = os.path.join(MODEL_SAVE_DIR, 'best_model.h5')
    
    callbacks_s2 = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ModelCheckpoint(final_model_path, monitor='val_accuracy', save_best_only=True, mode='max'),
        ModelCheckpoint(legacy_path, monitor='val_accuracy', save_best_only=True, mode='max'), # Save best legacy
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=3)
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE_2,
        callbacks=callbacks_s2
    )
    
    logger.info(f"Training complete! Best final models saved.")
    
    # --- EVALUATION ---
    logger.info("--- EVALUATING BEST MODEL ON VALIDATION SET ---")
    # Restore best weights dynamically, though EarlyStopping restore_best_weights handles this
    
    y_true = []
    y_pred_probs = []
    
    logger.info("Running predictions on validation data...")
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds)
        y_true.extend(labels.numpy())

    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    # Save Confusion Matrix figure
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Validation Confusion Matrix')
    cm_path = os.path.join(MODEL_SAVE_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    logger.info(f"Confusion Matrix saved to {cm_path}")
    print("="*50)

if __name__ == "__main__":
    train()
