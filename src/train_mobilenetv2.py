"""
MobileNetV2 comparison model for:
CNN-Based Classification of Images Containing One Person or Multiple People

Uses the existing local dataset; it does NOT download COCO.

Expected structure:
dataset/
  train/
    multiple_people/
    single_person/
  validation/
    multiple_people/
    single_person/
  test/
    multiple_people/
    single_person/

Run from the project root:
    python src/train_mobilenetv2.py

Optional prediction after training:
    python src/train_mobilenetv2.py --predict "C:\\path\\to\\photo.jpg"

Notes:
- The model predicts the whole image as single_person or multiple_people.
- By default, it uses ImageNet-pretrained MobileNetV2 weights (transfer learning).
- If downloading pretrained weights fails due to a network/SSL issue, set PRETRAINED = False.
"""

from pathlib import Path
import argparse
import json
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)

# ---------------- SETTINGS ----------------
ROOT = Path(__file__).resolve().parents[1]  # project root when this file is in src/
DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "outputs" / "mobilenetv2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42
PRETRAINED = True
CLASS_NAMES = ["multiple_people", "single_person"]  # fixed order; label 0 then 1
AUTOTUNE = tf.data.AUTOTUNE

keras.utils.set_random_seed(SEED)


def check_dataset():
    """Validate expected split/class folders and count supported image files."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    counts = {}
    for split in ("train", "validation", "test"):
        for class_name in CLASS_NAMES:
            folder = DATA_DIR / split / class_name
            if not folder.is_dir():
                raise FileNotFoundError(
                    f"Required folder is missing: {folder}\n"
                    "Check that your dataset has train/validation/test, each with "
                    "multiple_people/ and single_person/ folders."
                )
            count = sum(
                1 for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in extensions
            )
            counts[f"{split}/{class_name}"] = count
            if count == 0:
                raise ValueError(
                    f"No supported images found in {folder}. "
                    "Images must be JPG, JPEG, PNG, BMP, or WEBP."
                )

    print("\nDataset image counts:")
    for name, count in counts.items():
        print(f"  {name}: {count}")
    return counts


def load_datasets():
    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "validation",
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return (
        train_ds.prefetch(AUTOTUNE),
        val_ds.prefetch(AUTOTUNE),
        test_ds.prefetch(AUTOTUNE),
    )


def build_model(pretrained=True):
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="image")

    # MobileNetV2 expects pixel values scaled to [-1, 1].
    x = layers.Rescaling(1.0 / 127.5, offset=-1, name="mobilenetv2_preprocessing")(inputs)

    weights = "imagenet" if pretrained else None
    try:
        backbone = keras.applications.MobileNetV2(
            input_shape=(*IMG_SIZE, 3),
            include_top=False,
            weights=weights,
        )
    except Exception as exc:
        if not pretrained:
            raise
        print("\nWARNING: Could not download ImageNet weights.")
        print(f"Reason: {exc}")
        print("Falling back to MobileNetV2 initialized from scratch.")
        backbone = keras.applications.MobileNetV2(
            input_shape=(*IMG_SIZE, 3),
            include_top=False,
            weights=None,
        )

    # Freeze the feature extractor for a straightforward transfer-learning baseline.
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.30, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="binary_prediction")(x)

    model = keras.Model(inputs, outputs, name="MobileNetV2_Person_Classification")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_model(model, test_ds):
    y_true = np.concatenate([labels.numpy().reshape(-1) for _, labels in test_ds]).astype(int)
    probabilities = model.predict(test_ds, verbose=1).reshape(-1)
    y_pred = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "class_names": CLASS_NAMES,
        "test_images": int(len(y_true)),
    }

    print("\nTest metrics")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['precision_macro']:.4f}")
    print(f"Macro recall:    {metrics['recall_macro']:.4f}")
    print(f"Macro F1-score:  {metrics['f1_macro']:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("MobileNetV2 — Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=160)
    plt.show()

    with open(OUT_DIR / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **metrics,
                "confusion_matrix": cm.tolist(),
                "classification_report": classification_report(
                    y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
                    output_dict=True, zero_division=0
                ),
            },
            f,
            indent=2,
        )
    return metrics


def predict_image(model, image_path):
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = keras.utils.load_img(image_path, target_size=IMG_SIZE)
    array = keras.utils.img_to_array(image)
    probability_class_1 = float(model.predict(array[None, ...], verbose=0)[0, 0])

    # With CLASS_NAMES order above, sigmoid predicts the class at index 1.
    predicted_index = 1 if probability_class_1 >= 0.5 else 0
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probability_class_1 if predicted_index == 1 else 1.0 - probability_class_1

    print(f"\nImage: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence (model score): {confidence:.2%}")
    return predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate MobileNetV2 on the existing people image dataset.")
    parser.add_argument("--predict", type=str, default=None, help="Optional image path to classify.")
    args = parser.parse_args()

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Detected GPU(s): {tf.config.list_physical_devices('GPU') or 'None (CPU training)'}")
    print(f"Project root: {ROOT}")
    print(f"Dataset path: {DATA_DIR}")

    check_dataset()
    train_ds, val_ds, test_ds = load_datasets()
    model = build_model(pretrained=PRETRAINED)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(OUT_DIR / "best_mobilenetv2.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(OUT_DIR / "training_log.csv")),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]

    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    training_seconds = time.time() - start

    # Load the best validation-loss checkpoint before test evaluation.
    best_model_path = OUT_DIR / "best_mobilenetv2.keras"
    if best_model_path.exists():
        model = keras.models.load_model(best_model_path)

    model.save(OUT_DIR / "final_mobilenetv2.keras")

    # Save training curves.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("MobileNetV2 Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("MobileNetV2 Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_curves.png", dpi=160)
    plt.show()

    metrics = evaluate_model(model, test_ds)
    metrics["training_seconds"] = round(training_seconds, 2)
    metrics["trainable_parameters"] = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    with open(OUT_DIR / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model and results in: {OUT_DIR}")
    print(f"Training time: {training_seconds / 60:.2f} minutes")

    if args.predict:
        predict_image(model, args.predict)


if __name__ == "__main__":
    main()
