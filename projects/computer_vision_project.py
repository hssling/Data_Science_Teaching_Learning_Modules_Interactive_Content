#!/usr/bin/env python3
"""
Computer Vision Project: Image Classification with CNNs
======================================================

A complete end-to-end computer vision project demonstrating:
- Image preprocessing and augmentation
- Convolutional Neural Networks (CNNs)
- Transfer learning with pre-trained models
- Model training and evaluation
- Deployment considerations

Dataset: CIFAR-10 (simulated with synthetic data)
Goal: Classify images into 10 categories
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🖼️ COMPUTER VISION PROJECT: IMAGE CLASSIFICATION WITH CNNS")
print("=" * 65)

class ImageClassificationProject:
    """Complete image classification project using CNNs"""

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                          'dog', 'frog', 'horse', 'ship', 'truck']
        self.input_shape = (32, 32, 3)
        self.num_classes = len(self.class_names)

    def create_synthetic_dataset(self):
        """Create synthetic CIFAR-10 like dataset"""
        print("\n📊 1. DATA GENERATION AND PREPARATION")
        print("-" * 40)

        # Generate synthetic images (32x32x3 RGB images)
        n_samples = 10000
        n_samples_per_class = n_samples // self.num_classes

        print(f"Generating {n_samples} synthetic images...")
        print(f"Samples per class: {n_samples_per_class}")
        print(f"Image shape: {self.input_shape}")

        # Create synthetic images with class-specific patterns
        images = []
        labels = []

        for class_idx, class_name in enumerate(self.class_names):
            print(f"Generating images for class: {class_name}")

            for i in range(n_samples_per_class):
                # Create base image with some noise
                image = np.random.randint(0, 256, self.input_shape, dtype=np.uint8)

                # Add class-specific patterns
                if class_name == 'airplane':
                    # Horizontal lines for wings
                    image[14:18, 8:24] = [255, 255, 255]  # White wings
                    image[12:20, 16:18] = [255, 255, 255]  # Fuselage
                elif class_name == 'automobile':
                    # Rectangular shape for car body
                    image[18:26, 6:26] = [255, 0, 0]  # Red car body
                    image[20:24, 8:12] = [0, 0, 255]  # Blue windows
                elif class_name == 'bird':
                    # Wing-like pattern
                    image[12:20, 10:22] = [255, 255, 0]  # Yellow body
                    image[14:18, 6:14] = [0, 255, 255]  # Cyan wings
                elif class_name == 'cat':
                    # Cat-like features
                    image[14:22, 12:20] = [255, 165, 0]  # Orange body
                    image[16:20, 14:18] = [0, 0, 0]  # Black ears/nose
                elif class_name == 'deer':
                    # Deer-like pattern
                    image[12:24, 10:22] = [139, 69, 19]  # Brown body
                    image[14:18, 8:16] = [255, 255, 255]  # White spots
                elif class_name == 'dog':
                    # Dog-like features
                    image[16:24, 8:24] = [160, 82, 45]  # Brown dog
                    image[18:22, 10:14] = [0, 0, 0]  # Black nose
                elif class_name == 'frog':
                    # Frog-like pattern
                    image[18:24, 12:20] = [0, 255, 0]  # Green body
                    image[20:22, 14:18] = [255, 255, 255]  # White belly
                elif class_name == 'horse':
                    # Horse-like features
                    image[14:24, 8:24] = [160, 82, 45]  # Brown horse
                    image[16:20, 10:14] = [255, 255, 255]  # White blaze
                elif class_name == 'ship':
                    # Ship-like pattern
                    image[20:24, 4:28] = [0, 0, 255]  # Blue hull
                    image[18:22, 12:20] = [255, 255, 255]  # White deck
                elif class_name == 'truck':
                    # Truck-like features
                    image[18:26, 6:26] = [128, 128, 128]  # Gray truck
                    image[20:24, 8:12] = [0, 0, 0]  # Black windows

                images.append(image)
                labels.append(class_idx)

        # Convert to numpy arrays
        X = np.array(images, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        y = np.array(labels)

        print(f"\nDataset created successfully!")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Data type: {X.dtype}")
        print(f"Value range: [{X.min():.3f}, {X.max():.3f}]")

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nTraining set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Class distribution in training:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  {self.class_names[cls]}: {count} samples")

        return X, y

    def data_augmentation(self):
        """Set up data augmentation for training"""
        print("\n🔄 2. DATA AUGMENTATION SETUP")
        print("-" * 30)

        # Create data augmentation generator
        self.datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )

        # Fit the generator on training data
        self.datagen.fit(self.X_train)

        print("Data augmentation configured:")
        print("✓ Random rotation (±15°)")
        print("✓ Width and height shifts (±10%)")
        print("✓ Horizontal flipping")
        print("✓ Zoom (±10%)")
        print("✓ Nearest fill mode")

        # Visualize some augmented images
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))

        # Original images
        for i in range(5):
            axes[0, i].imshow(self.X_train[i])
            axes[0, i].set_title(f'Original\n{self.class_names[self.y_train[i]]}')
            axes[0, i].axis('off')

        # Augmented images
        for i in range(5):
            augmented = self.datagen.random_transform(self.X_train[i])
            axes[1, i].imshow(augmented)
            axes[1, i].set_title(f'Augmented\n{self.class_names[self.y_train[i]]}')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig('data_augmentation_examples.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("Data augmentation examples saved as 'data_augmentation_examples.png'")

    def build_custom_cnn(self):
        """Build a custom CNN architecture"""
        print("\n🏗️ 3. BUILDING CUSTOM CNN ARCHITECTURE")
        print("-" * 40)

        model = models.Sequential([
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # Compile the model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Custom CNN Architecture:")
        print("- 3 Convolutional blocks with increasing filters (32→64→128)")
        print("- Batch normalization after each conv layer")
        print("- Max pooling and dropout for regularization")
        print("- Dense layer with 512 units")
        print("- Output layer with softmax activation")
        print(f"- Total parameters: {model.count_params():,}")

        return model

    def build_transfer_learning_models(self):
        """Build models using transfer learning"""
        print("\n🔄 4. TRANSFER LEARNING MODELS")
        print("-" * 35)

        transfer_models = {}

        # VGG16 model
        print("Building VGG16 transfer learning model...")
        vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        vgg_base.trainable = False  # Freeze base layers

        vgg_model = models.Sequential([
            layers.Lambda(lambda x: tf.image.resize(x, (48, 48))),  # Resize for VGG16
            vgg_base,
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        vgg_model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        transfer_models['VGG16'] = vgg_model

        # ResNet50 model
        print("Building ResNet50 transfer learning model...")
        resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        resnet_base.trainable = False

        resnet_model = models.Sequential([
            layers.Lambda(lambda x: tf.image.resize(x, (64, 64))),  # Resize for ResNet50
            resnet_base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        resnet_model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        transfer_models['ResNet50'] = resnet_model

        # MobileNetV2 model (lighter)
        print("Building MobileNetV2 transfer learning model...")
        mobilenet_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        mobilenet_base.trainable = False

        mobilenet_model = models.Sequential([
            mobilenet_base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        mobilenet_model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        transfer_models['MobileNetV2'] = mobilenet_model

        print("Transfer learning models created:")
        for name, model in transfer_models.items():
            print(f"✓ {name}: {model.count_params():,} parameters")

        return transfer_models

    def train_and_evaluate_models(self):
        """Train and evaluate all models"""
        print("\n🚀 5. MODEL TRAINING AND EVALUATION")
        print("-" * 40)

        # Define callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        # Custom CNN training
        print("Training Custom CNN...")
        custom_cnn = self.build_custom_cnn()

        history_custom = custom_cnn.fit(
            self.datagen.flow(self.X_train, self.y_train, batch_size=64),
            epochs=50,
            validation_data=(self.X_test, self.y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # Evaluate custom CNN
        custom_loss, custom_accuracy = custom_cnn.evaluate(self.X_test, self.y_test, verbose=0)
        custom_pred = np.argmax(custom_cnn.predict(self.X_test), axis=1)

        self.models['Custom CNN'] = {
            'model': custom_cnn,
            'history': history_custom,
            'accuracy': custom_accuracy,
            'predictions': custom_pred
        }

        print(".3f"
        # Transfer learning models training
        transfer_models = self.build_transfer_learning_models()

        for name, model in transfer_models.items():
            print(f"\nTraining {name}...")

            # Adjust batch size for different models
            batch_size = 32 if name == 'VGG16' else 64

            history = model.fit(
                self.datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
                epochs=30,
                validation_data=(self.X_test, self.y_test),
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )

            # Evaluate
            loss, accuracy = model.evaluate(self.X_test, self.y_test, verbose=0)
            pred = np.argmax(model.predict(self.X_test), axis=1)

            self.models[name] = {
                'model': model,
                'history': history,
                'accuracy': accuracy,
                'predictions': pred
            }

            print(".3f"
        # Find best model
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['accuracy'])
        self.best_model = self.models[best_model_name]

        print(f"\n🏆 Best Model: {best_model_name}")
        print(".3f"
        return self.models

    def model_comparison_and_analysis(self):
        """Compare model performances and analyze results"""
        print("\n📊 6. MODEL COMPARISON AND ANALYSIS")
        print("-" * 40)

        # Create comparison dataframe
        results_df = pd.DataFrame({
            'Model': list(self.models.keys()),
            'Accuracy': [self.models[model]['accuracy'] for model in self.models.keys()]
        }).sort_values('Accuracy', ascending=False)

        print("Model Performance Comparison:")
        print(results_df.round(4))

        # Plot comparison
        plt.figure(figsize=(12, 6))
        bars = plt.bar(results_df['Model'], results_df['Accuracy'], color='skyblue')
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, acc in zip(bars, results_df['Accuracy']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    '.3f', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Detailed evaluation of best model
        print(f"\nDetailed Analysis of Best Model: {results_df.iloc[0]['Model']}")
        print("-" * 50)

        best_pred = self.best_model['predictions']

        # Classification report
        print("Classification Report:")
        print(classification_report(self.y_test, best_pred, target_names=self.class_names))

        # Confusion matrix
        cm = confusion_matrix(self.y_test, best_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix - Best Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Training history for best model
        if 'history' in self.best_model:
            history = self.best_model['history']
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()
            plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
            plt.show()

    def fine_tuning_experiment(self):
        """Experiment with fine-tuning pre-trained models"""
        print("\n🔧 7. FINE-TUNING EXPERIMENT")
        print("-" * 30)

        # Fine-tune MobileNetV2 (lighter model for fine-tuning)
        print("Fine-tuning MobileNetV2...")

        # Load base model
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)

        # Unfreeze some layers for fine-tuning
        base_model.trainable = True

        # Fine-tune from a specific layer onwards
        fine_tune_at = 100  # Fine-tune from this layer onwards

        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # Create fine-tuned model
        fine_tuned_model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # Compile with lower learning rate for fine-tuning
        fine_tuned_model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"Fine-tuned model parameters: {fine_tuned_model.count_params():,}")
        print(f"Trainable parameters: {sum([layer.count_params() for layer in fine_tuned_model.layers if layer.trainable]):,}")

        # Train with fine-tuning
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        )

        history_fine_tune = fine_tuned_model.fit(
            self.datagen.flow(self.X_train, self.y_train, batch_size=64),
            epochs=20,
            validation_data=(self.X_test, self.y_test),
            callbacks=[early_stopping],
            verbose=1
        )

        # Evaluate fine-tuned model
        loss, accuracy = fine_tuned_model.evaluate(self.X_test, self.y_test, verbose=0)

        print("
Fine-tuning Results:")
        print(".3f"
        print(".3f"
        # Compare with base transfer learning
        base_accuracy = self.models['MobileNetV2']['accuracy']
        improvement = accuracy - base_accuracy
        print(".3f"
        if improvement > 0:
            print("✓ Fine-tuning improved performance!")
        else:
            print("✗ Fine-tuning did not improve performance")

    def deployment_considerations(self):
        """Discuss deployment and production considerations"""
        print("\n🚀 8. DEPLOYMENT CONSIDERATIONS")
        print("-" * 35)

        print("Production Deployment Checklist:")
        print("✓ Model Optimization:")
        print("  - TensorFlow Lite conversion for mobile/edge deployment")
        print("  - Model quantization (float16/int8)")
        print("  - Pruning and compression techniques")

        print("\n✓ Inference Optimization:")
        print("  - Batch processing for multiple images")
        print("  - GPU acceleration with TensorRT/CUDA")
        print("  - Model serving with TensorFlow Serving")

        print("\n✓ API Development:")
        print("  - REST API using Flask/FastAPI")
        print("  - Image upload and preprocessing")
        print("  - Response formatting with confidence scores")

        print("\n✓ Scalability:")
        print("  - Containerization with Docker")
        print("  - Kubernetes orchestration")
        print("  - Load balancing and auto-scaling")

        print("\n✓ Monitoring & Maintenance:")
        print("  - Performance monitoring (latency, throughput)")
        print("  - Model drift detection")
        print("  - A/B testing for model updates")
        print("  - Automated retraining pipelines")

        # Example model conversion to TensorFlow Lite
        import tensorflow as tf

        print("\n📱 Example: TensorFlow Lite Conversion")
        print("-" * 40)

        # Convert best model to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.best_model['model'])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        # Save the model
        with open('image_classification_model.tflite', 'wb') as f:
            f.write(tflite_model)

        print("Model converted to TensorFlow Lite: 'image_classification_model.tflite'")
        print(f"Original model size: ~{self.best_model['model'].count_params() * 4 / 1024 / 1024:.1f} MB (estimated)")
        print(f"TFLite model size: {len(tflite_model) / 1024 / 1024:.1f} MB")

        # Example prediction function
        def predict_image(image_array):
            """
            Example prediction function for deployment

            Parameters:
            image_array (numpy.ndarray): Preprocessed image array (32x32x3)

            Returns:
            dict: Prediction results
            """
            # Ensure correct shape and type
            if image_array.shape != self.input_shape:
                # Resize if necessary
                image_array = tf.image.resize(image_array, self.input_shape[:2])
            if image_array.dtype != np.float32:
                image_array = image_array.astype(np.float32)

            # Add batch dimension
            image_batch = np.expand_dims(image_array, axis=0)

            # Make prediction
            predictions = self.best_model['model'].predict(image_batch, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]

            return {
                'predicted_class': self.class_names[predicted_class],
                'class_index': predicted_class,
                'confidence': confidence,
                'all_probabilities': predictions[0]
            }

        # Test prediction on a few examples
        print("\n🧪 Sample Predictions:")
        for i in range(5):
            result = predict_image(self.X_test[i])
            true_class = self.class_names[self.y_test[i]]
            print(f"Image {i+1}: True={true_class}, Predicted={result['predicted_class']}, Confidence={result['confidence']:.3f}")

    def run_complete_project(self):
        """Run the complete computer vision project"""
        print("Starting Complete Computer Vision Project")
        print("=" * 45)

        # Execute all steps
        self.create_synthetic_dataset()
        self.data_augmentation()
        self.train_and_evaluate_models()
        self.model_comparison_and_analysis()
        self.fine_tuning_experiment()
        self.deployment_considerations()

        print("\n" + "=" * 45)
        print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
        print("=" * 45)
        print("Generated files:")
        print("- data_augmentation_examples.png")
        print("- model_comparison.png")
        print("- confusion_matrix.png")
        print("- training_history.png")
        print("- image_classification_model.tflite")
        print("\nKey Achievements:")
        print("✓ Complete computer vision workflow implemented")
        print("✓ Custom CNN architecture from scratch")
        print("✓ Transfer learning with pre-trained models")
        print("✓ Data augmentation and regularization")
        print("✓ Model optimization and TensorFlow Lite conversion")
        print("✓ Production deployment considerations")
        print("✓ End-to-end image classification system")

if __name__ == "__main__":
    # Run the complete project
    project = ImageClassificationProject()
    project.run_complete_project()
