# Proyek Klasifikasi Gambar - Sign Language Classification

This project contains a full sign language image classification workflow in one notebook. It uses the Sign MNIST dataset to train a convolutional neural network and exports the trained model in multiple deployment formats.

## What is inside

- `Template_Submission_Akhir.ipynb`
  - Main notebook with complete code, explanations, and execution results.
- `sign_mnist_train.csv`
  - Training data with 27,455 images and label values for 24 sign language classes.
- `sign_mnist_test.csv`
  - Test data with 7,172 images used to evaluate final performance.
- `best_model.h5`
  - The best Keras model saved during training based on validation accuracy.
- `.gitignore`
  - Git ignore rules for Python, Jupyter notebook checkpoints, and local environment artifacts.
- `saved_model/`
  - TensorFlow SavedModel format of the trained model.
- `tfjs_model/model.json`
  - Exported model for TensorFlow.js browser deployment.
- `tflite/model.tflite`
  - TensorFlow Lite model for mobile or edge inference.
- `tflite/label.txt`
  - Labels file for the TFLite model.
- Example images: `amer_sign2.png`, `amer_sign3.png`, `american_sign_language.PNG`
  - Reference images used to illustrate the sign language theme.

## Step-by-step process

The project follows an ordered process from raw data to deployment-ready models.

1. **Import dependencies**
   - Load Python libraries: pandas, NumPy, TensorFlow, Matplotlib, and scikit-learn.
   - Print the TensorFlow version and GPU availability.

2. **Load dataset**
   - Read the CSV files `sign_mnist_train.csv` and `sign_mnist_test.csv`.
   - Extract the label column separately from image pixels.
   - Reshape pixel vectors into 28x28 grayscale images with shape `(28, 28, 1)`.
   - Confirm dataset size, image resolution, and number of classes.

3. **Preprocess images**
   - Normalize all pixel values by dividing by 255 so values range from 0 to 1.
   - Split the original training data into training and validation subsets.
   - Use stratified sampling to keep class balance in the validation set.
   - Report dataset sizes for training, validation, and test data.

4. **Define the model**
   - Build a sequential convolutional neural network in Keras.
   - Add data augmentation layers: random rotation and zoom.
   - Add convolutional and max pooling layers to learn spatial patterns.
   - Flatten the feature maps and add dense layers with dropout.
   - Use a final softmax layer for 25 output labels.
   - Compile the model with Adam optimizer and sparse categorical crossentropy loss.

5. **Configure training callbacks**
   - Create `EarlyStopping` to stop training when validation accuracy stops improving.
   - Add `ReduceLROnPlateau` to lower the learning rate if validation loss stalls.
   - Use `ModelCheckpoint` to save the best model weights to `best_model.h5`.

6. **Train the model**
   - Fit the model on the training set with validation data.
   - Train in batches of 64 for up to 30 epochs.
   - Monitor validation accuracy and stop early when improvements stop.
   - Save the best-performing model checkpoint automatically.

7. **Evaluate performance**
   - Evaluate the saved model on the test dataset.
   - Report test accuracy and test loss.
   - Visualize training history and model results in the notebook.

8. **Export deployment formats**
   - Save the final trained model in TensorFlow SavedModel format.
   - Export the model for browser use with TensorFlow.js.
   - Convert the model to TensorFlow Lite for mobile or edge inference.

## How to run

- Open `Template_Submission_Akhir.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
- Run each cell sequentially from top to bottom.
- Verify the output in each stage before continuing.
- After training, use the saved model files in `best_model.h5`, `saved_model/`, `tfjs_model/`, or `tflite/`.

## Notes

- This repository is intended for GitHub upload and project submission.
- A new `.gitignore` file has been added to keep local Python artifacts and notebook checkpoints out of version control.
- The notebook uses a clear stage-by-stage workflow with data loading, preprocessing, training, evaluation, and export.
- The language is simple and direct, with no vague conclusions.
