# cnn-model-basic

## Content
This model is used to detect images.

## File Structure
**To run the file below, pay attention to the strcuture of *testing* and *training* directory and *app.py***
The file structure is shown below:
- Working Directory
  - .git
  - .venv
  - checkpoint
  - saved_model
  - testing
      - [testingImages].jpg|.png|.jpeg
  - training
      - apel
        - [testingImages].jpg|.png|.jpeg
      - panda
        - [testingImages].jpg|.png|.jpeg
      - headphone
        - [testingImages].jpg|.png|.jpeg
  - app.py

# How it works
## Preprocess Image
The techniques used to preprocess the image is by resizing the image into 128 x 128. Then this image is resized into float so that it is divisible by 255.

## Model
The model used is CNN, the input size is 128 x 128 with 3 color channel (RGB). Inside it there are also 2 hidden layers. In the last layers, there is a list of nodes which represents the number of labels that exists that the model can understand

## Results
[Results](/results.png)
