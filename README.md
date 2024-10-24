# IRIS Identification System using CNN and VGG16

This project demonstrates an Iris Identification system using Convolutional Neural Networks (CNN) and the pre-trained VGG16 model from Keras. The system allows users to upload an image of an iris, compares it against a database of images, and identifies the closest match using feature extraction and cosine similarity. The system also provides a feature extraction visualization and allows for confusion matrix generation.

## Features
- **Image Upload**: Users can upload an iris image for identification.
- **Database Matching**: The system compares the uploaded image with a JSON-based database containing iris images and user details.
- **VGG16-based Feature Extraction**: Uses the VGG16 pre-trained model to extract features from both the uploaded image and the database images.
- **Cosine Similarity for Matching**: Matches the uploaded image with the database using cosine similarity.
- **Feature Visualization**: Highlights and visualizes the extracted features from the CNN model.
- **Confusion Matrix Generation**: Generates a confusion matrix (currently a placeholder with random data).
  
## Installation

1. **Install the required libraries**:
   ```
   pip install tensorflow keras opencv-python-headless numpy pillow matplotlib seaborn
   ```

2. **Run the application**:
   The application is developed using `Tkinter` for the GUI. Simply run the `iris_identification.py` script:
   ```
   python iris_identification.py
   ```

## Usage

1. **Browse and load the JSON database**: The system expects a JSON file containing paths to iris images and associated user details.
2. **Upload an image**: You can upload an iris image in `.jpg`, `.jpeg`, or `.png` formats.
3. **Start Iris Identification**: The system will compare the uploaded image with the database and display the match details (if any).
4. **Feature Extraction Visualization**: The user can visualize the features extracted by the CNN from the uploaded image.
5. **Confusion Matrix**: This option (currently commented out) allows for confusion matrix generation based on predicted and actual values.

## JSON Database Structure

The system uses a JSON file to store details of users and paths to iris images. The structure is as follows:

```json
{
  "1": {
    "Name": "John Doe",
    "Address": "123 Main St",
    "Phone": "1234567890",
    "Path": "path_to_image_1.jpg"
  },
  "2": {
    "Name": "Jane Smith",
    "Address": "456 Elm St",
    "Phone": "9876543210",
    "Path": "path_to_image_2.jpg"
  }
}
```

## Key Functions

- **`preprocess_image(image_path)`**: Preprocesses the image by resizing it and preparing it for the VGG16 model.
- **`extract_features(image, model)`**: Extracts features from the image using the VGG16 model.
- **`show_feature_extraction(image_path)`**: Visualizes and highlights the CNN features on the input image.
- **`run_identification()`**: Compares the uploaded image with images in the database and displays the best match.
- **`generate_confusion_matrix()`**: Generates a confusion matrix (currently with dummy data for testing).

## Notes

- The system uses the pre-trained VGG16 model, with feature extraction done at the `block5_conv3` layer.
- Cosine similarity is used to compare feature vectors between the uploaded image and database images.
- The current confusion matrix functionality is for demonstration purposes and uses randomly generated data.

## Author Information

- **Name**: Rohan Palkar
- **Class**: MSc. IT (AI)
- **Email**: rohanpalkar2000@gmail.com

## Demo Video

- Link: https://www.linkedin.com/posts/rohanpalkar2000_mscit-artificialintelligence-machinelearning-activity-7234094563865321472-30oa?utm_source=share&utm_medium=member_desktop


## Future Improvements

- Integrating a real dataset for confusion matrix generation.
- Improving the GUI for better usability and feedback.
- Optimizing the matching process for larger databases.

---

This `README.md` provides an overview of the project, instructions for setting it up, and details about the key features and functionality. Let me know if you need any adjustments or additional details!

