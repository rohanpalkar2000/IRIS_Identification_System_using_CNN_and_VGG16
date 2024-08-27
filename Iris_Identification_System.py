import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Label
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import json
import threading
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # Resize to fit VGG16 input size
    image = preprocess_input(image)  # Preprocess for VGG16
    return image

# Function to load image paths from JSON database
def load_database(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# Function to extract features using VGG16
def extract_features(image, model):
    image = np.expand_dims(image, axis=0)
    features = model.predict(image)
    return features

# Function to visualize and highlight extracted features on the input image
def show_feature_extraction(image_path):
    try:
        # Preprocess the image
        image = preprocess_image(image_path)

        # Load VGG16 model
        base_model = VGG16(weights='imagenet', include_top=False)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)

        # Extract features
        features = extract_features(image, model)

        # Create a feature-highlighted image by averaging the feature maps
        feature_map = np.mean(features[0], axis=-1)

        # Normalize the feature map for visualization
        feature_map = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Resize feature map back to the original image size
        highlighted_feature_map = cv2.resize(feature_map, (224, 224))

        # Convert the original image back to the range [0, 255] for visualization
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (224, 224))

        # Overlay the feature map on the original image
        overlay = cv2.applyColorMap(highlighted_feature_map, cv2.COLORMAP_JET)
        highlighted_image = cv2.addWeighted(image_resized, 0.6, overlay, 0.4, 0)

        # Convert to PIL image and display in a new window
        highlighted_image_pil = Image.fromarray(highlighted_image)
        highlighted_image_tk = ImageTk.PhotoImage(highlighted_image_pil)

        # Create a new window to display the highlighted image
        feature_window = Toplevel()
        feature_window.title("Feature Extraction Visualization")

        feature_label = Label(feature_window, image=highlighted_image_tk)
        feature_label.image = highlighted_image_tk  # Keep a reference to prevent garbage collection
        feature_label.pack()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while showing feature extraction: {e}")

# Main function
def main():
    root = tk.Tk()
    root.title("Iris Identification System using CNN")
    root.geometry("800x600")

    json_file_path_var = tk.StringVar()
    image_file_path_var = tk.StringVar()

    # Create a label for displaying the selected image
    image_label = tk.Label(root)
    image_label.pack()

    # Label to display matching accuracy
    match_label = tk.Label(root, text="", font=("Helvetica", 12))
    match_label.pack()

    def browse_json_file():
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Choose the database JSON file"
        )
        json_file_path_var.set(file_path)

    def upload_image():
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")],
            title="Upload an image"
        )
        image_file_path_var.set(file_path)

        # Display the selected image
        if file_path:
            image = Image.open(file_path)
            image = image.resize((300, 300))  # Resize the image for display
            photo = ImageTk.PhotoImage(image)
            image_label.config(image=photo)
            image_label.image = photo  # Keep a reference to the image to prevent it from being garbage collected

    def run_identification():
        processing_label.pack()
        root.update_idletasks()

        database_json = json_file_path_var.get()
        input_image_path = image_file_path_var.get()

        if not database_json:
            messagebox.showerror("Error", "No database JSON file selected.")
            processing_label.pack_forget()
            return

        if not input_image_path:
            messagebox.showerror("Error", "No input image selected.")
            processing_label.pack_forget()
            return

        try:
            input_image = preprocess_image(input_image_path)

            # Load VGG16 model
            base_model = VGG16(weights='imagenet', include_top=False)
            model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)

            # Extract features of input image
            input_features = extract_features(input_image, model)

            # Load database
            database = load_database(database_json)

            best_match_path = None
            best_similarity = -1
            best_match_details = None

            # Compare input image with database images
            for key, value in database.items():
                db_image_path = value['Path']
                db_image = preprocess_image(db_image_path)
                db_features = extract_features(db_image, model)

                # Calculate cosine similarity
                cosine_similarity = np.dot(input_features.flatten(), db_features.flatten()) / (np.linalg.norm(input_features) * np.linalg.norm(db_features))

                if cosine_similarity > best_similarity:
                    best_similarity = cosine_similarity
                    best_match_path = db_image_path
                    best_match_details = value

            threshold = 0.8
            if best_similarity > threshold:
                accuracy_percentage = best_similarity * 100  # Convert similarity to percentage
                match_label.config(text=f"Match Found\nSimilarity: {best_similarity:.2f}\nAccuracy: {accuracy_percentage:.2f}%")
                messagebox.showinfo(
                    "Match Found",
                    f"Iris match found with details:\n"
                    f"Name: {best_match_details['Name']}\n"
                    f"Address: {best_match_details['Address']}\n"
                    f"Phone: {best_match_details['Phone']}\n"
                    f"Similarity: {best_similarity:.2f}\n"
                    f"Accuracy: {accuracy_percentage:.2f}%"
                )
            else:
                match_label.config(text="No Match")
                messagebox.showinfo("No Match", "Iris match not found.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            processing_label.pack_forget()
            author_info_label.pack()

    def start_identification_thread():
        thread = threading.Thread(target=run_identification)
        thread.start()

    def generate_confusion_matrix():
        if not image_file_path_var.get():
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Dummy data for confusion matrix example
        y_true = np.random.randint(0, 2, size=100)
        y_pred = np.random.randint(0, 2, size=100)

        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix\n\nAccuracy: {accuracy:.2f},  Precision: {precision:.2f},  Recall: {recall:.2f},  Specificity: {specificity:.2f}')

        plt.show()

    # Layout adjustments for responsiveness
    tk.Label(root, text="JSON File:").pack(pady=5)
    json_entry = tk.Entry(root, textvariable=json_file_path_var, width=50)
    json_entry.pack(pady=5)
    tk.Button(root, text="Browse", command=browse_json_file).pack(pady=5)

    tk.Label(root, text="Upload Image:").pack(pady=10)
    tk.Button(root, text="Upload", command=upload_image).pack(pady=5)

    tk.Button(root, text="Start Iris Identification", command=start_identification_thread).pack(pady=20)

    # tk.Button(root, text="Generate Confusion Matrix", command=generate_confusion_matrix).pack(pady=20)

    tk.Button(root, text="Show Feature Extraction", command=lambda: show_feature_extraction(image_file_path_var.get())).pack(pady=15)

    processing_label = tk.Label(root, text="Processing...", font=("Helvetica", 12))
    author_info_label = tk.Label(root, text="Name: Rohan Palkar\nClass: Msc. IT (AI)\nEmail: rohanpalkar2000@gmail.com", font=("Helvetica", 10), anchor="e")

    # Adjusting layout for author info label
    author_info_label.place(relx=1.0, rely=1.0, anchor='se')

    root.mainloop()

if __name__ == "__main__":
    main()
