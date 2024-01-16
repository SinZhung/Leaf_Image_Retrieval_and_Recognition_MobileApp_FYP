import base64
import json
from flask import Flask, request, jsonify
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pyarrow.parquet as pq
from flask_cors import cross_origin, CORS

from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors



app = Flask(__name__)
CORS(app)

# FIRST ALGORITHM
def algorithm1(query_image):
    # Load the dataset_features_1.json
    with open('dataset_features_1.json', 'r') as file:
        json_data = json.load(file)

    dataset_features_1 = np.array(list(json_data.values()))
    keys = list(json_data.keys())

    # Normalize the dataset
    scaler = StandardScaler()
    scaled_dataset = scaler.fit_transform(dataset_features_1)

    # Preprocess the query image
    query_features = algorithm1_Feature_Extraction(query_image)
    scaled_query_features = scaler.transform(np.array(query_features).reshape(1, -1))
    print(scaled_query_features)

    # Calculate similarity scores
    similarity_scores = cosine_similarity(scaled_query_features, scaled_dataset)
    similar_indices = np.argsort(similarity_scores)[0][::-1][:10]
    similarity_scores = similarity_scores[0, similar_indices]

    # Filter similarity scores above the threshold
    filtered_indices = similar_indices[similarity_scores > 0.7]
    filtered_scores = similarity_scores[similarity_scores > 0.7]

    # Check if there are any items remaining after filtering
    if len(filtered_indices) == 0:
        results = {
            'Indices': [],
            'Result': []
        }
    else:
        # Get the corresponding keys for the filtered indices
        top10_keys = [keys[index] for index in filtered_indices]

        # Format the similarity scores with rounding
        rounded_scores = [f"Similarity Score: {score:.4f}" for score in filtered_scores]

        # Construct the results dictionary
        results = {
            'Indices': top10_keys,
            'Result': rounded_scores
        }
        print(results)

    return jsonify(results)
    
    
def algorithm1_Feature_Extraction(new_leaf):
    # Image Pre-processing
    def preprocess_image(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        filtered_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
        return thresholded_image, filtered_image

    # Color-based Feature Extraction
    def extract_color_features(image):
        features = []
        for i in range(3):
            plane_mean = np.mean(image[:, :, i])
            plane_var = np.var(image[:, :, i])
            features.extend([plane_mean, plane_var])
        return features

    # Structure-based Feature Extraction
    def extract_structure_features(image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return [0] * 5  # Return default values if no contour is found
        leaf_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(leaf_contour)
        perimeter = cv2.arcLength(leaf_contour, True)
        (x, y, w, h) = cv2.boundingRect(leaf_contour)
        aspect_ratio = float(w) / h
        return [area, perimeter, w, h, aspect_ratio]

    # Shape-based Feature Extraction
    def extract_shape_features(image):
        structuring_elements = [cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                                cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
                                ]
        shape_features = []
        for structuring_element in structuring_elements:
            eroded_image = cv2.erode(image, structuring_element)
            element_count = np.sum(eroded_image)
            shape_features.append(element_count)
        return shape_features

    def extract_texture_features(image):
        # Calculate the gray-level co-occurrence matrix
        glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        
        # Extract texture features
        contrast = graycoprops(glcm, 'contrast')
        correlation = graycoprops(glcm, 'correlation')
        energy = graycoprops(glcm, 'energy')
        homogeneity = graycoprops(glcm, 'homogeneity')
        
        # Return the texture features as a list
        texture_features = [contrast[0, 0], correlation[0, 0], energy[0, 0], homogeneity[0, 0]]
        return texture_features

    def extract_features(image):
        thresholded_image, filtered_image = preprocess_image(image)
        color_features = extract_color_features(image)
        structure_features = extract_structure_features(filtered_image)
        shape_features = extract_shape_features(filtered_image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        texture_features = extract_texture_features(gray_image)


        # Concatenate all features into a single feature vector
        features = color_features + structure_features + shape_features + texture_features
        return features
    
    return(extract_features(new_leaf))

#SECOND Algorithm
def algorithm2(query_image):
    def knn_search(query_descriptors, database_descriptors, k=2, distance_threshold=300):
        # Reshape query_descriptors to 2D array
        query_descriptors = query_descriptors.reshape(1, -1)

        knn = NearestNeighbors(n_neighbors=k, metric='manhattan')
        knn.fit(database_descriptors)
        distances, indices = knn.kneighbors(query_descriptors)

        # Apply distance threshold to consider only close match points
        close_indices = np.where(distances < distance_threshold)
        distances = distances[close_indices]
        indices = indices[close_indices]

        # Check if the indices array is empty after applying the distance threshold
        if len(indices) == 0:
            return [], []

        return distances, indices
        

    # Load dataset features from JSON
    with open('dataset_features_2.json', 'r') as file:
        json_data = json.load(file)

    keys = list(json_data.keys())
    dataset_features_2 = np.array([
        np.pad(descriptors, [(0, 64 - len(descriptors))], mode='constant') if len(descriptors) < 64 else descriptors
        for descriptors in json_data.values()
    ])

    # Extract features from the query image
    query_features = algorithm2_Feature_Extraction(query_image)
    print(query_features)

    # Perform KNN search
    distances, indices = knn_search(query_features, dataset_features_2, k=2, distance_threshold=300)

    # Convert distances to NumPy array
    distances = np.array(distances)

    if len(distances) == 0:
        # Return empty results
        results = {
            'Indices': [],
            'Result': []
        }
        print(results)
        return jsonify(results)

    else:
        # Sort matches based on distances
        sorted_indices = np.argsort(distances.flatten())
        sorted_distances = distances.flatten()[sorted_indices]

        # Get the top 10 sorted indices and distances
        top10_indices = indices.flatten()[sorted_indices][:10]
        top10_distances = sorted_distances[:10]

        # Get the corresponding keys for the top 10 indices
        top10_keys = [keys[index] for index in top10_indices]

        # Format the top10_distances with rounding
        rounded_distances = [f"Distance: {distance:.2f}" for distance in top10_distances]

        # Construct the results dictionary
        results = {
            'Indices': top10_keys,
            'Result': rounded_distances
        }

        print(rounded_distances)
        print(top10_keys)

        return jsonify(results)
    
def algorithm2_Feature_Extraction(new_leaf):
    # Preprocessing
    def preprocess_image(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        eroded_image = cv2.erode(gray_image, kernel, iterations=1)
        processed_image = cv2.dilate(eroded_image, kernel, iterations=1)
        return processed_image

    # Feature Extraction using SIFT
    def extract_features(image):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)

        if descriptors is not None:
            descriptors_concatenated = np.concatenate(descriptors, axis=0)
            descriptors_concatenated = descriptors_concatenated[:64] 
        return descriptors_concatenated
    
    # Process Image
    processed_image = preprocess_image(new_leaf)

    return extract_features(processed_image)

# THIRD ALGORITHM
def algorithm3(query_image):
    # Load the dataset features from the Parquet file
    dataset_table = pq.read_table("dataset_features_3.parquet")
    dataset_df = dataset_table.to_pandas()

    dataset_features_3 = dataset_df['Features'].values.tolist()
    dataset_features_3 = np.array(dataset_features_3)

    key = dataset_df['Index'].values.tolist()

    query_features = algorithm3_Feature_Extraction(query_image)

    # Calculate cosine similarity
    similarities = cosine_similarity(query_features.reshape(1, -1), dataset_features_3)
    similar_indices = np.argsort(similarities)[0][::-1][:10]
    similarity_scores = similarities[0, similar_indices]

    # Filter similarity scores above the threshold
    threshold = 0.55
    filtered_indices = similar_indices[similarity_scores > threshold]
    filtered_scores = similarity_scores[similarity_scores > threshold]
    
    # Check if there are any items remaining after filtering
    if len(filtered_indices) == 0:
        results = {
            'Indices': [],
            'Result': []
        }
    else:
        # Get the corresponding keys for the filtered indices
        top10_keys = [key[index] for index in filtered_indices]

        # Format the similarity scores with rounding
        rounded_scores = [f"Similarity Score: {score:.4f}" for score in filtered_scores]

        # Construct the results dictionary
        results = {
            'Indices': top10_keys,
            'Result': rounded_scores
        }
        print(top10_keys)
        print(rounded_scores)
    
    
    return jsonify(results)

def algorithm3_Feature_Extraction(new_leaf):

    def extract_leaf_shape(image):
        # Step 2: Canny Edge Detection
        edges = cv2.Canny(image, 75, 125)

        # Step 3: Boundary Point Extraction using Centroid-Radii Model
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundary_points = []
        for contour in contours:
            for point in contour:
                boundary_points.append(point[0])

        # Compute the centroid
        centroid = np.mean(boundary_points, axis=0)

        # Compute the distances from each boundary point to the centroid
        radii = np.linalg.norm(boundary_points - centroid, axis=1)

        # Step 4: Feature Extraction using Hu's Moments
        moments = cv2.moments(image)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Step 5: Shape Representation
        feature_vector = np.concatenate((radii, hu_moments))

        return feature_vector

    # Leaf Vein Extraction
    def extract_leaf_vein(image):
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30)))
        difference = cv2.absdiff(gray, opened)

        return difference

    # Preprocess the image
    gray = cv2.cvtColor(new_leaf, cv2.COLOR_BGR2GRAY)
    shape_features = extract_leaf_shape(gray)
    shape_features = shape_features[:72]
    vein_image = extract_leaf_vein(gray)
    features = np.concatenate((shape_features, vein_image.flatten())).astype(np.float32)
    return features


@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the query image from the request
    file = request.files['image']
    query_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    query_image = cv2.resize(query_image, (400, 300))
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    approach = request.form.get('approach')

    if approach == '1':
        return algorithm1(query_image)
    elif approach == '2':
        return algorithm2(query_image)
    elif approach == '3':
        return algorithm3(query_image)
    else:
        return jsonify({'error': 'Invalid approach selected'})
    

# Route to add a new leaf and update the dataset features
@app.route('/add_new_leaf', methods=['POST'])
@cross_origin()
def add_new_leaf():
    if 'image' in request.files:
        # Image file was uploaded
        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if 'image' in request.form:
        # Image was sent as base64 data
        base64_data = request.form['image']
        image_data = base64.b64decode(base64_data)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    
    image = cv2.resize(image, (400, 300))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ######## FIRST ALGORITHM
    newLeaf_features_1 = algorithm1_Feature_Extraction(image)
    newLeaf_features_1 = np.asarray(newLeaf_features_1)

    with open('dataset_features_1.json', 'r') as file:
        dataset_features_1 = json.load(file)

    # Find the largest index number in the "dataset_features_1" dictionary
    max_index = max(map(int, dataset_features_1.keys()), default=0)

    # Append the new feature entry to the dataset
    dataset_features_1[str(max_index + 1)] = newLeaf_features_1.tolist()  # Convert back to list before appending

    # Save the updated dataset features to "dataset_features_1.json"
    with open('dataset_features_1.json', 'w') as file:
        json.dump(dataset_features_1, file)

    ######## SECOND ALGORITHM
    newLeaf_features_2 = algorithm2_Feature_Extraction(image)
    newLeaf_features_2 = np.asarray(newLeaf_features_2)

    with open('dataset_features_2.json', 'r') as file:
        dataset_features_2 = json.load(file)

    # Append the new feature entry to the dataset
    dataset_features_2[str(max_index + 1)] = newLeaf_features_2.tolist()  # Convert back to list before appending

    # Save the updated dataset features to "dataset_features_2.json"
    with open('dataset_features_2.json', 'w') as file:
        json.dump(dataset_features_2, file)

    ######## FOURTH ALGORITHM
    newLeaf_features_3 = algorithm3_Feature_Extraction(image)
    newLeaf_features_3 = np.asarray(newLeaf_features_3)

    # Load the existing dataset features from the Parquet file
    dataset_features_3 = pq.read_table('dataset_features_3.parquet').to_pandas()
    print(len(dataset_features_3)-1)
    
    # Create a new entry for the new feature
    new_entry = {'Index': max_index+1, 'Features': newLeaf_features_3.tolist()}

    # Add the new entry to the dataset_features_3 DataFrame
    dataset_features_3.loc[len(dataset_features_3.index)] = new_entry
    print(len(dataset_features_3)-1)
    
    # Save the updated dataset features back to the Parquet file
    dataset_features_3.to_parquet('dataset_features_3.parquet')

    return jsonify("Approved successfully!")

@app.route('/delete_leaf', methods=['POST'])
@cross_origin()
def delete_leaf():
    try:
        index = int(request.form.get('index'))
        dataset_files = ['dataset_features_1.json', 'dataset_features_2.json']

        for dataset_file in dataset_files:
            with open(dataset_file, 'r') as file:
                dataset = json.load(file)

            # Remove the leaf entry with the matching index
            for key, value in dataset.items():
                if int(key) == index:
                    del dataset[key]
                    break  # Stop after deleting the first matching entry

            with open(dataset_file, 'w') as file:
                json.dump(dataset, file, indent=4)

        # Delete leaf from the Parquet file
        dataset_features_3 = pq.read_table('dataset_features_3.parquet').to_pandas()
        print(len(dataset_features_3)-1)

        # Find the row with the matching index
        dataset_features_3 = dataset_features_3[dataset_features_3['Index'] != index]
        print(len(dataset_features_3)-1)
        dataset_features_3 = dataset_features_3.reset_index(drop=True)

        # Save the updated dataset features back to the Parquet file
        dataset_features_3.to_parquet('dataset_features_3.parquet')

        return "Delete Successfully"


    except Exception as e:
        return f"Error deleting leaf: {str(e)}"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
