import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

def feature_extraction(X, sift):
    
    # Compute the keypoints and descriptors of all images
    kp_des_list = [sift.detectAndCompute(X, None) for i in range(len(X))]

    # Assing the keypoints and descriptors of the first image before the loop
    kp_vector = kp_des_list[0][0]
    des_vector = kp_des_list[0][1]

    # Append the keypoints and descriptors of the rest of the images
    for kp_des in kp_des_list[1:]:
        kp_vector = np.append(kp_vector, kp_des[0], axis=0)
        des_vector = np.append(des_vector, kp_des[1], axis=0)

    return (kp_vector, des_vector)

class BagOfVisualWords(BaseEstimator, TransformerMixin):

    def __init__(self, n_clusters):

        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        
        # Labels list
        self.labels_list = np.unique(y)

        # Labels dictionary with keypoints and descriptors of each label.
        self.labels_dict = {label: dict.fromkeys(['kp', 'des']) for label in self.labels_list}

        # SIFT object
        sift = cv2.SIFT_create()

        self.bneck_label = None
        self.bneck_value = float("inf")

        # For each label
        for label in self.labels_list:

            # Get all images of that label
            index = np.where(y == label)
            x_temp = X[index]
            
            # Extract features
            kp_vector, des_vector = feature_extraction(x_temp, sift)

            # Find bottleneck
            if len(kp_vector) < self.bneck_value:
                self.bneck_value = len(kp_vector)
                self.bneck_label = label

            # Save keypoints and descriptors
            self.labels_dict[label]['kp'] = kp_vector
            self.labels_dict[label]['des'] = des_vector

        self.n_descriptors = int(0.8*self.bneck_value)

        # Sort keypoints for each label
        for label in self.labels_list:
            self.labels_dict[label]['kp'] = sorted(self.labels_dict[label]['kp'],
                                        key=lambda x: x.response,
                                        reverse=True)

        # Assing the first n_descriptors (strongest) descriptors of the first label
        des_vector = self.labels_dict[self.labels_list[0]]['des'][0:self.n_descriptors]

        # Append the first n_descriptors (strongest) descriptors of the rest of the labels
        for label in self.labels_list[1:]:
            curr_des = self.labels_dict[label]['des'][0:self.n_descriptors]
            des_vector = np.append(des_vector, curr_des, axis=0)

        des_vector = np.float64(des_vector) # Converts to float64
        
        # Cluster descriptors: visual vocabulary or codebook
        self.bag = KMeans(n_clusters=self.n_clusters, random_state=0).fit(des_vector)

        self.n_features = len(des_vector) # Number of features used to build the bag of visual words

        return self

    def transform(self, X):

        N = len(X) # Number of images
        K = self.bag.n_clusters # Number of visual words

        # SIFT object
        sift = cv2.SIFT_create()

        # Feature vector histogram: new and better representation of the images
        feature_vector = np.zeros((N, K))
        visial_word_pos = 0 # Position of the visual word

        # For each image
        for i in range(N):

            # Extract the keypoints descriptors of the current image
            _, curr_des = sift.detectAndCompute(X[i], None)
    
            # Define the feature vector of the current image
            feature_vector_curr = np.zeros(self.bag.n_clusters, dtype=np.float32)

            # Uses the BoVW to predict the visual words of each keypoint descriptors of the current image
            word_vector = self.bag.predict(np.asarray(curr_des, dtype=float))
            
            # For each unique visual word
            for word in np.unique(word_vector):
                res = list(word_vector).count(word) # Count the number of word in word_vector
                feature_vector_curr[word] = res # Increments histogram for that word
            
            # Normalizes the current histogram
            cv2.normalize(feature_vector_curr, feature_vector_curr, norm_type=cv2.NORM_L2)

            feature_vector[visial_word_pos] = feature_vector_curr # Assined the current histogram to the feature vector
            visial_word_pos += 1 # Increments the position of the visual word

        return feature_vector