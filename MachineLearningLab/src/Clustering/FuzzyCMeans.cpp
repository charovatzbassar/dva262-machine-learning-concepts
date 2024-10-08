#include "FuzzyCMeans.h"
#include "../DataUtils/DataLoader.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/PCADimensionalityReduction.h"
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <random> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <limits.h>
#include <unordered_map> 
using namespace System::Windows::Forms; // For MessageBox


///  FuzzyCMeans class implementation  ///


// FuzzyCMeans function: Constructor for FuzzyCMeans class.//
FuzzyCMeans::FuzzyCMeans(int numClusters, int maxIterations, double fuzziness)
	: numClusters_(numClusters), maxIterations_(maxIterations), fuzziness_(fuzziness) {}


// fit function: Performs Fuzzy C-Means clustering on the given dataset and return the centroids of the clusters.//
void FuzzyCMeans::fit(const std::vector<std::vector<double>>& data) {
	// Create a copy of the data to preserve the original dataset
	std::vector<std::vector<double>> normalizedData = data;

	// Initialize membership matrix randomly
	this->initializeMembershipMatrix(normalizedData.size());
	centroids_.resize(this->numClusters_, std::vector<double>(normalizedData[0].size()));

	// Randomly initialize centroids
	std::srand(std::time(NULL));
	for (int i = 0; i < this->numClusters_; i++) {
		for (int j = 0; j < normalizedData[0].size(); j++) {
			this->centroids_[i][j] = (static_cast<double>(std::rand()) / RAND_MAX);
		}
	}

	double epsilon = 1e-5;   // Convergence threshold

	for (int iteration = 0; iteration < this->maxIterations_; iteration++) {
		std::vector<std::vector<double>> previousCentroids = centroids_;

		// Step 1: Update the membership matrix based on current centroids
		this->updateMembershipMatrix(normalizedData, centroids_);

		// Step 2: Update centroids based on the new membership matrix
		this->updateCentroids(normalizedData);

		// Step 3: Check for convergence (if centroids have not changed significantly)
		bool converged = true;
		for (int i = 0; i < centroids_.size(); i++) {
			for (int j = 0; j < centroids_[i].size(); j++) {
				if (std::abs(centroids_[i][j] - previousCentroids[i][j]) > epsilon) {
					converged = false;
					break;
				}
			}
			if (!converged) break;
		}

		// If centroids have converged, we can exit the loop
		if (converged) {
			break;
		}
	}

}



// initializeMembershipMatrix function: Initializes the membership matrix with random values that sum up to 1 for each data point.//
void FuzzyCMeans::initializeMembershipMatrix(int numDataPoints) {
	membershipMatrix_.clear();
	membershipMatrix_.resize(numDataPoints, std::vector<double>(numClusters_, 0.0));

	std::srand(std::time(NULL));

	// Initialize membership values for each data point
	for (auto& dataPoint : membershipMatrix_) {
		double sum = 0.0;

		// Generate random membership values for each cluster
		for (int j = 0; j < numClusters_; j++) {
			dataPoint[j] = static_cast<double>(std::rand()) / RAND_MAX;
			sum += dataPoint[j];
		}

		// Normalize the membership values so they sum to 1
		for (int j = 0; j < numClusters_; j++) {
			dataPoint[j] /= sum;
		}
	}
}


// updateMembershipMatrix function: Updates the membership matrix using the fuzzy c-means algorithm.//
void FuzzyCMeans::updateMembershipMatrix(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>> centroids_) {

	// Iterate through each data point
	for (int i = 0; i < data.size(); i++) {
		std::vector<double> distances(centroids_.size(), 0.0);

		// Calculate the distance between the data point and each centroid
		for (int j = 0; j < centroids_.size(); j++) {
			distances[j] = SimilarityFunctions::euclideanDistance(data[i], centroids_[j]);
		}

		// Update membership matrix
		for (int j = 0; j < centroids_.size(); j++) {
			double sum = 0.0;

			// Normalize membership values based on the inverse distance
			for (int k = 0; k < centroids_.size(); k++) {
				sum += pow(distances[j] / distances[k], 2.0 / (this->fuzziness_ - 1));
			}

			membershipMatrix_[i][j] = 1.0 / sum;
		}
	}
	
}


// updateCentroids function: Updates the centroids of the Fuzzy C-Means algorithm.//
std::vector<std::vector<double>> FuzzyCMeans::updateCentroids(const std::vector<std::vector<double>>& data) {
	int numClusters = this->centroids_.size();
	int numDataPoints = data.size();
	int numFeatures = data[0].size();

	std::vector<std::vector<double>> newCentroids(numClusters, std::vector<double>(numFeatures, 0.0));

	// Iterate over each cluster to update its centroid
	for (int k = 0; k < numClusters; k++) {
		std::vector<double> numerator(numFeatures, 0.0);
		double denominator = 0.0;

		// Iterate over each data point
		for (int i = 0; i < numDataPoints; i++) {
			double membershipValue = std::pow(membershipMatrix_[i][k], fuzziness_);

			// Update numerator and denominator
			for (int j = 0; j < numFeatures; j++) {
				numerator[j] += membershipValue * data[i][j];
			}
			denominator += membershipValue;
		}

		// Calculate the new centroid for cluster k
		for (int j = 0; j < numFeatures; j++) {
			newCentroids[k][j] = numerator[j] / denominator;
		}
	}

	// Update the centroids_
	centroids_ = newCentroids;

	return centroids_; // Return the updated centroids
}


// predict function: Predicts the cluster labels for the given data points using the Fuzzy C-Means algorithm.//
std::vector<int> FuzzyCMeans::predict(const std::vector<std::vector<double>>& data) const {
	std::vector<int> labels; // Create a vector to store the labels
	labels.reserve(data.size()); // Reserve space for the labels


    // Iterate through each point in the data
    for (int i = 0; i < data.size(); i++) {
        double minDistance = DBL_MAX;
        int bestCluster = -1;

        // Iterate through each centroid
        for (int j = 0; j < this->centroids_.size(); j++) {
            // Calculate the distance between the data point and the centroid
            double distance = SimilarityFunctions::euclideanDistance(data[i], centroids_[j]);

            // If the distance is smaller than the current minimum, update
            if (distance < minDistance) {
                minDistance = distance;
                bestCluster = j;  // Set the current closest centroid as the label
            }
        }

        // Add the label (index of the closest centroid) to the labels vector
        labels.push_back(bestCluster);
    }

    return labels;  // Return the labels vector

}


/// runFuzzyCMeans: this function runs the Fuzzy C-Means clustering algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<int>, std::vector<std::vector<double>>>
FuzzyCMeans::runFuzzyCMeans(const std::string& filePath) {
	DataPreprocessor DataPreprocessor;
	//try {
		// Check if the file path is empty
		if (filePath.empty()) {
			MessageBox::Show("Please browse and select the dataset file from your PC.");
			return {}; // Return an empty vector since there's no valid file path
		}

		// Attempt to open the file
		std::ifstream file(filePath);
		if (!file.is_open()) {
			MessageBox::Show("Failed to open the dataset file");
			return {}; // Return an empty vector since file couldn't be opened
		}

		std::vector<std::vector<double>> dataset; // Create an empty dataset vector
		DataLoader::loadAndPreprocessDataset(filePath, dataset);

		// Use the all dataset for training and testing sets.
		double trainRatio = 1.0;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData);

		// Make predictions on the training data
		std::vector<int> labels = predict(trainData);

		// Calculate evaluation metrics
		// Calculate Davies BouldinIndex using the actual features and predicted cluster labels
		double daviesBouldinIndex = Metrics::calculateDaviesBouldinIndex(trainData, labels);

		// Calculate Silhouette Score using the actual features and predicted cluster labels
		double silhouetteScore = Metrics::calculateSilhouetteScore(trainData, labels);


		// Create an instance of the PCADimensionalityReduction class
		PCADimensionalityReduction pca;

		// Perform PCA and project the data onto a lower-dimensional space
		int num_dimensions = 2; // Number of dimensions to project onto
		std::vector<std::vector<double>> reduced_data = pca.performPCA(trainData, num_dimensions);

		MessageBox::Show("Run completed");
		return std::make_tuple(daviesBouldinIndex, silhouetteScore, std::move(labels), std::move(reduced_data));
	//}
	//catch (const std::exception& e) {
	//	// Handle the exception
	//	MessageBox::Show("Not Working");
	//	std::cerr << "Exception occurred: " << e.what() << std::endl;
	//	return std::make_tuple(0.0, 0.0, std::vector<int>(), std::vector<std::vector<double>>());
	//}
}