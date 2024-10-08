#include "KMeans.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/PCADimensionalityReduction.h"
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map> 
#include <numeric>
#include <cstdlib> // For std::abs
using namespace System::Windows::Forms; // For MessageBox

///  KMeans class implementation  ///

// KMeans constructor
KMeans::KMeans(int numClusters, int maxIterations)
    : numClusters_(numClusters), maxIterations_(maxIterations) {}

// fit function: Performs K-means clustering on the given dataset and returns the centroids of the clusters.
void KMeans::fit(const std::vector<std::vector<double>>& data) {
    // Number of data points and features
    int n_samples = data.size();
    if (n_samples == 0) return;
    int n_features = data[0].size();

    // Initialize centroids randomly
    centroids_.clear();
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0..n_samples-1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    for (int i = 0; i < numClusters_; ++i) {
        centroids_.push_back(data[indices[i]]);
    }

    // Perform K-means clustering
    std::vector<int> labels(n_samples, -1);

    for (int iter = 0; iter < maxIterations_; ++iter) {
        bool changed = false; // Flag to check if any label has changed

        // Assign data points to the nearest centroid
        for (int i = 0; i < n_samples; ++i) {
            double min_dist = (std::numeric_limits<double>::max)();
            int closest_centroid = -1;

            for (int c = 0; c < numClusters_; ++c) {
                double dist = SimilarityFunctions::euclideanDistance(data[i], centroids_[c]);

                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = c;
                }
            }

            if (labels[i] != closest_centroid) {
                labels[i] = closest_centroid;
                changed = true;
            }
        }

        // Check for convergence
        if (!changed) {
            break;
        }

        // Update centroids
        std::vector<std::vector<double>> newCentroids(numClusters_, std::vector<double>(n_features, 0.0));
        std::vector<int> counts(numClusters_, 0);

        for (int i = 0; i < n_samples; ++i) {
            int cluster = labels[i];
            counts[cluster]++;
            for (int j = 0; j < n_features; ++j) {
                newCentroids[cluster][j] += data[i][j];
            }
        }

        std::uniform_int_distribution<> distrib(0, n_samples - 1);

        for (int c = 0; c < numClusters_; ++c) {
            if (counts[c] > 0) {
                for (int j = 0; j < n_features; ++j) {
                    newCentroids[c][j] /= counts[c];
                }
            }
            else {
                // Reinitialize centroid to a random data point
                int idx = distrib(gen);
                newCentroids[c] = data[idx];
            }
        }

        centroids_ = newCentroids;
    }
}

// predict function: Calculates the closest centroid for each point in the given data set and returns the labels of the closest centroids.
std::vector<int> KMeans::predict(const std::vector<std::vector<double>>& data) const {
    std::vector<int> labels;
    labels.reserve(data.size());

    int n_samples = data.size();
    if (n_samples == 0) return labels;
    int n_features = data[0].size();

    // For each data point
    for (int i = 0; i < n_samples; ++i) {
        double min_dist = (std::numeric_limits<double>::max)();
        int closest_centroid = -1;

        // Iterate through each centroid
        for (int c = 0; c < centroids_.size(); ++c) {
            double dist = SimilarityFunctions::euclideanDistance(data[i], centroids_[c]);

            if (dist < min_dist) {
                min_dist = dist;
                closest_centroid = c;
            }
        }

        // Add the closest centroid to the labels vector
        labels.push_back(closest_centroid);
    }

    return labels; // Return the labels vector
}

/// runKMeans: this function runs the KMeans clustering algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.
std::tuple<double, double, std::vector<int>, std::vector<std::vector<double>>>
KMeans::runKMeans(const std::string& filePath) {
    DataPreprocessor DataPreprocessor;
    try {
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
    }
    catch (const std::exception& e) {
        // Handle the exception
        MessageBox::Show("Not Working");
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return std::make_tuple(0.0, 0.0, std::vector<int>(), std::vector<std::vector<double>>());
    }
}
