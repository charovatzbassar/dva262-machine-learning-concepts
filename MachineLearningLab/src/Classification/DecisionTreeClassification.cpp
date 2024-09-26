#include "DecisionTreeClassification.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/EntropyFunctions.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <utility>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include "../DataUtils/DataPreprocessor.h"
using namespace System::Windows::Forms; // For MessageBox

// DecisionTreeClassification class implementation //


// DecisionTreeClassification is a constructor for DecisionTree class.//
DecisionTreeClassification::DecisionTreeClassification(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr) {}


// Fit is a function to fits a decision tree to the given data.//
void DecisionTreeClassification::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));
	root = growTree(X, y);
}


// Predict is a function that Traverses the decision tree and returns the prediction for a given input vector.//
std::vector<double> DecisionTreeClassification::predict(std::vector<std::vector<double>>& X) {
	std::vector<double> predictions;
	
	// For each input vector, traverse the tree to get a prediction
	for (std::vector<double>& sample : X) {
		predictions.push_back(traverseTree(sample, root));  // Traverse the tree starting from the root node
	}
	
	return predictions;
}


// growTree function: This function grows a decision tree using the given data and labelsand  return a pointer to the root node of the decision tree.//
Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
	
	// Stopping criteria
	if (y.size() < min_samples_split || depth >= max_depth) {
		double leaf_value = mostCommonlLabel(y);  // If stopping criteria are met, create a leaf node
		return new Node(-1, -1, nullptr, nullptr, leaf_value);
	}

	// Variables to store the best split
	double best_gain = -1.0;
	int split_idx = -1;
	double split_thresh = -1.0;

	// Loop through candidate features and potential split thresholds
	for (int feat_idx = 0; feat_idx < n_feats; feat_idx++) {
		std::vector<double> X_column;  // Collect the current feature's column
		for (int i = 0; i < X.size(); i++) {
			X_column.push_back(X[i][feat_idx]);
		}

		// Get unique values from the feature column for splitting
		std::sort(X_column.begin(), X_column.end());
		X_column.erase(std::unique(X_column.begin(), X_column.end()), X_column.end());

		// Try splitting at each unique value
		for (double threshold : X_column) {
			double gain = informationGain(y, X_column, threshold);

			// Check if the gain is better than the current best
			if (gain > best_gain) {
				best_gain = gain;
				split_idx = feat_idx;
				split_thresh = threshold;
			}
		}
	}

	// If no gain, return a leaf node with the most common label
	if (best_gain == -1) {
		double leaf_value = mostCommonlLabel(y);
		return new Node(-1, -1, nullptr, nullptr, leaf_value);
	}

	// Split the dataset based on the best feature and threshold
	std::vector<std::vector<double>> X_left, X_right;
	std::vector<double> y_left, y_right;

	for (int i = 0; i < X.size(); i++) {
		if (X[i][split_idx] <= split_thresh) {
			X_left.push_back(X[i]);
			y_left.push_back(y[i]);
		}
		else {
			X_right.push_back(X[i]);
			y_right.push_back(y[i]);
		}
	}

	// **Add this check to handle empty splits**
	if (y_left.empty() || y_right.empty()) {
		double leaf_value = mostCommonlLabel(y);
		return new Node(-1, -1, nullptr, nullptr, leaf_value);
	}

	// Recursively grow the left and right subtrees
	Node* left = growTree(X_left, y_left, depth + 1);
	Node* right = growTree(X_right, y_right, depth + 1);

	// Return a new node with the best split
	return new Node(split_idx, split_thresh, left, right);
}


/// informationGain function: Calculates the information gain of a given split threshold for a given feature column.
double DecisionTreeClassification::informationGain(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
	// Calculate the parent entropy
	double parent_entropy = EntropyFunctions::entropy(y);

	// Split the data based on the threshold
	std::vector<double> y_left, y_right;
	for (int i = 0; i < X_column.size(); i++) {
		if (X_column[i] <= split_thresh) {
			y_left.push_back(y[i]);
		}
		else {
			y_right.push_back(y[i]);
		}
	}

	// If no data is split, return 0 gain
	if (y_left.empty() || y_right.empty()) {
		return 0.0;
	}

	// Calculate the weighted entropy of the children
	double left_entropy = EntropyFunctions::entropy(y_left);
	double right_entropy = EntropyFunctions::entropy(y_right);
	double n = y.size();
	double n_left = y_left.size();
	double n_right = y_right.size();
	double child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy;

	// Information gain is the difference between parent entropy and child entropy
	return parent_entropy - child_entropy;
}


// mostCommonlLabel function: Finds the most common label in a vector of labels.//
double DecisionTreeClassification::mostCommonlLabel(std::vector<double>& y) {	
	std::unordered_map<double, int> label_counts;

	// Count the occurrences of each label
	for (double label : y) {
		label_counts[label]++;
	}

	// Find the most common label
	double most_common_label = y[0];
	int max_count = 0;
	for (const auto& pair : label_counts) {
		if (pair.second > max_count) {
			max_count = pair.second;
			most_common_label = pair.first;
		}
	}

	return most_common_label;  // Return the most frequent label
}


// traverseTree function: Traverses a decision tree given an input vector and a node.//
double DecisionTreeClassification::traverseTree(std::vector<double>& x, Node* node) {
	// If the node is a leaf node, return its value
	if (node->isLeafNode()) {
		return node->value;
	}

	// Otherwise, decide to go left or right based on the threshold
	if (x[node->feature] <= node->threshold) {
		return traverseTree(x, node->left);  // Go to the left subtree
	} else {
		return traverseTree(x, node->right);  // Go to the right subtree
	}
}


/// runDecisionTreeClassification: this function runs the decision tree classification algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
DecisionTreeClassification::runDecisionTreeClassification(const std::string& filePath, int trainingRatio) {
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

		// Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
		double trainRatio = trainingRatio * 0.01;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData, trainLabels);//

		// Make predictions on the test data
		std::vector<double> testPredictions = predict(testData);

		// Calculate accuracy using the true labels and predicted labels for the test data
		double test_accuracy = Metrics::accuracy(testLabels, testPredictions);


		// Make predictions on the training data
		std::vector<double> trainPredictions = predict(trainData);

		// Calculate accuracy using the true labels and predicted labels for the training data
		double train_accuracy = Metrics::accuracy(trainLabels, trainPredictions);

		MessageBox::Show("Run completed");
		return std::make_tuple(train_accuracy, test_accuracy,
			std::move(trainLabels), std::move(trainPredictions),
			std::move(testLabels), std::move(testPredictions));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, std::vector<double>(),
			std::vector<double>(), std::vector<double>(),
			std::vector<double>());
	}
}