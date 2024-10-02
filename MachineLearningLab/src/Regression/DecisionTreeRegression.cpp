#include "DecisionTreeRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <numeric>
#include <unordered_set>
using namespace System::Windows::Forms; // For MessageBox



///  DecisionTreeRegression class implementation  ///


// Constructor for DecisionTreeRegression class.//
DecisionTreeRegression::DecisionTreeRegression(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr)
{
}


// fit function:Fits a decision tree regression model to the given data.//
void DecisionTreeRegression::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));
	root = growTree(X, y);
}


// predict function:Traverses the decision tree and returns the predicted value for a given input vector.//
std::vector<double> DecisionTreeRegression::predict(std::vector<std::vector<double>>& X) {

	std::vector<double> predictions;
	
	// Implement the function
	// TODO

	for (auto& row : X) {
		double pred = this->traverseTree(row, this->root);
		predictions.push_back(pred);
	}

	return predictions;
}


// growTree function: Grows a decision tree regression model using the given data and parameters //
Node* DecisionTreeRegression::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {

	// Define stopping criteria
	if (depth >= this->max_depth || y.size() <= this->min_samples_split) {
		return new Node(-1, -1.0, nullptr, nullptr, this->mean(y)); // Create a leaf node with the prediction value
	}



	int numFeatures = X[0].size();
	double bestMSE = DBL_MAX; // Initialize to a large value
	int split_idx = -1;  // Index of the best feature to split
	double split_thresh = 0.0;  // Best threshold to split on

	// Loop through candidate features
	for (int feature_idx = 0; feature_idx < numFeatures; ++feature_idx) {
		std::vector<double> featureValues;
		for (const auto& row : X) {
			featureValues.push_back(row[feature_idx]);
		}

		// Sort the feature values to find possible split points
		std::sort(featureValues.begin(), featureValues.end());

		// Loop through potential split thresholds (midpoints between consecutive feature values)
		for (int i = 1; i < featureValues.size(); ++i) {
			double threshold = (featureValues[i - 1] + featureValues[i]) / 2.0;

			double mse = this->meanSquaredError(y, X[i], threshold);

			// If this split is better, store it
			if (mse < bestMSE) {
				bestMSE = mse;
				split_idx = feature_idx;
				split_thresh = threshold;
			}
		}
	}

	// If no split improved the MSE, return a leaf node
	if (split_idx == -1) {
		return new Node(-1, -1.0, nullptr, nullptr, this->mean(y));
	}

	// Split the data on the best feature and threshold
	std::vector<std::vector<double>> left_X, right_X;
	std::vector<double> left_y, right_y;
	for (int j = 0; j < X.size(); ++j) {
		if (X[j][split_idx] <= split_thresh) {
			left_X.push_back(X[j]);
			left_y.push_back(y[j]);
		}
		else {
			right_X.push_back(X[j]);
			right_y.push_back(y[j]);
		}
	}

	if (left_y.empty() || right_y.empty()) {
		return new Node(-1, -1.0, nullptr, nullptr, this->mean(y)); // Ensure y is not empty
	}


	// Recursively grow the children nodes
	Node* left = growTree(left_X, left_y, depth + 1);
	Node* right = growTree(right_X, right_y, depth + 1);

	// Return the node with the best split
	return new Node(split_idx, split_thresh, left, right);

}


/// meanSquaredError function: Calculates the mean squared error for a given split threshold.
double DecisionTreeRegression::meanSquaredError(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {

	
	// Calculate the mse
	// TODO


	// Split the target vector `y` based on the feature column `X_column` and split threshold
	std::vector<double> left_y, right_y;

	for (int i = 0; i < X_column.size(); ++i) {
		if (X_column[i] <= split_thresh) {
			left_y.push_back(X_column[i]);
		}
		else {
			right_y.push_back(X_column[i]);
		}
	}

	double leftMean = this->mean(left_y);
	double rightMean = this->mean(right_y);

	double leftSE = 0.0, rightSE = 0.0;

	for (int i = 0; i < left_y.size(); i++) {
		leftSE += pow(left_y[i] - leftMean, 2);
	}

	for (int i = 0; i < right_y.size(); i++) {
		rightSE += pow(right_y[i] - rightMean, 2);
	}

	return (leftSE + rightSE) / X_column.size();
	
}

// mean function: Calculates the mean of a given vector of doubles.//
double DecisionTreeRegression::mean(std::vector<double>& values) {

	double meanValue = 0.0;


	for (const auto& value : values) {
		meanValue += value;
	}
	
	return meanValue /= values.size();
}

// traverseTree function: Traverses the decision tree and returns the predicted value for the given input vector.//
double DecisionTreeRegression::traverseTree(std::vector<double>& x, Node* node) {

	/* Implement the following:
		--- If the node is a leaf node, return its value
		--- If the feature value of the input vector is less than or equal to the node's threshold, traverse the left subtree
		--- Otherwise, traverse the right subtree
	*/
	// TODO

	if (node->isLeafNode()) return node->value;

		if (x[node->feature] <= node->threshold) {
			return this->traverseTree(x, node->left);
		}
		else {
			return this->traverseTree(x, node->right);
		}
	

	return 0.0;
}


/// runDecisionTreeRegression: this function runs the Decision Tree Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.

std::tuple<double, double, double, double, double, double,
	std::vector<double>, std::vector<double>,
	std::vector<double>, std::vector<double>>
	DecisionTreeRegression::runDecisionTreeRegression(const std::string& filePath, int trainingRatio) {
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
		// Load the dataset from the file path
		std::vector<std::vector<std::string>> data = DataLoader::readDatasetFromFilePath(filePath);

		// Convert the dataset from strings to doubles
		std::vector<std::vector<double>> dataset;
		bool isFirstRow = true; // Flag to identify the first row

		for (const auto& row : data) {
			if (isFirstRow) {
				isFirstRow = false;
				continue; // Skip the first row (header)
			}

			std::vector<double> convertedRow;
			for (const auto& cell : row) {
				try {
					double value = std::stod(cell);
					convertedRow.push_back(value);
				}
				catch (const std::exception& e) {
					// Handle the exception or set a default value
					std::cerr << "Error converting value: " << cell << std::endl;
					// You can choose to set a default value or handle the error as needed
				}
			}
			dataset.push_back(convertedRow);
		}

		// Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
		double trainRatio = trainingRatio * 0.01;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData, trainLabels);

		// Make predictions on the test data
		std::vector<double> testPredictions = predict(testData);

		// Calculate evaluation metrics (e.g., MAE, MSE)
		double test_mae = Metrics::meanAbsoluteError(testLabels, testPredictions);
		double test_rmse = Metrics::rootMeanSquaredError(testLabels, testPredictions);
		double test_rsquared = Metrics::rSquared(testLabels, testPredictions);

		// Make predictions on the training data
		std::vector<double> trainPredictions = predict(trainData);

		// Calculate evaluation metrics for training data
		double train_mae = Metrics::meanAbsoluteError(trainLabels, trainPredictions);
		double train_rmse = Metrics::rootMeanSquaredError(trainLabels, trainPredictions);
		double train_rsquared = Metrics::rSquared(trainLabels, trainPredictions);

		MessageBox::Show("Run completed");
		return std::make_tuple(test_mae, test_rmse, test_rsquared,
			train_mae, train_rmse, train_rsquared,
			std::move(trainLabels), std::move(trainPredictions),
			std::move(testLabels), std::move(testPredictions));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			std::vector<double>(), std::vector<double>(),
			std::vector<double>(), std::vector<double>());
	}
}

