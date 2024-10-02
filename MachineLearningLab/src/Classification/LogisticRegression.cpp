#include "LogisticRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <string>
#include <vector>
#include <utility>
#include <set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include <unordered_map> 

using namespace System::Windows::Forms; // For MessageBox


                                            ///  LogisticRegression class implementation  ///
// Constractor

LogisticRegression::LogisticRegression(double learning_rate, int num_epochs)
    : learning_rate(learning_rate), num_epochs(num_epochs) {}

// Fit method for training the logistic regression model
void LogisticRegression::fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train) {
    int num_features = X_train[0].size();
    int num_classes = std::set<double>(y_train.begin(), y_train.end()).size();

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-0.01, 0.01);

    this->weights = std::vector<std::vector<double>>(num_classes, std::vector<double>(num_features + 1)); 
    for (auto& weight_vec : this->weights) {
        for (auto& weight : weight_vec) {
            weight = distribution(generator);
        }
    }


    for (int c = 0; c < num_classes; c++) {
        std::vector<double> binary_labels(y_train.size());

        std::transform(y_train.begin(), y_train.end(), binary_labels.begin(),
            [c](double label) { return label == c ? 1.0 : 0.0; });

        for (int epoch = 0; epoch < this->num_epochs; epoch++) {
            for (int i = 0; i < X_train.size(); i++) {
                // Add bias term (1) to the input vector
                std::vector<double> x_with_bias = X_train[i];
                x_with_bias.insert(x_with_bias.begin(), 1.0); // Add 1 at the beginning for bias

                // Compute the weighted sum and the sigmoid

                double z = this->weights[c][0];

                for (int i = 0; i < X_train[i].size(); ++i) {
                    z += this->weights[c][i + 1] * X_train[i][i]; // Weighted sum
                }

                double predicted = this->sigmoid(z);

                // Update weights using gradient descent
                for (int j = 0; j < this->weights[c].size(); j++) {
                    this->weights[c][j] -= this->learning_rate * (predicted - binary_labels[i]) * x_with_bias[j];
                }
            }
        }
    }

}

// Predict method to predict class labels for test data
std::vector<double> LogisticRegression::predict(const std::vector<std::vector<double>>& X_test) {
    std::vector<double> predictions;

    for (const auto& testPoint : X_test) {
        std::vector<double> x_with_bias = testPoint;
        x_with_bias.insert(x_with_bias.begin(), 1.0);

        std::vector<double> class_scores(weights.size(), 0.0);

        // Compute the score for each class
        for (int c = 0; c < weights.size(); c++) {
            double score = weights[c][0]; // Bias term
            for (size_t i = 1; i < weights[c].size(); ++i) {
                score += weights[c][i] * x_with_bias[i];
            }
            class_scores[c] = score; // Store score for the class
        }

        // Predict the class with the highest score
        auto max_score_it = std::max_element(class_scores.begin(), class_scores.end());
        int predicted_class = std::distance(class_scores.begin(), max_score_it);
        predictions.push_back(predicted_class);
    }
    
    return predictions;
}

/// runLogisticRegression: this function runs the logistic regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> 
LogisticRegression::runLogisticRegression(const std::string& filePath, int trainingRatio) {

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
        fit(trainData, trainLabels);

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