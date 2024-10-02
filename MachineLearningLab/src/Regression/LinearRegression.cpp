#include "LinearRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include <cmath>
#include <string>
#include <algorithm>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <unordered_map>
#include <msclr\marshal_cppstd.h>
#include <stdexcept>
#include "../MainForm.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
using namespace System::Windows::Forms; // For MessageBox



										///  LinearRegression class implementation  ///


// Function to fit the linear regression model to the training data //
void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels) {

    // 1. Check if the sizes of trainData and trainLabels match
    if (trainData.size() != trainLabels.size()) {
        throw std::invalid_argument("The number of training samples and labels must be the same.");
    }

    size_t num_samples = trainData.size();      // Number of training examples
    size_t num_features = trainData[0].size(); // Number of features

    // 2. Convert trainData to Eigen matrix and construct design matrix X with intercept
    Eigen::MatrixXd X(num_samples, num_features + 1); // +1 for the intercept term

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            X(i, j) = trainData[i][j];
        }
        X(i, num_features) = 1.0; // Adding intercept term (bias)
    }

    // 3. Convert trainLabels to Eigen vector y
    Eigen::VectorXd y(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        y(i) = trainLabels[i];
    }

    m_coefficients =  X.colPivHouseholderQr().solve(y);
}


// Function to fit the linear regression model using Gradient Descent
// Function to fit the linear regression model using Gradient Descent
void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels,
    double learning_rate, int num_epochs) {

    // Check if the sizes of trainData and trainLabels match
    if (trainData.size() != trainLabels.size()) {
        throw std::invalid_argument("The number of training samples and labels must be the same.");
    }

    size_t num_samples = trainData.size();
    size_t num_features = trainData[0].size();

    // Convert trainData to Eigen matrix and add intercept column
    Eigen::MatrixXd X(num_samples, num_features + 1); // +1 for the intercept term
    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            X(i, j) = trainData[i][j];
        }
        X(i, num_features) = 1.0; // Adding intercept term (bias)
    }

    // Convert trainLabels to Eigen vector y
    Eigen::VectorXd y(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        y(i) = trainLabels[i];
    }

    // Initialize coefficients (theta) to small random values
    m_coefficients = Eigen::VectorXd::Zero(num_features + 1);

    // Gradient Descent loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Compute predictions: y_pred = X * coefficients
        Eigen::VectorXd predictions = X * m_coefficients;

        // Calculate the error: error = y_pred - y
        Eigen::VectorXd error = predictions - y;

        // Compute the gradient: gradient = (1/m) * (X^T * error)
        Eigen::VectorXd gradient = (X.transpose() * error) / num_samples;

        // Update coefficients: theta := theta - learning_rate * gradient
        m_coefficients = m_coefficients - learning_rate * gradient;
    }
}



// Function to make predictions on new data //
std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData) {

    // 1. Check if the model has been fitted by verifying if coefficients are set
    if (m_coefficients.size() == 0) {
        throw std::runtime_error("The model has not been fitted yet. Please call fit() before predict().");
    }

    size_t num_samples = testData.size();      // Number of test examples
    size_t num_features = testData[0].size(); // Number of features

    // 2. Convert testData to Eigen matrix and construct design matrix X with intercept
    Eigen::MatrixXd X_test(num_samples, num_features + 1); // +1 for the intercept term

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            X_test(i, j) = testData[i][j];
        }
        X_test(i, num_features) = 1.0; // Adding intercept term (bias)
    }

    // 3. Make predictions: y_pred = X_test * θ
    Eigen::VectorXd predictions = X_test * m_coefficients;

    // 4. Convert Eigen vector to std::vector<double>
    std::vector<double> result(predictions.data(), predictions.data() + predictions.size());

    return result;
}


/// runLinearRegression: this function runs the Linear Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets. ///

std::tuple<double, double, double, double, double, double,
    std::vector<double>, std::vector<double>,
    std::vector<double>, std::vector<double>>
    LinearRegression::runLinearRegression(const std::string& filePath, int trainingRatio) {
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

        // Manually set the learning rate (alpha) and number of epochs
        double learning_rate = 0.000001;
        int num_epochs = 1000;
        // Choose the model to fit the training data
        fit(trainData, trainLabels);
        //fit(trainData, trainLabels, learning_rate, num_epochs);

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