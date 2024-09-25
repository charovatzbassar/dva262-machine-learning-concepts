#include "EntropyFunctions.h"
#include <vector>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>


									// EntropyFunctions class implementation //



/// Calculates the entropy of a given set of labels "y".///
double EntropyFunctions::entropy(const std::vector<double>& y) {
    int total_samples = y.size();  // Total number of labels
    if (total_samples == 0) {
        return 0.0; // No samples, entropy is 0
    }
    
    // Step 1: Count occurrences of each label
    std::unordered_map<double, int> label_map;
    for (const double& label : y) {
        label_map[label]++;
    }

    // Step 2: Compute the probability of each label
    double entropy = 0.0;
    for (const auto& pair : label_map) {
        double label_count = pair.second;  // Count of the label
        double p = label_count / total_samples;  // Probability of the label

        // Step 3: Calculate entropy
        if (p > 0) {  // Prevent log(0)
            entropy -= p * std::log2(p);  // Shannon entropy formula
        }
    }

    return entropy;
}



/// Calculates the entropy of a given set of labels "y" and the indices of the labels "idxs".///
double EntropyFunctions::entropy(const std::vector<double>& y, const std::vector<int>& idxs) {
    int total_samples = idxs.size();  // Total number of selected samples
    if (total_samples == 0) {
        return 0.0; // No samples, entropy is 0
    }

    // Step 1: Count occurrences of each label in the selected indices
    std::unordered_map<double, int> label_map;
    for (const int& idx : idxs) {
        double label = y[idx];
        label_map[label]++;
    }

    // Step 2: Compute the probability of each label
    double entropy = 0.0;
    for (const auto& pair : label_map) {
        double label_count = pair.second;  // Count of the label
        double p = label_count / total_samples;  // Probability of the label

        // Step 3: Calculate entropy
        if (p > 0) {  // Prevent log(0)
            entropy -= p * std::log2(p);  // Shannon entropy formula
        }
    }

    return entropy;
}



