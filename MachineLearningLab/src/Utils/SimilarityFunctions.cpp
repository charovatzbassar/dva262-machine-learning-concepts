#include "SimilarityFunctions.h"
#include <cmath>
#include <stdexcept>

// SimilarityFunctions class implementation //


/// hammingDistance function: Calculates the Hamming distance between two vectors.
double SimilarityFunctions::hammingDistance(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must be of equal length.");
    }
    double dist = 0.0;

    // Compute the Hamming Distance
    for (int i = 0; i < v1.size(); i++) {
        if (v1[i] != v2[i]) {
            dist = dist + 1.0;
        }
    }

    return dist;
}


/// jaccardDistance function: Calculates the Jaccard distance between two vectors.
double SimilarityFunctions::jaccardDistance(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of equal length.");
    }
    double num = 0.0;
    double den = 0.0;
    double dist = 0.0;

    // Compute the Jaccard Distance
    for (int i = 0; i < a.size(); i++) {
        if (a[i] != 0.0 || b[i] != 0.0) {
            den = den + 1.0;
            if (a[i] != 0.0 && b[i] != 0.0) {
                num = num + 1.0;
            }
        }
    }

    if (den == 0.0) {
        dist = 0.0; // Both vectors are zero vectors
    }
    else {
        dist = 1.0 - (num / den);
    }

    return dist;
}


/// cosineDistance function: Calculates the cosine distance between two vectors.
double SimilarityFunctions::cosineDistance(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of equal length.");
    }
    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    double cosinedist = 0.0;

    // Compute the cosine Distance
    for (int i = 0; i < a.size(); i++) {
        dotProduct = dotProduct + (a[i] * b[i]);
        normA = normA + (a[i] * a[i]);
        normB = normB + (b[i] * b[i]);
    }

    if (normA == 0.0 || normB == 0.0) {
        throw std::invalid_argument("One of the vectors is zero.");
    }

    cosinedist = 1.0 - (dotProduct / (sqrt(normA) * sqrt(normB)));

    return cosinedist;
}


/// euclideanDistance function: Calculates the Euclidean distance between two vectors.
double SimilarityFunctions::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of equal length.");
    }
    double dist = 0.0;

    // Compute the Euclidean Distance
    for (int i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        dist = dist + (diff * diff);
    }

    dist = sqrt(dist);

    return dist;
}


/// manhattanDistance function: Calculates the Manhattan distance between two vectors.
double SimilarityFunctions::manhattanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of equal length.");
    }
    double dist = 0.0;

    // Compute the Manhattan Distance
    for (int i = 0; i < a.size(); i++) {
        dist = dist + fabs(a[i] - b[i]);
    }

    return dist;
}

/// minkowskiDistance function: Calculates the Minkowski distance between two vectors.
double SimilarityFunctions::minkowskiDistance(const std::vector<double>& a, const std::vector<double>& b, int p) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of equal length.");
    }
    double dist = 0.0;

    // Compute the Minkowski Distance
    for (int i = 0; i < a.size(); i++) {
        dist = dist + pow(fabs(a[i] - b[i]), p);
    }

    dist = pow(dist, 1.0 / p);

    return dist;
}
