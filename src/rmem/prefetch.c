/*
 * prefetch.c - dedicated prefetcher implementations
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "rmem/prefetch.h"

// XGBoost C API
#include <xgboost/c_api.h>

// Feature indices based on the training script
#define FEATURE_PC 0
#define FEATURE_OFFSET 1  
#define FEATURE_DELTA 2
#define FEATURE_OFFSET_FROM_FAULTING 3
#define NUM_FEATURES 4

// Model configuration (from training script)
#define N_ESTIMATORS 50
#define MAX_DEPTH 6

typedef struct {
    BoosterHandle booster;
    int is_loaded;
} XGBoostModel;

static XGBoostModel* global_model = NULL;

typedef struct {
    uint32_t pc;
    int offset;
    float delta;
    int offset_from_faulting;
} FeatureVector;

typedef struct {
    int cache_hit;
    int cache_miss; 
    int misprefetch;
    double precision;
    double recall;
    double accuracy;
} PredictionMetrics;

/**
 * Initialize XGBoost model structure
 */
XGBoostModel* init_model() {
    XGBoostModel* model = (XGBoostModel*)malloc(sizeof(XGBoostModel));
    if (!model) {
        fprintf(stderr, "Failed to allocate memory for model\n");
        return NULL;
    }
    
    model->booster = NULL;
    model->is_loaded = 0;
    return model;
}

/**
 * Load trained XGBoost model from file
 */
int load_model(XGBoostModel* model, const char* model_path) {
    if (!model || !model_path) {
        fprintf(stderr, "Invalid model or path\n");
        return -1;
    }
    
    // First check if file exists and is readable
    FILE* test_file = fopen(model_path, "r");
    if (!test_file) {
        fprintf(stderr, "Error: Cannot open model file '%s'\n", model_path);
        perror("File error");
        return -1;
    }
    fclose(test_file);
    printf("Model file exists and is readable: %s\n", model_path);
    
    printf("Creating booster\n");
    // Create booster
    if (XGBoosterCreate(NULL, 0, &model->booster) != 0) {
        fprintf(stderr, "Failed to create XGBooster\n");
        const char* error_msg = XGBGetLastError();
        if (error_msg) {
            fprintf(stderr, "XGBoost error: %s\n", error_msg);
        }
        return -1;
    }
    
    // Set objective to avoid plugin loading issues in shared library
    if (XGBoosterSetParam(model->booster, "objective", "binary:logistic") != 0) {
        fprintf(stderr, "Failed to set objective parameter\n");
        const char* error_msg = XGBGetLastError();
        if (error_msg) {
            fprintf(stderr, "XGBoost error: %s\n", error_msg);
        }
    }
    
    printf("Loading model (config: {\"validate_parameters\": \"0\"})\n");
    // Disable parameter validation to avoid objective function check
    XGBoosterSetParam(model->booster, "validate_parameters", "0");
    
    // Load model from file
    if (XGBoosterLoadModel(model->booster, model_path) != 0) {
        printf("Loading failed\n");
        const char* error_msg = XGBGetLastError();
        if (error_msg) {
            fprintf(stderr, "XGBoost error: %s\n", error_msg);
        }
        fprintf(stderr, "Failed to load model from %s\n", model_path);
        XGBoosterFree(model->booster);
        model->booster = NULL;
        return -1;
    }
    
    model->is_loaded = 1;
    printf("Successfully loaded XGBoost model from %s\n", model_path);
    return 0;
}

/**
 * Preprocess features according to training script logic
 */
void preprocess_features(FeatureVector* features) {
    // Convert PC to categorical (in training, PC was converted to category)
    // For inference, we keep PC as-is since XGBoost handles categorical features
    
    // Scale delta by 4096 (as done in training)
    features->delta = features->delta / 4096.0f;
}

/**
 * Make prediction for a single sample
 */
int predict_single(XGBoostModel* model, FeatureVector* features, float* prediction) {
    if (!model || !model->is_loaded || !features || !prediction) {
        fprintf(stderr, "Invalid input parameters\n");
        return -1;
    }
    
    // Preprocess features
    preprocess_features(features);
    
    // Prepare feature array
    float feature_array[NUM_FEATURES];
    feature_array[FEATURE_PC] = (float)features->pc;
    feature_array[FEATURE_OFFSET] = (float)features->offset;
    feature_array[FEATURE_DELTA] = features->delta;
    feature_array[FEATURE_OFFSET_FROM_FAULTING] = (float)features->offset_from_faulting;
    
    // Create DMatrix for single prediction
    DMatrixHandle dtest;
    if (XGDMatrixCreateFromMat(feature_array, 1, NUM_FEATURES, -1, &dtest) != 0) {
        fprintf(stderr, "Failed to create DMatrix\n");
        return -1;
    }
    
    // Make prediction
    bst_ulong out_len;
    const float* out_result;
    
    if (XGBoosterPredict(model->booster, dtest, 0, 0, 0, &out_len, &out_result) != 0) {
        fprintf(stderr, "Failed to make prediction\n");
        XGDMatrixFree(dtest);
        return -1;
    }
    
    if (out_len != 1) {
        fprintf(stderr, "Unexpected prediction output length: %lu\n", out_len);
        XGDMatrixFree(dtest);
        return -1;
    }
    
    *prediction = out_result[0];
    
    // Clean up
    XGDMatrixFree(dtest);
    return 0;
}

/**
 * Make predictions for batch of samples
 */
int predict_batch(XGBoostModel* model, FeatureVector* features_batch, 
                  int batch_size, float* predictions) {
    if (!model || !model->is_loaded || !features_batch || !predictions || batch_size <= 0) {
        fprintf(stderr, "Invalid input parameters\n");
        return -1;
    }
    
    // Prepare feature matrix
    float* feature_matrix = (float*)malloc(batch_size * NUM_FEATURES * sizeof(float));
    if (!feature_matrix) {
        fprintf(stderr, "Failed to allocate memory for feature matrix\n");
        return -1;
    }
    
    // Fill feature matrix and preprocess
    for (int i = 0; i < batch_size; i++) {
        FeatureVector temp_features = features_batch[i];
        preprocess_features(&temp_features);
        
        int base_idx = i * NUM_FEATURES;
        feature_matrix[base_idx + FEATURE_PC] = (float)temp_features.pc;
        feature_matrix[base_idx + FEATURE_OFFSET] = (float)temp_features.offset;
        feature_matrix[base_idx + FEATURE_DELTA] = temp_features.delta;
        feature_matrix[base_idx + FEATURE_OFFSET_FROM_FAULTING] = (float)temp_features.offset_from_faulting;
    }
    
    // Create DMatrix
    DMatrixHandle dtest;
    if (XGDMatrixCreateFromMat(feature_matrix, batch_size, NUM_FEATURES, -1, &dtest) != 0) {
        fprintf(stderr, "Failed to create DMatrix for batch\n");
        free(feature_matrix);
        return -1;
    }
    
    // Make predictions
    bst_ulong out_len;
    const float* out_result;
    
    if (XGBoosterPredict(model->booster, dtest, 0, 0, 0, &out_len, &out_result) != 0) {
        fprintf(stderr, "Failed to make batch predictions\n");
        XGDMatrixFree(dtest);
        free(feature_matrix);
        return -1;
    }
    
    if (out_len != (bst_ulong)batch_size) {
        fprintf(stderr, "Unexpected batch prediction output length: %lu (expected %d)\n", 
                out_len, batch_size);
        XGDMatrixFree(dtest);
        free(feature_matrix);
        return -1;
    }
    
    // Copy results
    memcpy(predictions, out_result, batch_size * sizeof(float));
    
    // Clean up
    XGDMatrixFree(dtest);
    free(feature_matrix);
    return 0;
}

/**
 * Convert probability to binary prediction (threshold = 0.5)
 */
int probability_to_prediction(float probability) {
    return (probability >= 0.5) ? 1 : 0;
}

/**
 * Calculate prediction metrics
 */
void calculate_metrics(int* predictions, int* actual_labels, int count, 
                      PredictionMetrics* metrics) {
    if (!predictions || !actual_labels || !metrics || count <= 0) {
        return;
    }
    
    metrics->cache_hit = 0;
    metrics->cache_miss = 0;
    metrics->misprefetch = 0;
    
    for (int i = 0; i < count; i++) {
        if (actual_labels[i] == 1 && predictions[i] == 1) {
            metrics->cache_hit++;
        } else if (actual_labels[i] == 1 && predictions[i] == 0) {
            metrics->cache_miss++;
        } else if (actual_labels[i] == 0 && predictions[i] == 1) {
            metrics->misprefetch++;
        }
    }
    
    // Calculate precision: cache_hits / (cache_hits + misprefetch)
    int total_predicted_positive = metrics->cache_hit + metrics->misprefetch;
    metrics->precision = (total_predicted_positive > 0) ? 
                        (double)metrics->cache_hit / total_predicted_positive : 0.0;
    
    // Calculate recall: cache_hits / (cache_hits + cache_miss)
    int total_actual_positive = metrics->cache_hit + metrics->cache_miss;
    metrics->recall = (total_actual_positive > 0) ? 
                     (double)metrics->cache_hit / total_actual_positive : 0.0;
    
    // Calculate accuracy
    int correct_predictions = 0;
    for (int i = 0; i < count; i++) {
        if (predictions[i] == actual_labels[i]) {
            correct_predictions++;
        }
    }
    metrics->accuracy = (double)correct_predictions / count;
}

/**
 * Free model resources
 */
void free_model(XGBoostModel* model) {
    if (model) {
        if (model->booster) {
            XGBoosterFree(model->booster);
        }
        free(model);
    }
}


void init_prefetcher() {
    const char* model_path = "/data1/deku/models/random-ll-fltrace_xgboost_model.json";
    printf("Initializing prefetcher...\n");
    
    global_model = init_model();
    if (!global_model) {
        printf("Model not initialized\n");
        return;
    }
    printf("Model structure created\n");
    
    if (load_model(global_model, model_path) != 0) {
        printf("Model loading failed\n");
        free_model(global_model);
        global_model = NULL;
        return;
    }
    printf("Prefetcher initialized successfully\n");
}

/*
 * This function is run before the page is fetched.
 * It can do prefetching based on non-page content related
 * information.
 */
unsigned long page_prefetch() {

}

/*
 * This function allows fetching pages after the page 
 * contents are accessible. It fetches additional pages
 * after the currently faulted page is already fetched.
 */
unsigned long page_postfetch() {
    printf("Post prefetch called\n");
    
    if (!global_model || !global_model->is_loaded) {
        printf("Model not initialized. Call init_prefetcher() first.\n");
        return 1;
    }
    
    // Example: Single prediction
    printf("\n=== Single Prediction Example ===\n");
    FeatureVector test_features = {
        .pc = 0x50c9f0,
        .offset = 0,
        .delta = 1024.0f,
        .offset_from_faulting = 64
    };
    
    float prediction_prob;
    if (predict_single(global_model, &test_features, &prediction_prob) == 0) {
        int prediction_binary = probability_to_prediction(prediction_prob);
        printf("Input: PC=0x%x, Offset=%d, Delta=%.2f, OffsetFromFaulting=%d\n",
               test_features.pc, test_features.offset, test_features.delta, 
               test_features.offset_from_faulting);
        printf("Prediction probability: %.6f\n", prediction_prob);
        printf("Binary prediction: %d (%s)\n", prediction_binary, 
               prediction_binary ? "Cache Hit" : "Cache Miss");
    } else {
        fprintf(stderr, "Single prediction failed\n");
    }
    
    // Example: Batch prediction
    printf("\n=== Batch Prediction Example ===\n");
    const int batch_size = 5;
    FeatureVector batch_features[5] = {
        {0x50c9f0, 0, 1024.0f, 64},
        {0x50c9f0, 16, 2048.0f, 32},
        {0x50c9f0, -8, 512.0f, 128},
        {0x50c9f0, 32, 4096.0f, 16},
        {0x50c9f0, 8, 256.0f, 256}
    };
    
    float batch_predictions[5];
    if (predict_batch(global_model, batch_features, batch_size, batch_predictions) == 0) {
        printf("Batch predictions:\n");
        for (int i = 0; i < batch_size; i++) {
            int binary_pred = probability_to_prediction(batch_predictions[i]);
            printf("  Sample %d: %.6f -> %d (%s)\n", i, batch_predictions[i], 
                   binary_pred, binary_pred ? "Cache Hit" : "Cache Miss");
        }
    } else {
        fprintf(stderr, "Batch prediction failed\n");
    }
    
    printf("\nInference completed successfully!\n");
    return 0;
}


/**
 * Example usage and testing function
 */
/*int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model_file>\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    
    // Initialize model
    XGBoostModel* model = init_model();
    if (!model) {
        return 1;
    }
    
    // Load model
    if (load_model(model, model_path) != 0) {
        free_model(model);
        return 1;
    }
    
    // Example: Single prediction
    printf("\n=== Single Prediction Example ===\n");
    FeatureVector test_features = {
        .pc = 0x50c9f0,
        .offset = 0,
        .delta = 1024.0f,
        .offset_from_faulting = 64
    };
    
    float prediction_prob;
    if (predict_single(model, &test_features, &prediction_prob) == 0) {
        int prediction_binary = probability_to_prediction(prediction_prob);
        printf("Input: PC=0x%x, Offset=%d, Delta=%.2f, OffsetFromFaulting=%d\n",
               test_features.pc, test_features.offset, test_features.delta, 
               test_features.offset_from_faulting);
        printf("Prediction probability: %.6f\n", prediction_prob);
        printf("Binary prediction: %d (%s)\n", prediction_binary, 
               prediction_binary ? "Cache Hit" : "Cache Miss");
    } else {
        fprintf(stderr, "Single prediction failed\n");
    }
    
    // Example: Batch prediction
    printf("\n=== Batch Prediction Example ===\n");
    const int batch_size = 5;
    FeatureVector batch_features[5] = {
        {0x50c9f0, 0, 1024.0f, 64},
        {0x50c9f0, 16, 2048.0f, 32},
        {0x50c9f0, -8, 512.0f, 128},
        {0x50c9f0, 32, 4096.0f, 16},
        {0x50c9f0, 8, 256.0f, 256}
    };
    
    float batch_predictions[5];
    if (predict_batch(model, batch_features, batch_size, batch_predictions) == 0) {
        printf("Batch predictions:\n");
        for (int i = 0; i < batch_size; i++) {
            int binary_pred = probability_to_prediction(batch_predictions[i]);
            printf("  Sample %d: %.6f -> %d (%s)\n", i, batch_predictions[i], 
                   binary_pred, binary_pred ? "Cache Hit" : "Cache Miss");
        }
    } else {
        fprintf(stderr, "Batch prediction failed\n");
    }
    
    // Clean up
    free_model(model);
    
    printf("\nInference completed successfully!\n");
    return 0;
} */

/*
 * Compilation instructions:
 * 
 * 1. Install XGBoost with C API support
 * 2. Compile with:
 *    gcc -o xgboost_inference xgboost_inference.c -lxgboost -lm
 * 
 * 3. Run with:
 *    ./xgboost_inference path/to/your/model.json
 * 
 * Note: Make sure the XGBoost C library is in your library path
 */
