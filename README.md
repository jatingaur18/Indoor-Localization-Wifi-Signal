# Indoor Localization with Deep Learning and Hybrid Models

## Overview

This project addresses **indoor localization** using WiFi RSSI (Received Signal Strength Indicator) data. The goal is to determine a device’s **environment (building and floor)** and its **location (longitude, latitude)**.

Two complementary approaches are implemented:

1. **Multi-task Deep Learning Model**

   * A single **Conv1D-based neural network** jointly predicts:

     * **Environment ID** (building + floor classification).
     * **Location** (longitude and latitude regression).

2. **Hybrid Model (Deep Learning + KNN)**

   * A CNN classifier predicts the **environment ID**.
   * For each environment, a **K-Nearest Neighbors (KNN) regressor** estimates the device’s location.

Both methods are trained and evaluated on WiFi fingerprint datasets.

---

## Dataset

* **Training Data:** `trainingData.csv`
* **Validation Data:** `validationData.csv`
* Each row contains:

  * **520 WiFi Access Point (WAP) RSSI values**.
  * **Building ID, Floor, Longitude, Latitude**.
* Preprocessing:

  * RSSI value `100` (no signal) is replaced with `-110 dBm`.
  * Features are standardized using `StandardScaler`.

---

## Multi-Task Deep Learning Model

### Architecture

* **Input:** 520 RSSI values reshaped to `(520, 1)`.
* **Layers:**

  * Conv1D → BatchNorm → MaxPooling (stacked 3 times).
  * Flatten → Dense(256, relu) → Dropout(0.4).
* **Outputs:**

  1. **Environment Classification (env\_id):**

     * Dense with softmax activation.
     * Loss: categorical crossentropy.
  2. **Location Regression (loc):**

     * Dense(128, relu) → Dropout → Dense(2, linear).
     * Loss: mean squared error (MSE).

### Training

* Optimizer: **Adam**.
* Loss Weights: `{env_id: 1.0, loc: 0.5}`.
* Callbacks:

  * **EarlyStopping** (patience=5, monitor validation accuracy).
  * **ReduceLROnPlateau** (reduce learning rate when validation accuracy plateaus).
* Batch Size: 32, Epochs: 50.

### Evaluation

* Environment accuracy (classification).
* Location mean absolute error (MAE).

### Example Results

```
Val Env Acc: 96.50%
Val Loc MAE: 4.12 meters
```

---

## Hybrid Model: CNN + KNN

### Step 1: Environment Classification

* CNN classifier (Conv1D → MaxPooling → Flatten → Dense).
* Output: softmax over all unique (building, floor) combinations.

### Step 2: Localization per Environment

* For each environment class, train a **KNN regressor** on location `(longitude, latitude)`.
* During inference:

  1. CNN predicts environment class.
  2. Corresponding KNN regressor estimates position.

### Example Results

```
Validation Environment Identification Accuracy: 95.20%
Validation Indoor Localization Mean Absolute Error: 5.02 meters
```

---

## Visualizations

1. **WiFi RSSI Heatmap**

   * Rows: samples.
   * Columns: WiFi Access Points.
   * Color intensity shows signal strength distribution.
   * Helps visualize sparsity and availability of WiFi signals.

2. **Indoor Position Distribution**

   * Scatter plot of longitude vs latitude.
   * Points colored by floor level.
   * Shows spatial spread of training samples.

Both plots are generated and saved as:

* `wifi_rssi_heatmap.png`
* `indoor_position_distribution.png`

---

## Applications

* Indoor navigation in shopping malls, airports, or campuses.
* Location-based services where GPS is unreliable.
* WiFi fingerprinting research and benchmarking.

