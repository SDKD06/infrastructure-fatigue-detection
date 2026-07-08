/*
 * esp32_validation.ino  -  proves the on-chip Isolation Forest is FAITHFUL
 * and measures inference latency. Run this ONCE on the ESP32; no sensor needed.
 *
 * STEP 1 (on your laptop): dump your 5 reference feature vectors with:
 *   python -c "import numpy as np; d=np.load('data/features.npz'); v=d['features'][:5]; \
 *   print('{'); [print(' {'+','.join(f'{x:.8f}f' for x in r)+'},') for r in v]; print('};')"
 *   ...and also print the Python scores to compare against:
 *   python -c "import joblib,numpy as np; d=np.load('data/features.npz'); \
 *   m=joblib.load('models/isolation_forest_model.pkl')['model']; print(m.score_samples(d['features'][:5]))"
 *
 * STEP 2: paste the array into REF_VECTORS below.
 * STEP 3: flash, open Serial Monitor @115200. Compare each printed score to the
 *         Python values (should match to ~1e-5) and note the microseconds/inference.
 */

#include "micro_forest.h"   // your generated header (defines N_FEATURES, predict_anomaly)

// <<< PASTE your dumped 5x30 array here (replace these zeros) >>>
float REF_VECTORS[5][N_FEATURES] = {
  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
};

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== ESP32 Isolation-Forest VALIDATION ===");
  Serial.println("Compare these scores to Python score_samples (match ~1e-5):\n");

  for (int i = 0; i < 5; i++) {
    float score = 0.0f;
    // time the inference
    unsigned long t0 = micros();
    bool anomaly = predict_anomaly(REF_VECTORS[i], &score);
    unsigned long dt = micros() - t0;

    Serial.print("vector "); Serial.print(i);
    Serial.print("  score = "); Serial.print(score, 6);
    Serial.print("  ["); Serial.print(anomaly ? "ANOMALY" : "normal"); Serial.print("]");
    Serial.print("  latency = "); Serial.print(dt); Serial.println(" us");
  }

  // average latency over many runs for a stable paper number
  const int N = 200;
  float s; unsigned long t0 = micros();
  for (int k = 0; k < N; k++) predict_anomaly(REF_VECTORS[0], &s);
  unsigned long avg = (micros() - t0) / N;
  Serial.print("\nAvg inference latency over ");
  Serial.print(N); Serial.print(" runs: "); Serial.print(avg); Serial.println(" us");
  Serial.println("Report this latency + the flash/RAM from the compile output.");
}

void loop() {}
