/*
 * esp32_datalogger.ino  -  logs raw MPU6050 acceleration as CSV over serial.
 *
 * Produces the SAME column format as your Phyphox exports, so the existing
 * Python pipeline (preprocess.py etc.) reads it with no changes:
 *     time,acc_x,acc_y,acc_z
 *
 * HOW TO COLLECT ONE RECORDING:
 *   1. Flash this sketch. Board = ESP32 Dev Module, Port = COM5.
 *   2. Open Serial Monitor @115200. You'll see a countdown, then CSV rows.
 *   3. Perform the condition (stay still / tap / load / etc.) for the duration.
 *   4. When it prints "=== DONE ===", SELECT ALL the CSV lines (from the
 *      'time,acc_x,...' header to the last data row), copy, and paste into a
 *      text file. Save as e.g.  esp_normal1.csv
 *   5. Press EN to record the next one.
 *
 * NOTE: Serial Monitor works but Serial is easiest to copy from. For cleaner
 * capture you can use PuTTY/CoolTerm to log straight to a file, but copy/paste
 * from Serial Monitor is fine for 45-60s recordings.
 */

#include <Wire.h>

const int   FS        = 256;         // sample rate (Hz) - matches TARGET_FS
const int   MPU_ADDR  = 0x68;
const float REC_SECONDS = 45.0;      // recording length per run
const long  N_SAMPLES = (long)(FS * REC_SECONDS);

void mpu_init() {
  Wire.begin(21, 22);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); Wire.write(0);   // wake
  Wire.endTransmission(true);
}

void mpu_read(float* ax, float* ay, float* az) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);
  int16_t rx = (Wire.read() << 8) | Wire.read();
  int16_t ry = (Wire.read() << 8) | Wire.read();
  int16_t rz = (Wire.read() << 8) | Wire.read();
  const float S = 16384.0f, G = 9.80665f;
  *ax = rx / S * G; *ay = ry / S * G; *az = rz / S * G;
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  mpu_init();

  Serial.println("\n=== ESP32 MPU6050 DATA LOGGER ===");
  Serial.print("Will record "); Serial.print(REC_SECONDS);
  Serial.print("s at "); Serial.print(FS); Serial.println("Hz.");
  Serial.println("Get ready to perform the condition...");
  for (int c = 3; c >= 1; c--) { Serial.print(c); Serial.print("... "); delay(1000); }
  Serial.println("GO\n");

  // CSV header (matches Phyphox column layout your pipeline expects)
  Serial.println("time,acc_x,acc_y,acc_z");

  const unsigned long period_us = 1000000UL / FS;
  float ax, ay, az;
  unsigned long start = micros();
  for (long i = 0; i < N_SAMPLES; i++) {
    unsigned long t0 = micros();
    mpu_read(&ax, &ay, &az);
    float t = (t0 - start) / 1e6f;             // seconds since start
    Serial.print(t, 5);   Serial.print(",");
    Serial.print(ax, 5);  Serial.print(",");
    Serial.print(ay, 5);  Serial.print(",");
    Serial.println(az, 5);
    while (micros() - t0 < period_us) { /* hold rate */ }
  }
  Serial.println("=== DONE === copy the rows above into a .csv, then press EN for next.");
}

void loop() {}
