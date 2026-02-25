/*
  Online Adaptive Cyber-Physical Control System - ESP32 Firmware
  Handles real-time sensor data acquisition and actuator control.
  Communication: Wi-Fi HTTP POST to Python Backend.
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

const char* ssid = "YOUR_SSID";
const char* password = "YOUR_PASSWORD";
const char* serverUrl = "http://YOUR_BACKEND_IP:5000/update";

const int SENSOR_PIN = 34; // Analog input (e.g., potentiometer)
const int ACTUATOR_PIN = 25; // PWM output (e.g., LED or Motor)

void setup() {
  Serial.begin(115200);
  pinMode(SENSOR_PIN, INPUT);
  ledcSetup(0, 5000, 8); // Channel 0, 5kHz, 8-bit resolution
  ledcAttachPin(ACTUATOR_PIN, 0);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected");
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    int sensorValue = analogRead(SENSOR_PIN);
    float y_actual = (float)sensorValue * (3.3 / 4095.0); // Convert to voltage
    float y_ref = 1.65; // Target setpoint (half of 3.3V)

    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    StaticJsonDocument<200> doc;
    doc["y_ref"] = y_ref;
    doc["y_actual"] = y_actual;
    doc["dt"] = 0.1;

    String requestBody;
    serializeJson(doc, requestBody);

    int httpResponseCode = http.POST(requestBody);

    if (httpResponseCode > 0) {
      String response = http.getString();
      StaticJsonDocument<200> respDoc;
      deserializeJson(respDoc, response);

      float u_adaptive = respDoc["u_adaptive"];
      
      // Map u_adaptive (-100 to 100) to PWM (0 to 255)
      int pwmValue = map((int)u_adaptive, -100, 100, 0, 255);
      pwmValue = constrain(pwmValue, 0, 255);
      
      ledcWrite(0, pwmValue);
      
      Serial.print("Y_Actual: "); Serial.print(y_actual);
      Serial.print(" | U_Adaptive: "); Serial.println(u_adaptive);
    } else {
      Serial.print("Error in HTTP request: ");
      Serial.println(httpResponseCode);
    }
    http.end();
  }
  delay(100); // 10Hz control loop
}
