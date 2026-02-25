# Hardware Connection Guide

This document describes the physical connections for the "Online Adaptive Cyber-Physical Control System".

## Component List
1. **Microcontroller**: ESP32 (e.g., NodeMCU-32S)
2. **Sensor**: 10k Potentiometer (for simulation of plant state) or MPU6050 Accelerometer.
3. **Actuator**: DC Motor with L298N Driver or an LED (for PWM monitoring).
4. **Power Supply**: 5V/3.3V as per ESP32 requirements.

## Wiring Diagram

| Component | ESP32 Pin | Description |
| :--- | :--- | :--- |
| **Potentiometer VCC** | 3.3V | Power |
| **Potentiometer GND** | GND | Ground |
| **Potentiometer Signal**| GPIO 34 | Analog Input (Sensor) |
| **Actuator (LED/Motor)**| GPIO 25 | PWM Output (Control Signal) |

## Setup Instructions
1. Connect the Potentiometer center pin to GPIO 34.
2. Connect an LED (with a 220-ohm resistor) or motor driver input to GPIO 25.
3. Ensure the ESP32 and the Backend computer are on the same Wi-Fi network.
4. Update the `ssid`, `password`, and `serverUrl` in `esp32_firmware.ino`.
5. Flash the code using Arduino IDE or VS Code PlatformIO.
