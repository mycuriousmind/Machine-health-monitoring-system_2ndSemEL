#include <Wire.h>
#include <Adafruit_MLX90614.h>

Adafruit_MLX90614 mlx = Adafruit_MLX90614();

#define ACS712_PIN 34
#define MOTOR_IN1  25
#define MOTOR_IN2  26
#define MOTOR_ENA  27

#define TEMP_THRESHOLD    45.0
#define CURRENT_THRESHOLD  3.0

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  mlx.begin();

  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);

  // New ESP32 v3.x LEDC API
  ledcAttach(MOTOR_ENA, 5000, 8);
  ledcWrite(MOTOR_ENA, 180);

  digitalWrite(MOTOR_IN1, HIGH);
  digitalWrite(MOTOR_IN2, LOW);

  Serial.println("System Started...");
}

void loop() {
  float temperature = mlx.readObjectTempC();

  int rawCurrent = analogRead(ACS712_PIN);
  float voltage = (rawCurrent / 4095.0) * 3.3;
  float current = abs((voltage - 1.65) / 0.185);

  Serial.print("Temp: ");
  Serial.print(temperature);
  Serial.print("°C | Current: ");
  Serial.println(current);

  if (temperature > TEMP_THRESHOLD || current > CURRENT_THRESHOLD) {
    Serial.println("FAULT DETECTED!");
  } else {
    Serial.println("Machine Healthy");
  }

  delay(1000);
}