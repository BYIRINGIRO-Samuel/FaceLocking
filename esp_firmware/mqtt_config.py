# esp_firmware/mqtt_config.py
# Flash this alongside main.py onto the ESP8266.
# DO NOT commit this file with real credentials to a public repo.

TEAM = "team^_^TopDog"          # reuse or rename as you like -- used to namespace topics

WIFI_SSID = "Y3A"
WIFI_PASS = "RCA@2024"

MQTT_BROKER = "broker.benax.rw"
MQTT_PORT = 1883

# Servo calibration -- measured from esp_firmware/calibrate.py on your hardware.
# Sweep showed real range ~20 (min) to ~130 (max); using a small safety margin
# so normal operation never pushes the servo into its physical hard stop.
PAN_MIN_DUTY = 22
PAN_MAX_DUTY = 128

PAN_PIN = 14  # D5 on NodeMCU/Wemos D1 mini = GPIO14 (matches your wiring)

# Safe mechanical limits (degrees) -- clamp here so the servo never over-rotates
PAN_MIN_ANGLE = 10
PAN_MAX_ANGLE = 170

HOME_PAN = 90

# MQTT topics (mirrors the Falcon Eye reference doc's naming)
T_CMD = b"falcon/eye/servo/cmd"        # Python -> board: JSON {"pan": N}
T_STATUS = b"falcon/eye/servo/status"  # board -> Python: current angle state