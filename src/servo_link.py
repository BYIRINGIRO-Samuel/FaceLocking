# src/servo_link.py
"""
Phase 3 sanity check: confirm Python can talk to the ESP8266 servo
controller over MQTT, completely independent of any camera code.

Run:
    python -m src.servo_link

Expected: the turret should visibly move to each test angle in sequence.
"""
import time
import json

import paho.mqtt.client as mqtt

MQTT_BROKER = "broker.benax.rw"
MQTT_PORT = 1883
T_CMD = "falcon/eye/servo/cmd"
T_STATUS = "falcon/eye/servo/status"


def on_connect(client, userdata, flags, rc):
    print(f"[servo_link] connected (rc={rc})")
    client.subscribe(T_STATUS)


def on_message(client, userdata, msg):
    print(f"[servo_link] status: {msg.payload.decode()}")


def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[servo_link] connecting to {MQTT_BROKER}:{MQTT_PORT} ...")
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_start()

    time.sleep(1.5)  # let the connection settle

    test_sequence = [
        {"pan": 90},    # center
        {"pan": 45},    # left
        {"pan": 135},   # right
        {"pan": 90},    # back to center
        {"home": True}, # explicit home
    ]

    for cmd in test_sequence:
        payload = json.dumps(cmd)
        print(f"[servo_link] publishing: {payload}")
        client.publish(T_CMD, payload)
        time.sleep(2.0)  # give the servo time to physically move

    print("[servo_link] test sequence done.")
    time.sleep(1.0)
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()