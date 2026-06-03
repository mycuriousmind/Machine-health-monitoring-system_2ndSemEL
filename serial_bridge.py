import serial
import json
import os
import time
import argparse

def get_shared_data():
    if os.path.exists("shared_data.json"):
        try:
            with open("shared_data.json", "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def main():
    parser = argparse.ArgumentParser(description="Bridge between ESP32 and AI Dashboard")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (must match ESP32)")
    args = parser.parse_args()

    print(f"--- Serial Bridge Started ---")
    print(f"Listening on {args.port} at {args.baud} baud.")
    
    try:
        ser = serial.Serial(args.port, args.baud, timeout=2)
    except Exception as e:
        print(f"Error opening serial port: {e}")
        print("Please check the port name and ensure the ESP32 is connected.")
        print("You can run this script with: python serial_bridge.py --port COM4")
        return

    vibration_buffer = [[0.0, 0.0, 0.0] for _ in range(100)]

    while True:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"ESP32: {line}")
                    
                    # Format 1: "Temp: 34.5°C | Current: 2.1"
                    if "Temp:" in line and "Current:" in line:
                        parts = line.split("|")
                        temp_str = parts[0].replace("Temp:", "").replace("°C", "").strip()
                        curr_str = parts[1].replace("Current:", "").strip()
                        
                        try:
                            temp = float(temp_str)
                            curr = float(curr_str)
                            
                            shared = get_shared_data()
                            shared["esp_data"] = {
                                "Temperature": temp,
                                "Current": curr,
                                "Timestamp": time.time()
                            }
                            
                            with open("shared_data.json", "w") as f:
                                json.dump(shared, f)
                                
                        except ValueError:
                            pass
                    
                    # Format 2: "ax,ay,az,temp,current" (comma-separated values from MPU6050/ACS712 code)
                    else:
                        parts = line.split(",")
                        if len(parts) == 5:
                            try:
                                ax = float(parts[0])
                                ay = float(parts[1])
                                az = float(parts[2])
                                temp = float(parts[3])
                                curr = float(parts[4])
                                
                                vibration_buffer.pop(0)
                                vibration_buffer.append([ax, ay, az])
                                
                                shared = get_shared_data()
                                shared["esp_data"] = {
                                    "Temperature": temp,
                                    "Current": curr,
                                    "VibrationBuffer": list(vibration_buffer),
                                    "Timestamp": time.time()
                                }
                                
                                with open("shared_data.json", "w") as f:
                                    json.dump(shared, f)
                                    
                            except ValueError:
                                pass
        except Exception as e:
            print(f"Error reading from serial: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
