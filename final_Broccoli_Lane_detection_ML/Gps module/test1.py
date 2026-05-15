import serial
import csv
import time

# Set Serial Port (Ensure it's correct)
SERIAL_PORT = "COM5"  # Your ESP8266 is connected here
BAUD_RATE = 9600
OUTPUT_FILE = "gps_path.csv"

# Open Serial Connection to ESP8266
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# Open CSV File for Saving Data
with open(OUTPUT_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Latitude", "Longitude"])  # CSV Headers

    print("\n🚀 Recording GPS Path... Press CTRL+C to stop.\n")

    try:
        while True:
            # Read raw data from ESP8266
            raw_data = ser.readline()
            print(f"📡 Raw Data: {raw_data}")  # Debug: Show raw serial data

            # Decode the data (handling errors)
            line = raw_data.decode("utf-8", errors="replace").strip()

            if line:
                try:
                    # Expecting: "Latitude: 23.456789 | Longitude: 72.678912"
                    lat, lon = line.split("|")
                    latitude = lat.split(":")[1].strip()
                    longitude = lon.split(":")[1].strip()

                    # Get current timestamp
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                    # Save to CSV
                    writer.writerow([timestamp, latitude, longitude])

                    # Print GPS Data on Console
                    print(f"✅ GPS Data → Timestamp: {timestamp}, Latitude: {latitude}, Longitude: {longitude}")

                except Exception as e:
                    print(f"❌ Error parsing line: {line} | {e}")

            time.sleep(1)  # Read GPS data every second

    except KeyboardInterrupt:
        print("\n🛑 GPS Recording Stopped. File saved as:", OUTPUT_FILE)

    finally:
        ser.close()
