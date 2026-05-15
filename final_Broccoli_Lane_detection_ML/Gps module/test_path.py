import serial
import time
import folium
from flask import Flask, render_template_string

# Configure Serial Port (Modify for your ESP8266)
SERIAL_PORT = "COM5"  # Use "COMx" on Windows (e.g., "COM5")
BAUD_RATE = 115200

# Create Serial Connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

# Store GPS Path
gps_path = []

# Flask Web App
app = Flask(__name__)

# Webpage Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live GPS Tracker</title>
</head>
<body>
    <h2>Live GPS Tracking</h2>
    <iframe src="{{ map_url }}" width="100%" height="600px"></iframe>
    <meta http-equiv="refresh" content="5">
</body>
</html>
"""

def get_gps_data():
    """Read and parse GPS data from Serial"""
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if "," in line:
                lat, lng = map(float, line.split(","))
                print(f"📍 GPS: {lat}, {lng}")
                gps_path.append((lat, lng))
                if len(gps_path) > 50:  # Keep last 50 locations
                    gps_path.pop(0)
                return lat, lng
        except Exception as e:
            print(f"Error reading GPS: {e}")
            return None, None

def generate_map():
    """Generate and save an interactive map with the GPS path"""
    if not gps_path:
        return None

    lat, lng = gps_path[-1]  # Get latest location
    gps_map = folium.Map(location=[lat, lng], zoom_start=15)

    # Draw GPS path
    folium.PolyLine(gps_path, color="red", weight=3).add_to(gps_map)
    
    # Add Markers
    for point in gps_path:
        folium.Marker(location=point, icon=folium.Icon(color='blue')).add_to(gps_map)

    map_filename = "templates/map.html"
    gps_map.save(map_filename)
    return map_filename

@app.route("/")
def index():
    """Serve Webpage with Map"""
    generate_map()
    return render_template_string(HTML_TEMPLATE, map_url="/map.html")

@app.route("/map.html")
def map_page():
    """Serve Generated Map"""
    return open("templates/map.html").read()

if __name__ == "__main__":
    print("🌍 Starting GPS Tracker...")
    app.run(host="0.0.0.0", port=5000, debug=True)
