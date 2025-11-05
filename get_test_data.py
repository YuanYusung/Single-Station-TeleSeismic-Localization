# -*- coding: utf-8 -*-
"""
Download and process seismic data for testing purposes.
This script retrieves the seismic data and instrument response from IRIS,
preprocesses the data (detrend, filter, remove response), and saves it in MSEED format.
"""

import os
import numpy as np
from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from obspy import read_inventory

# Define the event and station parameters
event_lat = 52.498  # Event latitude
event_lon = 160.2637  # Event longitude
event_depth = 35  # Event depth (km)

# Initialize the IRIS client
client = Client("IRIS") 

# P-wave arrival time and time window for data extraction
P_arrival = UTCDateTime("2025-07-29T23:35:27")
starttime = P_arrival - 20  # Start time (20 seconds before P-wave arrival)
endtime = starttime + 200  # End time (200 seconds after start time)

# ========== Download seismic data ==========
# Check if data already exists locally, if not, download it
if os.path.exists("kcka_ANMO.mseed"):
    st = read("kcka_ANMO.mseed")
else:
    st = client.get_waveforms(
        network="IU",
        station="ANMO",
        location="00",
        channel="BH*",
        starttime=starttime,
        endtime=endtime,
    )
    st.write("kcka_ANMO.mseed", format="MSEED")

# ========== Retrieve instrument response data ==========
# Check if instrument response data exists, if not, download it
if os.path.exists("ANMO_stations.xml"):
    inv = read_inventory("ANMO_stations.xml")
else:
    inv = client.get_stations(
        network="IU",
        station="ANMO",
        location="00",
        channel="BH*",
        starttime=starttime,
        level="response",  # Get response data for the station
    )
    inv.write("ANMO_stations.xml", format="STATIONXML")

# Get station information
sta = inv[0][0]
sta_lat = sta.latitude
sta_lon = sta.longitude
print(f"Station latitude: {sta_lat}, longitude: {sta_lon}")

# ========== Data Preprocessing ==========
# Detrend the data (demean and linear trend removal)
st.detrend("demean")
st.detrend("linear")

# Apply cosine taper to reduce edge effects
st.taper(max_percentage=0.05, type="cosine")

# Remove instrument response (converting to velocity)
st.remove_response(
    inventory=inv,
    pre_filt=(0.005, 0.01, 5.0, 10.0),
    output="VEL",       # Output in velocity units
    water_level=60,     # Apply water level to the response removal
    plot=False,          # Don't plot the response removal
)

# Apply bandpass filter
st.filter(
    "bandpass",
    freqmin=0.1,
    freqmax=10,
    corners=4,
    zerophase=True,
)

# Apply cosine taper again after filtering
st.taper(max_percentage=0.05, type="cosine")

# Create copies of the processed data for further analysis
st_previous = st.copy()
st_processed = st.copy()

# ========== Rotation of components to ZNE coordinates ==========
# Calculate the azimuth and back azimuth between the event and station
dist_m, az, baz = gps2dist_azimuth(
    lat1=event_lat, lon1=event_lon,
    lat2=sta_lat, lon2=sta_lon
)

# Rotate the components to ZNE (vertical, north, east)
st_processed.rotate(method="->ZNE", inventory=inv)

# Ensure data is of type float32 for efficiency
for tr in st_processed:
    tr.data = tr.data.astype(np.float32)

# Save the processed data to a new MSEED file
st_processed.write("kcka_ANMO_ZNE.mseed", format="MSEED", encoding="FLOAT32")

# Optional: Plot the processed data
# for tr in st_processed:
#     plt.plot(tr.times(), tr.data)
# plt.show()

print("Data preprocessing complete. Processed data saved as 'kcka_ANMO_ZNE.mseed'.")
