# Single-Station-TeleSeismic-Localization
> **⚠️ Status: Proof-of-Concept**  
> An experimental implementation for research and testing. Core algorithms are functional but require further validation. Active development is planned to expand features and robustness in future releases.

## 📖 Overview

This project implements a **single-station teleseismic localization** method. By leveraging a single three-component seismic station, it determines the **back azimuth** and **incident angle** of teleseismic events. These parameters, combined with **P-wave and S-wave arrival times**, are used in an inversion process to estimate the epicenter and depth of distant earthquakes.

### 🔬 Core Methodology: P-Wave Polarization Analysis

The cornerstone of this approach is **P-wave polarization analysis**. Unlike traditional methods that rely on dense arrays, this technique exploits the particle motion characteristics of the initial P-wave arrival:

1.  **Particle Motion**: In an isotropic medium, the P-wave particle motion is linear and aligns with the ray path.
2.  **Back Azimuth Estimation**: By analyzing the covariance matrix of the three-component (Z, N, E) waveforms in a time window around the P-onset, we can determine the dominant direction of particle motion. This yields the **back azimuth** ($\phi$) from the station to the event.
3.  **Incident Angle**: The ratio of vertical to horizontal amplitude components allows for the estimation of the **incident angle** ($i$).

Once the back azimuth and incident angle are constrained, they serve as critical inputs for the location inversion alongside the $P-S$ travel time difference.


## 🚀 Key Features

-   **Single-Station Location**: Capable of locating teleseismic events using data from just one three-component station.
-   **Polarization-Based Azimuth**: Robust estimation of back azimuth using P-wave particle motion properties.
-   **Joint Inversion**: Integrates geometric constraints (azimuth/incident angle) with temporal constraints ($t_S - t_P$) for improved location accuracy.
-   **Station Orientation Check**: If the earthquake location is already known (e.g., from a global catalog), this tool can be reversed to verify or calibrate the orientation of the station's horizontal components (North/East alignment), crucial for correcting misaligned OBS deployments.

## 🪐 Applications & Potential

While validated primarily on terrestrial data, this methodology holds significant promise for **planetary seismology**:

-   **Mars Seismology**: With sparse station coverage (e.g., the InSight mission), single-station techniques are essential for locating marsquakes.
-   **Lunar Seismology**: Applicable to future lunar networks where station density will be low.
-   **Remote Earth Environments**: Ideal for ocean-bottom seismometers (OBS) and polar deployments, serving dual purposes: locating events where arrays are unavailable and calibrating the horizontal orientation of isolated sensors.
