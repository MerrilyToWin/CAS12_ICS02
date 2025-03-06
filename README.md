# Problem Statement
Urban areas face severe traffic congestion due to inefficient fixed-timer traffic signals, leading to delays, increased fuel consumption, and road safety risks. Emergency vehicles often struggle to pass through heavy traffic, causing critical delays in urgent situations. Additionally, unregulated pedestrian crossings in high-traffic areas increase the risk of accidents.

# Proposed Solution
The project aims to create intelligent traffic management systems using AI and IoT to reduce congestion and improve street efficiency. The system analyzes real-time traffic data from CCTV cameras to dynamically control the traffic signal. Emergency vehicle detection is included to prioritize ambulances and other important vehicles to quickly pass through intersections. Additionally, the system improves pedestrian safety by optimizing zebra cross-signals and capturing pedestrians in high traffic areas to prevent accidents. Simulation models real traffic conditions and integrates algorithms for traffic optimization of AI drives on a user-friendly monitoring dashboard.

# Tech Stack
Backend:
Flask (Python-based web framework for API and dashboard), 
YOLOv7 (Object detection for emergency vehicles & pedestrians), 
Reinforcement Learning (DQN, LSTM) (For traffic signal optimization), 

Frontend:
Flask (Jinja2 templates),  
JavaScript (Fetch and display live traffic data), 
HTML/CSS (For dashboard UI styling)

Simulations:
SUMO (Traffic simulation setup)

