# Problem Statement
Urban areas face severe traffic congestion due to inefficient fixed-timer traffic signals, leading to delays, increased fuel consumption, and road safety risks. Emergency vehicles often struggle to pass through heavy traffic, causing critical delays in urgent situations. Additionally, unregulated pedestrian crossings in high-traffic areas increase the risk of accidents.

# Proposed Solution
The project aims to create intelligent traffic management systems using AI and IoT to reduce congestion and improve street efficiency. The system analyzes real-time traffic data from CCTV cameras to dynamically control the traffic signal. Emergency vehicle detection is included to prioritize ambulances and other important vehicles to quickly pass through intersections. Additionally, the system improves pedestrian safety by optimizing zebra cross-signals and capturing pedestrians in high traffic areas to prevent accidents. Simulation models real traffic conditions and integrates algorithms for traffic optimization of AI drives on a user-friendly monitoring dashboard.

## 🔧 **Implementation Details**  
### **1️⃣ Vehicle Detection Module**  
- Uses **YOLOv8** to detect vehicles in live traffic feeds.  
- Classifies detected vehicles into **cars, bikes, buses, trucks, and rickshaws**.  

### **2️⃣ Signal Switching Algorithm**  
- Dynamically adjusts **red, yellow, and green signal durations**.  
- Takes into account:  
  ✅ **Vehicle count per lane**  
  ✅ **Vehicle type (car, bus, etc.)**  
  ✅ **Average vehicle speed**  

### **3️⃣ Simulation Module**  
- Built using **[Pygame](https://www.pygame.org/news)** to simulate:  
  ✅ **Traffic signals**  
  ✅ **Vehicle movements**  
  ✅ **Signal timing adjustments**  
 

---

## 🚀 **Features**  
✅ **YOLOv8-Based Vehicle Detection** – Detects vehicles from real-time traffic video feeds.  
✅ **LSTM-Based Traffic Prediction** – Forecasts future congestion trends.  
✅ **Automated Traffic Signal Control** – Adjusts green light durations dynamically.  
✅ **Multiple Simulation Runs** – Runs the simulation multiple times and saves data.  
✅ **Excel Report Output** – Stores final traffic analysis results.  

---

## 🛠 **How It Works**  
1. **Vehicle Detection:** YOLOv8 detects vehicles from live video feeds.  
2. **Traffic Data Processing:** The detected vehicle counts are analyzed.  
3. **Signal Adjustment:** Green light durations are set based on vehicle density.  
4. **Prediction with LSTM:** Future congestion trends are forecasted.  
5. **Simulation Execution:** The model is tested through multiple runs.  
6. **Result Storage:** Data is saved in an **Excel sheet** for further analysis.




