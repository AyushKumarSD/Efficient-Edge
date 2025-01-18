# **Efficient Edge: Deep Learning-Based Energy Forecasting for IIoT-Enabled Data Servers**

## **Abstract**
In the era of Industrial Internet of Things (IIoT) 4.0, optimizing energy consumption is essential for improving operational performance and sustainability. This project focuses on forecasting energy consumption in data servers using advanced deep learning models, including Sequence-to-Sequence, RNNs, LSTMs, GRUs, and Transformers. By leveraging a dataset collected over 245 days at a one-second sampling interval, we analyze multivariate metrics (electrical, environmental, and server utilization) and propose a scalable solution for real-time energy monitoring and predictive analytics.

---

## **Features**
- **Energy Prediction Models:** Implementations of Sequence-to-Sequence, RNNs, LSTMs, GRUs, and Transformers for multivariate time-series forecasting.  
- **Real-Time Monitoring:** Integration with ESP32 hardware and the MQTT protocol to simulate real-time IIoT-based data collection.  
- **Scalable Framework:** Designed for real-world applications in IIoT-enabled environments.  
- **Actionable Insights:** Identify patterns to optimize energy usage and improve sustainability.

---

## **Dataset**
The dataset was collected from an HP Z440 workstation at the Information Technology Center (CTI) of Escuela Superior Politécnica del Litoral (ESPOL).

### **Key Features**
- **Duration:** 245 days (1 million observations).  
- **Granularity:** High-resolution (1-second intervals).  
- **Variables:**  
  - Electrical Metrics: Voltage, Current, Power, Frequency, Active Energy, Power Factor.  
  - Environmental Metrics: ESP32 Temperature.  
  - Server Utilization Metrics: CPU/GPU consumption, temperature, RAM memory usage, and power consumption.

### **Data Collection System**
- **Hardware:** ESP32 microcontroller.  
- **Protocol:** MQTT for lightweight communication and data transfer.

---

## **Installation**
To use this project locally, follow these steps:

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/efficient-edge-energy-forecasting.git
cd efficient-edge-energy-forecasting
```

### **2. Install Dependencies**
Create a virtual environment and install required Python packages:
```bash
python -m venv venv
source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

### **3. Download the Dataset**
Make sure the dataset is placed in the `data/` directory or update the configuration file with the correct path.

---

## **Usage**
### **1. Preprocess Data**
Run the preprocessing script to clean and prepare the data:
```bash
python preprocess.py
```

### **2. Train Models**
Train the deep learning models using the training script:
```bash
python train.py --model <model_name>
```
Replace `<model_name>` with `seq2seq`, `rnn`, `lstm`, `gru`, or `transformer`.

### **3. Evaluate Models**
Evaluate the trained models on the test dataset:
```bash
python evaluate.py --model <model_name>
```

### **4. Real-Time Monitoring**
Simulate real-time predictions using the ESP32-based system:
```bash
python real_time_monitoring.py
```

---

## **Project Structure**
```
efficient-edge-energy-forecasting/
│
├── data/                   # Dataset folder (place your dataset here)
├── models/                 # Saved models and checkpoints
├── scripts/
│   ├── preprocess.py       # Preprocessing script
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── real_time_monitoring.py  # Real-time monitoring script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── results/                # Folder for evaluation metrics and visualizations
```

---

## **Results**
- The models achieved state-of-the-art accuracy for energy consumption forecasting.  
- Performance metrics (e.g., MAE, RMSE) and visualizations can be found in the `results/` directory.

---

## **Team Members**
1. **Ayush Kumar:** Dataset preprocessing, Sequence-to-Sequence, RNN model implementation.  
2. **Vedant Kumar:** LSTM, GRU model implementation, and model evaluation.  
3. **Anwesha Sarangi:** Transformer model implementation, IIoT hardware integration.

---

## **Future Work**
- Expand the dataset to include other IIoT applications.  
- Improve model generalization for deployment in diverse environments.

---

## **License**
This project is licensed under the MIT License. See `LICENSE` for more information.

---

## **Acknowledgments**
- Escuela Superior Politécnica del Litoral (ESPOL) for providing the dataset.  
- Guidance from instructors and collaborators at AAI-530.
