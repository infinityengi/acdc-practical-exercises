# ACDC Practical Exercises - Student Portfolio

A comprehensive portfolio showcasing practical work in Automated and Connected Driving Challenges (ACDC). This repository demonstrates hands-on experience across key areas of autonomous vehicle technology including perception, mapping, fusion, planning, and connected driving.

## 🚗 Repository Overview

This portfolio contains practical implementations and exercises covering the core technologies behind autonomous driving systems:

### 📁 Module Structure

| Module | Focus Area | Key Technologies |
|--------|------------|------------------|
| **01_image_segmentation** | Computer Vision | Semantic/Instance segmentation, CNNs, U-Net |
| **02_point_cloud_processing** | LiDAR Processing | Point cloud filtering, registration, clustering |
| **03_object_tracking** | Dynamic Perception | Kalman filters, multi-object tracking, SORT/DeepSORT |
| **04_occupancy_mapping** | Environment Mapping | Grid maps, probabilistic mapping, SLAM |
| **05_vehicle_guidance** | Path Planning & Control | A*, RRT, PID control, MPC |
| **06_v2x_communication** | Connected Driving | V2V, V2I communication, DSRC, 5G |

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.8+
- Git
- Jupyter Notebook

### Installation
```bash
# Clone the repository
git clone https://github.com/infinityengi/acdc-practical-exercises.git
cd acdc-practical-exercises

# Create virtual environment
python -m venv acdc_env
source acdc_env/bin/activate  # On Windows: acdc_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter for notebooks
jupyter notebook
```

## 📚 Module Descriptions

### 🖼️ Image Segmentation
Semantic and instance segmentation techniques for understanding road scenes, including lane detection, vehicle identification, and road infrastructure recognition.

**Key Techniques:** U-Net, DeepLab, Mask R-CNN, data augmentation
**Applications:** Lane keeping, traffic sign recognition, obstacle detection

### ☁️ Point Cloud Processing  
Processing and analysis of LiDAR data for 3D environment understanding, including point cloud filtering, registration, and object clustering.

**Key Techniques:** PCL, RANSAC, ICP, clustering algorithms
**Applications:** 3D object detection, mapping, localization

### 🎯 Object Tracking
Multi-object tracking in dynamic environments using computer vision and sensor fusion techniques.

**Key Techniques:** Kalman filtering, Hungarian algorithm, SORT/DeepSORT
**Applications:** Pedestrian tracking, vehicle following, collision avoidance

### 🗺️ Occupancy Mapping
Grid-based environment representation and probabilistic mapping for autonomous navigation.

**Key Techniques:** Occupancy grids, Bayesian inference, SLAM
**Applications:** Path planning, obstacle avoidance, map building

### 🚙 Vehicle Guidance
Path planning and vehicle control algorithms for autonomous navigation in various scenarios.

**Key Techniques:** A*, RRT, PID control, Model Predictive Control
**Applications:** Route planning, trajectory following, parking assistance

### 📡 V2X Communication
Vehicle-to-everything communication protocols and applications for connected driving scenarios.

**Key Techniques:** DSRC, 5G-V2X, message protocols, networking
**Applications:** Traffic coordination, safety warnings, cooperative driving

## 📊 Portfolio Highlights

- **Practical Implementation**: Working code examples for each major concept
- **Real-world Data**: Exercises using actual automotive datasets
- **Visualization**: Interactive plots and 3D visualizations
- **Documentation**: Comprehensive explanations of theory and implementation
- **Modular Design**: Each module can be studied independently

## 🎓 Learning Outcomes

By working through this portfolio, you will gain practical experience in:

- Computer vision techniques for autonomous driving
- Sensor data processing and fusion
- Real-time object detection and tracking
- Path planning and control algorithms
- Connected vehicle technologies
- System integration and testing

## 📖 Usage

Each module contains:
- **README.md**: Detailed module documentation
- **notebooks/**: Jupyter notebooks with interactive examples
- **src/**: Source code and utilities
- **data/**: Sample datasets and test files
- **docs/**: Additional documentation and references
- **examples/**: Standalone example scripts

Navigate to any module directory and start with the README to understand the concepts and run the example notebooks.

## 🤝 Contributing

This is a student portfolio showcasing learning progress. Suggestions and improvements are welcome through issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ACDC MOOC instructors and course materials
- Open source autonomous driving community
- Dataset providers and tool developers

---

**Author**: Om Prakash Sahu  
**Purpose**: Student portfolio for ACDC (Automated and Connected Driving Challenges)  
**Last Updated**: December 2024
