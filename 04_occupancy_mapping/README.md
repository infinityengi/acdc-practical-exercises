# Occupancy Mapping for Autonomous Driving

This module covers occupancy grid mapping techniques essential for spatial understanding and navigation in autonomous vehicles. Occupancy maps provide a probabilistic representation of the environment, distinguishing between free space, occupied space, and unknown areas.

## 🎯 Learning Objectives

- Understand probabilistic occupancy mapping fundamentals
- Implement grid-based environment representation
- Apply Bayesian inference for map updates
- Perform SLAM (Simultaneous Localization and Mapping)
- Handle dynamic environments and moving objects

## 📋 Module Contents

### Notebooks
- `01_occupancy_grid_basics.ipynb` - Introduction to occupancy grids
- `02_bayesian_mapping.ipynb` - Probabilistic map updates
- `03_lidar_mapping.ipynb` - LiDAR-based occupancy mapping
- `04_slam_implementation.ipynb` - Basic SLAM algorithm
- `05_dynamic_mapping.ipynb` - Handling moving objects

### Source Code
- `src/mapping/` - Core mapping algorithms
- `src/grid/` - Grid representation and operations
- `src/sensors/` - Sensor models for mapping
- `src/slam/` - SLAM implementations
- `src/visualization/` - Map visualization tools

### Datasets
- `data/laser_scans/` - LiDAR scan sequences
- `data/odometry/` - Vehicle motion data
- `data/maps/` - Reference maps and ground truth
- `data/sensor_logs/` - Multi-sensor datasets

## 🛠️ Key Technologies

### Mapping Frameworks
- **Occupancy Grids**: Discrete grid-based representation
- **Probabilistic Mapping**: Bayesian inference for updates
- **SLAM**: Simultaneous localization and mapping
- **Multi-Resolution Maps**: Hierarchical representations

### Sensor Integration
- **LiDAR Mapping**: Range sensor integration
- **Camera-based Mapping**: Visual feature mapping
- **Sonar/Ultrasonic**: Short-range obstacle detection
- **Multi-Sensor Fusion**: Combining multiple modalities

### Algorithms
- **Bresenham's Algorithm**: Ray tracing in grids
- **Bayes' Rule**: Probabilistic updates
- **Particle Filters**: Non-parametric localization
- **Graph SLAM**: Pose graph optimization

## 📊 Core Concepts

### Occupancy Grid Representation
```python
class OccupancyGrid:
    def __init__(self, width, height, resolution):
        self.width = width
        self.height = height
        self.resolution = resolution  # meters per cell
        self.grid = np.ones((height, width)) * 0.5  # Unknown = 0.5
        
    def world_to_grid(self, x, y):
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)
        return grid_x, grid_y
```

### Probability Values
- **Free Space**: P(occupied) = 0.0 to 0.3
- **Unknown**: P(occupied) = 0.5 
- **Occupied**: P(occupied) = 0.7 to 1.0

### Sensor Models
Range sensors follow inverse sensor models:
```python
def inverse_sensor_model(r, theta, max_range, grid):
    """
    Update occupancy probabilities based on range measurement
    r: measured range
    theta: measurement angle
    max_range: sensor maximum range
    """
    for i in range(int(r / grid.resolution)):
        x = i * grid.resolution * cos(theta)
        y = i * grid.resolution * sin(theta)
        grid_x, grid_y = grid.world_to_grid(x, y)
        
        if i < r / grid.resolution - 1:
            # Free space
            grid.update_cell(grid_x, grid_y, 0.3)
        else:
            # Occupied space
            grid.update_cell(grid_x, grid_y, 0.7)
```

## 🚀 Quick Start

1. **Environment Setup**
```bash
cd 04_occupancy_mapping
pip install -r requirements.txt
```

2. **Download Mapping Data**
```bash
python src/utils/download_mapping_data.py
```

3. **Run Basic Mapping**
```bash
jupyter notebook notebooks/01_occupancy_grid_basics.ipynb
```

## 📚 Theoretical Background

### Bayesian Map Updates
Occupancy mapping uses Bayesian inference to update cell probabilities:

**Log-Odds Representation:**
```
l(m_i|z_1:t, x_1:t) = l(m_i|z_t, x_t) + l(m_i|z_1:t-1, x_1:t-1) - l(m_i)
```

Where:
- `l(m_i)`: Prior log-odds of cell i
- `z_t`: Sensor measurement at time t
- `x_t`: Robot pose at time t

**Probability Conversion:**
```python
def log_odds_to_prob(log_odds):
    return 1 - 1 / (1 + np.exp(log_odds))

def prob_to_log_odds(prob):
    return np.log(prob / (1 - prob))
```

### SLAM Problem
Simultaneously estimate:
1. **Robot trajectory**: x_1:t = {x_1, x_2, ..., x_t}
2. **Map**: m = {m_1, m_2, ..., m_n}

Given:
- **Sensor measurements**: z_1:t = {z_1, z_2, ..., z_t}
- **Control inputs**: u_1:t = {u_1, u_2, ..., u_t}

### Motion Models
Vehicle motion can be modeled as:
```python
def motion_model(x_prev, u, dt):
    """
    x_prev: [x, y, theta] previous pose
    u: [v, omega] control input (velocity, angular velocity)
    dt: time step
    """
    x_new = x_prev[0] + u[0] * np.cos(x_prev[2]) * dt
    y_new = x_prev[1] + u[0] * np.sin(x_prev[2]) * dt
    theta_new = x_prev[2] + u[1] * dt
    return np.array([x_new, y_new, theta_new])
```

## 🔬 Practical Exercises

### Exercise 1: Basic Occupancy Grid
Implement a simple occupancy grid with manual updates.

**Tasks:**
- Create grid data structure
- Implement coordinate transformations
- Add simple obstacle placement
- Visualize grid states

**Implementation:**
```python
class SimpleOccupancyGrid:
    def __init__(self, width, height, resolution):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid = np.ones((height, width)) * 0.5
        
    def add_obstacle(self, x, y, size):
        gx, gy = self.world_to_grid(x, y)
        for dx in range(-size, size+1):
            for dy in range(-size, size+1):
                if 0 <= gx+dx < self.width and 0 <= gy+dy < self.height:
                    self.grid[gy+dy, gx+dx] = 1.0
```

### Exercise 2: LiDAR-based Mapping
Build an occupancy map from LiDAR range data.

**Objectives:**
- Process laser scan data
- Implement ray tracing algorithms
- Apply inverse sensor model
- Handle sensor noise and uncertainty

### Exercise 3: Probabilistic Map Updates
Implement Bayesian updates for occupancy probabilities.

**Components:**
- Log-odds probability representation
- Sequential map updates
- Handling conflicting measurements
- Convergence analysis

### Exercise 4: Basic SLAM
Combine localization and mapping in a unified framework.

**Implementation Steps:**
1. Predict robot motion using odometry
2. Update map with new sensor data
3. Correct robot pose using map features
4. Iterate for sequential processing

## 📈 Advanced Topics

### Multi-Resolution Mapping
```python
class HierarchicalGrid:
    def __init__(self, levels=3):
        self.levels = levels
        self.grids = []
        for i in range(levels):
            resolution = 0.1 * (2 ** i)  # Increasing resolution
            self.grids.append(OccupancyGrid(100, 100, resolution))
    
    def update_all_levels(self, scan_data, pose):
        for grid in self.grids:
            grid.update(scan_data, pose)
```

### Dynamic Environment Handling
- **Temporal Filtering**: Remove temporary obstacles
- **Object Classification**: Distinguish static vs dynamic objects
- **Persistence Modeling**: Track object permanence
- **Map Cleaning**: Remove outdated information

### Memory-Efficient Mapping
```python
class SparseOccupancyGrid:
    def __init__(self):
        self.occupied_cells = set()
        self.free_cells = set()
        
    def is_occupied(self, x, y):
        return (x, y) in self.occupied_cells
    
    def set_occupied(self, x, y):
        self.occupied_cells.add((x, y))
        self.free_cells.discard((x, y))
```

### Loop Closure Detection
- **Feature Matching**: Identify revisited locations
- **Pose Graph Optimization**: Correct accumulated drift
- **Map Consistency**: Maintain global map coherence
- **Robust Estimation**: Handle false loop closures

## 📊 Evaluation Metrics

### Map Quality
- **Accuracy**: Comparison with ground truth maps
- **Completeness**: Coverage of explored areas
- **Consistency**: Internal map coherence
- **Resolution**: Level of detail captured

### Computational Performance
- **Update Speed**: Cells processed per second
- **Memory Usage**: Storage requirements
- **Scalability**: Performance with map size
- **Real-time Capability**: Meeting timing constraints

### SLAM Performance
- **Trajectory Error**: Difference from ground truth path
- **Map Error**: Deviation from reference map
- **Loop Closure Success**: Correct closure detection rate
- **Drift Accumulation**: Pose error over time

## 🎯 Real-World Applications

### Autonomous Navigation
- **Path Planning**: Free space identification
- **Obstacle Avoidance**: Real-time hazard detection
- **Parking**: Detailed spatial mapping
- **Lane Keeping**: Road boundary detection

### Industrial Applications
- **Warehouse Robotics**: Inventory environment mapping
- **Construction Sites**: Dynamic obstacle tracking
- **Mining Operations**: Underground navigation
- **Agriculture**: Field boundary mapping

### Safety Systems
- **Emergency Braking**: Obstacle detection and mapping
- **Blind Spot Monitoring**: Surrounding environment awareness
- **Collision Prediction**: Trajectory analysis
- **Rescue Operations**: Unknown environment exploration

## 📖 Additional Resources

### Datasets
- [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php): Vehicle SLAM
- [TUM RGB-D](https://vision.in.tum.de/data/datasets/rgbd-dataset): Indoor SLAM
- [EuRoC MAV](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets): Aerial vehicle SLAM
- [Oxford RobotCar](https://robotcar-dataset.robots.ox.ac.uk/): Long-term autonomous driving

### Key Papers
- [Occupancy Grid Mapping](https://www.cse.iitd.ac.in/~pkalra/col341/2021/slides/Occupancy_Grid_Mapping.pdf)
- [Simultaneous Localization and Mapping](http://robots.stanford.edu/papers/thrun.graphslam.pdf)
- [Real-Time Loop Closure in 2D LIDAR SLAM](https://research.google.com/pubs/pub45466.html)

### Software Libraries
- [GMapping](http://openslam.org/gmapping.html): Grid-based SLAM
- [Hector SLAM](http://wiki.ros.org/hector_slam): Real-time SLAM
- [Cartographer](https://github.com/cartographer-project/cartographer): Google's SLAM solution
- [RTAB-Map](http://introlab.github.io/rtabmap/): RGB-D SLAM

## 🎯 Assessment Criteria

- **Map Quality** (35%): Accuracy and completeness of generated maps
- **Algorithm Implementation** (30%): Correct probabilistic updates
- **Real-time Performance** (20%): Processing speed and efficiency
- **Documentation** (15%): Clear explanations and analysis

## 🔄 Integration with Other Modules

This module connects with:
- **Point Cloud Processing**: 3D mapping and localization
- **Object Tracking**: Dynamic object handling
- **Vehicle Guidance**: Path planning in mapped environments
- **Image Segmentation**: Visual feature mapping

---

*Occupancy mapping provides the spatial foundation for autonomous vehicle navigation, enabling safe path planning and obstacle avoidance in complex environments.*