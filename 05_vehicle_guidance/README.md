# Vehicle Guidance for Autonomous Driving

This module covers path planning and vehicle control algorithms essential for autonomous navigation. Vehicle guidance encompasses trajectory planning, path following, and control systems that enable safe and efficient autonomous vehicle operation.

## 🎯 Learning Objectives

- Understand path planning algorithms for autonomous vehicles
- Implement trajectory generation and optimization techniques
- Apply vehicle control systems (PID, MPC, LQR)
- Handle dynamic obstacle avoidance and replanning
- Integrate perception data with motion planning

## 📋 Module Contents

### Notebooks
- `01_path_planning_basics.ipynb` - Introduction to path planning
- `02_astar_rrt_algorithms.ipynb` - Classical planning algorithms
- `03_trajectory_optimization.ipynb` - Smooth trajectory generation
- `04_vehicle_control.ipynb` - Control system implementation
- `05_dynamic_planning.ipynb` - Real-time planning and replanning

### Source Code
- `src/planning/` - Path planning algorithms
- `src/control/` - Vehicle control systems
- `src/trajectory/` - Trajectory generation and optimization
- `src/dynamics/` - Vehicle dynamics models
- `src/simulation/` - Planning simulation environments

### Datasets
- `data/maps/` - Road network and obstacle maps
- `data/scenarios/` - Planning scenarios and test cases
- `data/trajectories/` - Reference trajectories
- `data/vehicle_params/` - Vehicle configuration files

## 🛠️ Key Technologies

### Path Planning Algorithms
- **A\***: Graph-based optimal path search
- **RRT/RRT\***: Rapidly-exploring Random Trees
- **Dijkstra**: Shortest path algorithm
- **D\* Lite**: Dynamic replanning algorithm

### Trajectory Optimization
- **Polynomial Trajectories**: Smooth curve fitting
- **Spline Interpolation**: Piecewise smooth paths
- **Optimal Control**: Variational methods
- **Model Predictive Control (MPC)**: Receding horizon optimization

### Vehicle Control
- **PID Control**: Proportional-Integral-Derivative
- **LQR**: Linear Quadratic Regulator
- **Pure Pursuit**: Geometric path following
- **Stanley Controller**: Front-wheel steering control

## 📊 Core Concepts

### Vehicle Kinematics
For autonomous vehicles, we commonly use:

**Bicycle Model:**
```python
def bicycle_model(state, control, dt, wheelbase):
    """
    state: [x, y, theta, v] - position, heading, velocity
    control: [delta, a] - steering angle, acceleration
    """
    x, y, theta, v = state
    delta, a = control
    
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + (v / wheelbase) * np.tan(delta) * dt
    v_new = v + a * dt
    
    return np.array([x_new, y_new, theta_new, v_new])
```

### Path Representation
Paths can be represented as:
```python
class Path:
    def __init__(self):
        self.waypoints = []  # List of (x, y) coordinates
        self.headings = []   # Orientation at each waypoint
        self.velocities = [] # Speed profile
        self.curvatures = [] # Path curvature
        
    def add_waypoint(self, x, y, heading=None, velocity=None):
        self.waypoints.append((x, y))
        self.headings.append(heading)
        self.velocities.append(velocity)
```

### Planning Problem Formulation
Given:
- **Start state**: x_start = [x_0, y_0, θ_0, v_0]
- **Goal state**: x_goal = [x_g, y_g, θ_g, v_g]
- **Obstacles**: O = {O_1, O_2, ..., O_n}
- **Vehicle constraints**: Kinematic and dynamic limits

Find: Optimal trajectory that minimizes cost while satisfying constraints.

## 🚀 Quick Start

1. **Environment Setup**
```bash
cd 05_vehicle_guidance
pip install -r requirements.txt
```

2. **Install Planning Libraries**
```bash
pip install casadi  # For optimization
pip install ompl-thin  # For motion planning
```

3. **Run Basic Planning**
```bash
jupyter notebook notebooks/01_path_planning_basics.ipynb
```

## 📚 Theoretical Background

### A* Algorithm
Graph-based search with heuristic guidance:

```python
def a_star(start, goal, graph, heuristic):
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            return reconstruct_path(came_from, current)
        
        for neighbor in graph.neighbors(current):
            tentative_g = g_score[current] + graph.cost(current, neighbor)
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found
```

### RRT Algorithm
Probabilistic path planning through random sampling:

```python
class RRT:
    def __init__(self, start, goal, obstacles, step_size=1.0):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.step_size = step_size
        self.tree = {start: None}
    
    def sample(self):
        if random.random() < 0.1:  # Goal bias
            return self.goal
        return (random.uniform(-10, 10), random.uniform(-10, 10))
    
    def nearest(self, point):
        min_dist = float('inf')
        nearest_node = None
        for node in self.tree:
            dist = np.linalg.norm(np.array(node) - np.array(point))
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        return nearest_node
    
    def steer(self, from_node, to_point):
        direction = np.array(to_point) - np.array(from_node)
        distance = np.linalg.norm(direction)
        if distance <= self.step_size:
            return to_point
        return tuple(np.array(from_node) + (direction / distance) * self.step_size)
```

### Model Predictive Control (MPC)
Optimal control with receding horizon:

```python
def mpc_controller(current_state, reference_path, horizon, constraints):
    """
    Solve optimization problem:
    min Σ(||x_k - x_ref||² + ||u_k||²)
    s.t. x_{k+1} = f(x_k, u_k)
         u_min ≤ u_k ≤ u_max
    """
    from casadi import *
    
    # Decision variables
    X = MX.sym('X', 4, horizon+1)  # States
    U = MX.sym('U', 2, horizon)    # Controls
    
    # Objective function
    obj = 0
    for k in range(horizon):
        # State cost
        obj += mtimes((X[:, k] - reference_path[k]).T, 
                     Q @ (X[:, k] - reference_path[k]))
        # Control cost
        obj += mtimes(U[:, k].T, R @ U[:, k])
    
    # Constraints
    g = []
    # Dynamics constraints
    for k in range(horizon):
        x_next = bicycle_model(X[:, k], U[:, k], dt, wheelbase)
        g.append(X[:, k+1] - x_next)
    
    # Solve optimization problem
    nlp = {'x': vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))), 
           'f': obj, 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', nlp)
    
    return solver
```

## 🔬 Practical Exercises

### Exercise 1: Graph-Based Path Planning
Implement A* algorithm for finding optimal paths in grid environments.

**Tasks:**
- Create grid representation of environment
- Implement A* with Euclidean heuristic
- Handle obstacles and constraints
- Visualize search process and final path

**Sample Implementation:**
```python
class GridPlanner:
    def __init__(self, grid_map):
        self.grid = grid_map
        self.height, self.width = grid_map.shape
    
    def plan_path(self, start, goal):
        path = a_star(start, goal, self.grid, self.euclidean_heuristic)
        return self.smooth_path(path)
    
    def smooth_path(self, path):
        # Apply path smoothing algorithm
        return smoothed_path
```

### Exercise 2: RRT Path Planning
Develop RRT algorithm for continuous space planning.

**Objectives:**
- Implement basic RRT algorithm
- Add goal biasing and path optimization
- Handle non-holonomic constraints
- Compare with RRT* variant

### Exercise 3: Trajectory Optimization
Generate smooth, executable trajectories from waypoint paths.

**Components:**
- Polynomial trajectory generation
- Velocity profile optimization
- Comfort and safety constraints
- Real-time trajectory modification

### Exercise 4: Vehicle Control Implementation
Implement various control algorithms for path following.

**Controllers to Implement:**
1. **PID Controller** for speed control
2. **Pure Pursuit** for path following
3. **Stanley Controller** for lateral control
4. **MPC** for integrated control

## 📈 Advanced Topics

### Dynamic Obstacle Avoidance
```python
class DynamicPlanner:
    def __init__(self):
        self.static_obstacles = []
        self.dynamic_obstacles = []
        
    def replan(self, current_pose, goal, predictions):
        """
        Replan considering predicted obstacle trajectories
        """
        # Predict future obstacle positions
        future_obstacles = self.predict_obstacles(predictions)
        
        # Generate collision-free path
        path = self.plan_with_moving_obstacles(
            current_pose, goal, future_obstacles)
        
        return path
```

### Lattice-Based Planning
```python
class LatticePlanner:
    def __init__(self, resolution=0.5):
        self.resolution = resolution
        self.motion_primitives = self.generate_primitives()
    
    def generate_primitives(self):
        """Generate set of feasible motion primitives"""
        primitives = []
        for steer in [-30, -15, 0, 15, 30]:  # degrees
            for velocity in [1, 2, 3]:  # m/s
                primitive = self.generate_primitive(steer, velocity)
                primitives.append(primitive)
        return primitives
```

### Behavior Planning
```python
class BehaviorPlanner:
    def __init__(self):
        self.states = ['FOLLOW_LANE', 'CHANGE_LANE_LEFT', 
                      'CHANGE_LANE_RIGHT', 'STOP']
        self.current_state = 'FOLLOW_LANE'
    
    def update_behavior(self, perception_data):
        # Finite state machine for high-level behavior
        if self.should_change_lane(perception_data):
            self.current_state = self.select_lane_change()
        elif self.should_stop(perception_data):
            self.current_state = 'STOP'
        else:
            self.current_state = 'FOLLOW_LANE'
        
        return self.current_state
```

### Trajectory Optimization with Constraints
```python
def optimize_trajectory(waypoints, vehicle_params, constraints):
    """
    Optimize trajectory considering:
    - Kinematic constraints (max steering, acceleration)
    - Dynamic constraints (friction limits)
    - Comfort constraints (jerk, lateral acceleration)
    - Safety constraints (obstacle avoidance)
    """
    from scipy.optimize import minimize
    
    def objective(params):
        trajectory = generate_trajectory(params, waypoints)
        return calculate_trajectory_cost(trajectory)
    
    def constraint_function(params):
        trajectory = generate_trajectory(params, waypoints)
        return check_constraints(trajectory, constraints)
    
    result = minimize(objective, initial_params, 
                     constraints={'type': 'ineq', 'fun': constraint_function})
    
    return result.x
```

## 📊 Evaluation Metrics

### Path Quality
- **Path Length**: Total distance of planned path
- **Smoothness**: Curvature and discontinuity measures
- **Clearance**: Minimum distance to obstacles
- **Feasibility**: Compliance with vehicle constraints

### Control Performance
- **Tracking Error**: Deviation from reference path
- **Control Effort**: Magnitude of control inputs
- **Stability**: System response characteristics
- **Settling Time**: Time to reach steady state

### Real-Time Performance
- **Planning Time**: Time to generate path
- **Update Rate**: Frequency of replanning
- **Success Rate**: Percentage of successful plans
- **Computational Load**: CPU and memory usage

## 🎯 Real-World Applications

### Highway Driving
- **Lane Keeping**: Maintain vehicle in lane center
- **Lane Changing**: Safe lane change maneuvers
- **Merge/Exit**: Highway on-ramp and off-ramp navigation
- **Adaptive Cruise**: Speed and following distance control

### Urban Driving
- **Intersection Navigation**: Traffic light and sign handling
- **Parking**: Parallel and perpendicular parking maneuvers
- **Roundabouts**: Complex multi-lane navigation
- **Construction Zones**: Temporary obstacle avoidance

### Emergency Scenarios
- **Collision Avoidance**: Emergency steering and braking
- **Obstacle Avoidance**: Sudden obstacle appearance
- **Vehicle Recovery**: Handling loss of control
- **Safe Stop**: Emergency stop procedures

## 📖 Additional Resources

### Datasets
- [NGSIM](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm): Highway vehicle trajectories
- [highD](https://www.highd-dataset.com/): Highway drone dataset
- [inD](https://www.ind-dataset.com/): Intersection drone dataset
- [INTERACTION](http://interaction-dataset.com/): Interactive driving scenarios

### Key Papers
- [Practical Search Techniques in Path Planning for Autonomous Driving](https://ai.stanford.edu/~ddolgov/papers/dolgov_gpp_stair08.pdf)
- [Real-time Motion Planning Methods for Autonomous On-road Driving](https://www.ri.cmu.edu/pub_files/2015/7/IJRR_2015_Real-time_Motion_Planning_Methods_Autonomous_On-road_Driving.pdf)
- [Model Predictive Control for Autonomous Driving](https://arxiv.org/abs/1502.02791)

### Software Tools
- [OMPL](https://ompl.kavrakilab.org/): Open Motion Planning Library
- [CasADi](https://web.casadi.org/): Optimization framework
- [SUMO](https://www.eclipse.org/sumo/): Traffic simulation
- [CARLA](http://carla.org/): Autonomous driving simulator

## 🎯 Assessment Criteria

- **Algorithm Implementation** (35%): Correct planning and control algorithms
- **Path Quality** (25%): Smoothness, safety, and optimality
- **Real-time Performance** (25%): Computational efficiency
- **Integration** (15%): Combining planning and control components

## 🔄 Integration with Other Modules

This module connects with:
- **Occupancy Mapping**: Using maps for planning
- **Object Tracking**: Dynamic obstacle consideration
- **Point Cloud Processing**: 3D environment understanding
- **V2X Communication**: Cooperative planning information

---

*Vehicle guidance transforms perception and mapping information into actionable control commands, enabling safe and efficient autonomous vehicle navigation in complex environments.*