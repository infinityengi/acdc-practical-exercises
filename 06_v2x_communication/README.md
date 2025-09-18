# V2X Communication for Connected Driving

This module covers Vehicle-to-Everything (V2X) communication technologies essential for connected and cooperative autonomous driving. V2X enables vehicles to communicate with other vehicles, infrastructure, pedestrians, and cloud services to enhance safety and efficiency.

## 🎯 Learning Objectives

- Understand V2X communication protocols and standards
- Implement vehicle-to-vehicle (V2V) messaging systems
- Apply vehicle-to-infrastructure (V2I) communication
- Handle message security and reliability in V2X networks
- Develop cooperative driving applications using V2X data

## 📋 Module Contents

### Notebooks
- `01_v2x_fundamentals.ipynb` - Introduction to V2X communication
- `02_v2v_messaging.ipynb` - Vehicle-to-vehicle communication
- `03_v2i_systems.ipynb` - Vehicle-to-infrastructure protocols
- `04_security_protocols.ipynb` - V2X security and privacy
- `05_cooperative_applications.ipynb` - Collaborative driving scenarios

### Source Code
- `src/communication/` - V2X protocol implementations
- `src/messaging/` - Message encoding/decoding utilities
- `src/security/` - Cryptographic functions and certificates
- `src/applications/` - Cooperative driving applications
- `src/simulation/` - V2X network simulation tools

### Datasets
- `data/traces/` - Real-world V2X communication traces
- `data/scenarios/` - Cooperative driving scenarios
- `data/messages/` - Sample V2X message datasets
- `data/certificates/` - PKI certificates for testing

## 🛠️ Key Technologies

### Communication Standards
- **DSRC**: Dedicated Short Range Communications (IEEE 802.11p)
- **5G-V2X**: Cellular V2X communication (3GPP Release 14+)
- **WiFi-6E**: Extended range WiFi for V2X
- **LTE-V2X**: LTE-based vehicular communication

### Message Types
- **BSM**: Basic Safety Messages (cooperative awareness)
- **CAM**: Cooperative Awareness Messages (ETSI standard)
- **DENM**: Decentralized Environmental Notification Messages
- **SPAT**: Signal Phase and Timing messages
- **MAP**: Geographic intersection geometry

### Network Protocols
- **IEEE 1609.4**: Multi-channel operations
- **IEEE 1609.3**: Network services
- **IEEE 1609.2**: Security services
- **ETSI ITS-G5**: European V2X standard

## 📊 Core Concepts

### V2X Communication Types
```python
class V2XMessageTypes:
    V2V = "Vehicle-to-Vehicle"          # Direct vehicle communication
    V2I = "Vehicle-to-Infrastructure"   # Traffic lights, road signs
    V2P = "Vehicle-to-Pedestrian"       # Smartphones, wearables
    V2N = "Vehicle-to-Network"          # Cloud services, traffic management
    V2D = "Vehicle-to-Device"           # Any smart device
```

### Basic Safety Message (BSM) Structure
```python
class BasicSafetyMessage:
    def __init__(self):
        self.msg_id = 20  # BSM identifier
        self.temp_id = None  # Temporary vehicle ID
        self.sec_mark = None  # Time within current minute
        self.latitude = None  # Vehicle latitude (1/10 micro degrees)
        self.longitude = None  # Vehicle longitude (1/10 micro degrees)
        self.elevation = None  # Vehicle elevation (decimeters)
        self.speed = None  # Vehicle speed (0.02 m/s units)
        self.heading = None  # Vehicle heading (0.0125 degrees)
        self.steering = None  # Steering wheel angle
        self.acceleration = None  # Longitudinal acceleration
        self.brakes = None  # Brake system status
        
    def encode(self):
        """Encode BSM into ASN.1 format"""
        return asn1_encode(self.__dict__)
        
    def decode(self, data):
        """Decode BSM from received data"""
        decoded = asn1_decode(data)
        self.__dict__.update(decoded)
```

### Communication Range and Reliability
- **DSRC Range**: 300-1000 meters
- **5G-V2X Range**: Up to several kilometers
- **Update Rate**: 1-10 Hz for safety applications
- **Latency**: <100ms for safety-critical messages

## 🚀 Quick Start

1. **Environment Setup**
```bash
cd 06_v2x_communication
pip install -r requirements.txt
```

2. **Install V2X Libraries**
```bash
pip install paho-mqtt  # For MQTT messaging
pip install websockets  # For WebSocket communication
pip install cryptography  # For security protocols
```

3. **Run Basic Communication**
```bash
jupyter notebook notebooks/01_v2x_fundamentals.ipynb
```

## 📚 Theoretical Background

### DSRC/IEEE 802.11p Protocol Stack
```
┌─────────────────┐
│   Applications  │ (Cooperative awareness, safety apps)
├─────────────────┤
│   IEEE 1609.3   │ (WAVE networking services)
├─────────────────┤
│   IEEE 1609.4   │ (Multi-channel coordination)
├─────────────────┤
│   IEEE 802.11p  │ (Physical and MAC layers)
└─────────────────┘
```

### Message Priority and Channel Access
V2X uses Enhanced Distributed Channel Access (EDCA) with four priority levels:
- **AC_VO**: Voice (highest priority) - emergency messages
- **AC_VI**: Video - safety-related messages
- **AC_BE**: Best Effort - general information
- **AC_BK**: Background (lowest priority) - non-safety data

### Security Architecture
```python
class V2XSecurity:
    def __init__(self):
        self.certificate_authority = None
        self.pseudonym_certificates = []
        self.private_key = None
        
    def sign_message(self, message):
        """Sign message with current pseudonym certificate"""
        signature = self.create_signature(message, self.private_key)
        return {
            'message': message,
            'signature': signature,
            'certificate': self.get_current_certificate()
        }
        
    def verify_message(self, signed_message):
        """Verify message signature and certificate"""
        cert_valid = self.verify_certificate(signed_message['certificate'])
        sig_valid = self.verify_signature(
            signed_message['message'], 
            signed_message['signature'],
            signed_message['certificate']
        )
        return cert_valid and sig_valid
```

## 🔬 Practical Exercises

### Exercise 1: Basic V2V Communication
Implement simple vehicle-to-vehicle messaging system.

**Tasks:**
- Create BSM message encoder/decoder
- Implement UDP-based communication
- Simulate multiple vehicles exchanging messages
- Visualize communication patterns

**Sample Implementation:**
```python
import socket
import json
import threading

class V2VCommunicator:
    def __init__(self, vehicle_id, port):
        self.vehicle_id = vehicle_id
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('localhost', port))
        self.neighbors = {}
        
    def send_bsm(self, position, speed, heading):
        bsm = {
            'vehicle_id': self.vehicle_id,
            'timestamp': time.time(),
            'position': position,
            'speed': speed,
            'heading': heading
        }
        
        message = json.dumps(bsm).encode('utf-8')
        # Broadcast to known neighbors
        for neighbor_addr in self.get_neighbor_addresses():
            self.socket.sendto(message, neighbor_addr)
    
    def receive_messages(self):
        while True:
            data, addr = self.socket.recvfrom(1024)
            message = json.loads(data.decode('utf-8'))
            self.process_bsm(message)
```

### Exercise 2: V2I Traffic Light Communication
Develop communication system between vehicles and traffic infrastructure.

**Objectives:**
- Implement SPAT message handling
- Create intersection geometry (MAP) messages
- Develop green light optimal speed advisory
- Handle traffic light preemption requests

### Exercise 3: Cooperative Collision Avoidance
Build cooperative safety application using V2X messaging.

**Features:**
- Collision threat assessment using V2V data
- Emergency brake warning propagation
- Intersection collision avoidance
- Vulnerable road user protection

### Exercise 4: Message Security Implementation
Implement security protocols for V2X communication.

**Components:**
- Certificate management system
- Message signing and verification
- Pseudonym certificate rotation
- Misbehavior detection

## 📈 Advanced Topics

### 5G-V2X Network Slicing
```python
class NetworkSlice:
    def __init__(self, slice_type, qos_requirements):
        self.slice_type = slice_type  # URLLC, eMBB, mMTC
        self.qos_requirements = qos_requirements
        self.allocated_resources = {}
        
    def allocate_resources(self, application_type):
        if application_type == "safety_critical":
            return {
                'latency': '1ms',
                'reliability': '99.999%',
                'bandwidth': '10Mbps'
            }
        elif application_type == "traffic_efficiency":
            return {
                'latency': '10ms',
                'reliability': '99.9%',
                'bandwidth': '1Mbps'
            }
```

### Edge Computing Integration
```python
class EdgeComputingNode:
    def __init__(self, node_id, processing_capacity):
        self.node_id = node_id
        self.processing_capacity = processing_capacity
        self.active_services = []
        
    def deploy_service(self, service_definition):
        """Deploy cooperative driving service at edge"""
        if self.has_capacity(service_definition):
            service = self.instantiate_service(service_definition)
            self.active_services.append(service)
            return service
        return None
        
    def process_v2x_data(self, v2x_messages):
        """Process aggregated V2X data for cooperative applications"""
        processed_data = {}
        for service in self.active_services:
            result = service.process(v2x_messages)
            processed_data[service.name] = result
        return processed_data
```

### Machine Learning for V2X
```python
class V2XDataAnalyzer:
    def __init__(self):
        self.traffic_predictor = self.load_traffic_model()
        self.anomaly_detector = self.load_anomaly_model()
        
    def analyze_traffic_patterns(self, v2x_data):
        """Analyze traffic patterns from aggregated V2X data"""
        features = self.extract_features(v2x_data)
        traffic_prediction = self.traffic_predictor.predict(features)
        anomalies = self.anomaly_detector.detect(features)
        
        return {
            'traffic_flow': traffic_prediction,
            'congestion_probability': traffic_prediction['congestion'],
            'detected_anomalies': anomalies
        }
```

### Quality of Service (QoS) Management
```python
class V2XQoSManager:
    def __init__(self):
        self.message_priorities = {
            'emergency': 0,      # Highest priority
            'safety': 1,
            'traffic_efficiency': 2,
            'infotainment': 3    # Lowest priority
        }
        
    def schedule_messages(self, message_queue):
        """Schedule messages based on priority and timing constraints"""
        sorted_messages = sorted(message_queue, 
                               key=lambda msg: (
                                   self.message_priorities[msg.type],
                                   msg.deadline
                               ))
        return sorted_messages
        
    def adapt_transmission_parameters(self, channel_conditions):
        """Adapt transmission parameters based on channel quality"""
        if channel_conditions['congestion'] > 0.8:
            return {
                'transmission_power': 'high',
                'message_rate': 'reduced',
                'coding_rate': 'robust'
            }
        else:
            return {
                'transmission_power': 'normal',
                'message_rate': 'standard',
                'coding_rate': 'efficient'
            }
```

## 📊 Performance Metrics

### Communication Performance
- **Packet Delivery Ratio (PDR)**: Successfully received messages
- **End-to-End Delay**: Message transmission time
- **Throughput**: Data rate achieved
- **Channel Utilization**: Spectrum efficiency

### Application Performance
- **Awareness Quality**: Accuracy of cooperative awareness
- **Safety Improvement**: Reduction in accident probability
- **Traffic Efficiency**: Travel time reduction
- **Fuel Consumption**: Environmental impact

### Security Metrics
- **Authentication Success Rate**: Valid message verification
- **Certificate Validation Time**: PKI processing overhead
- **Misbehavior Detection Accuracy**: False positive/negative rates
- **Privacy Protection Level**: Anonymity preservation

## 🎯 Real-World Applications

### Safety Applications
- **Emergency Electronic Brake Light (EEBL)**: Warn following vehicles
- **Forward Collision Warning (FCW)**: Prevent rear-end collisions
- **Intersection Movement Assist (IMA)**: Intersection safety
- **Left Turn Assist (LTA)**: Oncoming traffic warning

### Traffic Efficiency
- **Adaptive Traffic Signal Control**: Optimize signal timing
- **Eco-Speed Control**: Fuel-efficient speed recommendations
- **Platooning**: Coordinated vehicle following
- **Dynamic Route Guidance**: Real-time traffic optimization

### Mobility Services
- **Cooperative Cruise Control**: Vehicle coordination
- **Automated Valet Parking**: Infrastructure-assisted parking
- **Emergency Vehicle Preemption**: Priority lane clearing
- **Public Transport Priority**: Bus signal priority

## 📖 Implementation Examples

### MQTT-based V2X Communication
```python
import paho.mqtt.client as mqtt

class MQTTV2XClient:
    def __init__(self, client_id, broker_host):
        self.client = mqtt.Client(client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(broker_host, 1883, 60)
        
    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        # Subscribe to relevant V2X topics
        client.subscribe("v2x/safety/+")
        client.subscribe("v2x/traffic/+")
        
    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = json.loads(msg.payload.decode())
        self.process_v2x_message(topic, payload)
        
    def publish_bsm(self, vehicle_data):
        topic = f"v2x/safety/{vehicle_data['vehicle_id']}"
        payload = json.dumps(vehicle_data)
        self.client.publish(topic, payload)
```

### WebSocket V2X Server
```python
import asyncio
import websockets
import json

class V2XWebSocketServer:
    def __init__(self):
        self.connected_vehicles = set()
        
    async def register_vehicle(self, websocket, path):
        self.connected_vehicles.add(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.broadcast_message(data, websocket)
        finally:
            self.connected_vehicles.remove(websocket)
            
    async def broadcast_message(self, message, sender):
        if self.connected_vehicles:
            # Broadcast to all connected vehicles except sender
            recipients = self.connected_vehicles - {sender}
            await asyncio.gather(
                *[websocket.send(json.dumps(message)) for websocket in recipients],
                return_exceptions=True
            )
            
    def start_server(self, host='localhost', port=8765):
        return websockets.serve(self.register_vehicle, host, port)
```

## 📖 Additional Resources

### Standards Documents
- [IEEE 1609.2](https://standards.ieee.org/standard/1609_2-2016.html): Security Services
- [IEEE 1609.3](https://standards.ieee.org/standard/1609_3-2016.html): Networking Services
- [ETSI EN 302 637-2](https://www.etsi.org/deliver/etsi_en/302600_302699/30263702/01.04.01_60/en_30263702v010401p.pdf): CAM specification
- [3GPP TS 23.285](https://www.3gpp.org/ftp/Specs/archive/23_series/23.285/): V2X Services

### Key Papers
- [Vehicle-to-Everything (V2X) Communication: A Survey](https://ieeexplore.ieee.org/document/8303773)
- [5G-V2X: Standardization, Architecture, and Use Cases](https://ieeexplore.ieee.org/document/8570848)
- [Security and Privacy in V2X Communication](https://ieeexplore.ieee.org/document/8365159)

### Simulation Tools
- [SUMO](https://www.eclipse.org/sumo/): Traffic simulation with V2X
- [OMNeT++](https://omnetpp.org/): Network simulation framework
- [ns-3](https://www.nsnam.org/): Network simulator with V2X modules
- [CARLA](http://carla.org/): Autonomous driving simulator with V2X support

## 🎯 Assessment Criteria

- **Protocol Implementation** (35%): Correct V2X message handling
- **Application Development** (30%): Cooperative driving applications
- **Security Implementation** (20%): Message authentication and privacy
- **Performance Analysis** (15%): Communication efficiency evaluation

## 🔄 Integration with Other Modules

This module connects with:
- **Vehicle Guidance**: Cooperative path planning
- **Object Tracking**: Shared perception data
- **Occupancy Mapping**: Distributed mapping information
- **Image Segmentation**: Shared computer vision results

---

*V2X communication enables the transition from isolated autonomous vehicles to cooperative intelligent transportation systems, enhancing safety, efficiency, and mobility through connected technologies.*