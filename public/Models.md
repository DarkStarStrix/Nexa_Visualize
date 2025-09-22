# Neural Network Architecture 3D Visualization Guides

## Table of Contents
1. [Convolutional Neural Networks (CNNs)](#cnn-guide)
2. [Mixture of Experts (MoE)](#moe-guide)
3. [YOLO Object Detection](#yolo-guide)
4. [Animation Techniques](#animation-techniques)

---

## CNN Guide {#cnn-guide}

### Architecture Overview
CNN process images through hierarchical feature extraction using convolution, pooling, and fully connected layers.

### 3D Visualization Structure

#### Layer Components:
```JavaScript
// Input Image Layer
Position: (0, -10, 0)
Geometry: BoxGeometry(width, height, 3) // RGB channels
Material: TextureLoader for actual image display
Animation: Rotate slowly to show 3D nature
Label: "Input Image (H×W×3)"

// Convolutional Layers
Position: (0, y_level, 0) where y_level increases by 4
Geometry: Multiple BoxGeometry representing feature maps
- Size decreases with depth (width/height)
- Depth increases (more channels)
Material: Different colors for different channels
Animation: Sliding kernel effect, feature activation waves
Label: "Conv Layer (filters×H×W)"

// Pooling Layers  
Position: Between conv layers
Geometry: Smaller boxes showing downsampling
Material: Semi-transparent blue
Animation: "Shrinking" effect to show dimension reduction
Label: "Max/Avg Pool (H/2×W/2)"

// Fully Connected Layers
Position: (0, 8, 0)
Geometry: Linear arrangement of spheres
Material: Gradient from blue to green
Animation: Neuron activation pulses
Label: "FC Layer (N neurons)"
```

#### Key Visualizations:

**1. Convolution Operation:**
```javascript
// 3D Kernel sliding across feature map
const kernelGeometry = new THREE.BoxGeometry(0.5, 0.5, 0.3);
const kernelMaterial = new THREE.MeshLambertMaterial({
    color: 0xff0000,
    transparent: true,
    opacity: 0.7
});

// Animation: Kernel slides across input
function animateConvolution(kernel, inputLayer) {
    const positions = generateSlidingPositions(inputLayer);
    // GSAP timeline for smooth sliding motion
    const timeline = gsap.timeline({repeat: -1});
    positions.forEach((pos, i) => {
        timeline.to(kernel.position, {
            duration: 0.2,
            x: pos.x,
            y: pos.y,
            ease: "none"
        });
    });
}
```

**2. Feature Map Activation:**
```JavaScript
// Show how different filters detect different features
function visualizeFeatureActivation(featureMap, filterType) {
    const activationIntensity = calculateActivation(filterType);
    
    // Color coding for different features
    const colors = {
        'edges': 0x00ff00,
        'corners': 0xff0000,
        'textures': 0x0000ff
    };
    
    // Animate based on feature strength
    gsap.to(featureMap.material, {
        color: colors[filterType],
        emissiveIntensity: activationIntensity,
        duration: 0.5,
        repeat: -1,
        yoyo: true
    });
}
```

**3. Pooling Visualization:**
```javascript
// Show max pooling selection
function animateMaxPooling(inputRegion, outputNeuron) {
    // Highlight maximum value in region
    const maxNeuron = findMaxInRegion(inputRegion);
    
    // Create connection line
    const connectionGeometry = new THREE.BufferGeometry().setFromPoints([
        maxNeuron.position,
        outputNeuron.position
    ]);
    
    // Animate value transfer
    createFlowingParticle(maxNeuron.position, outputNeuron.position);
}
```

### Training Process Animation:
1. **Forward Pass**: Show data flowing through layers
2. **Feature Detection**: Highlight activated features
3. **Pooling**: Animate dimension reduction
4. **Classification**: Show final decision process

---

## MOE Guide {#moe-guide}

### Architecture Overview
MoE uses multiple specialized expert networks with a gating mechanism that routes tokens to the most relevant experts, enabling larger model capacity without proportional computational cost.

### 3D Visualization Structure:

#### Expert Networks Layout:
```javascript
// Circular arrangement of experts
const numExperts = 8;
const radius = 5;

for (let i = 0; i < numExperts; i++) {
    const angle = (i / numExperts) * Math.PI * 2;
    const expertPos = new THREE.Vector3(
        Math.cos(angle) * radius,
        0,
        Math.sin(angle) * radius
    );
    
    // Expert Network
    const expertGeometry = new THREE.CylinderGeometry(0.8, 0.8, 2, 8);
    const expertMaterial = new THREE.MeshLambertMaterial({
        color: getExpertColor(i),
        transparent: true,
        opacity: 0.3 // Initially inactive
    });
    
    const expert = new THREE.Mesh(expertGeometry, expertMaterial);
    expert.position.copy(expertPos);
    expert.userData = {
        type: 'expert',
        id: i,
        specialization: getExpertSpecialization(i),
        activity: 0
    };
    
    scene.add(expert);
}

// Central Gating Network
const gateGeometry = new THREE.SphereGeometry(1, 16, 12);
const gateMaterial = new THREE.MeshLambertMaterial({
    color: 0xFFD700,
    emissive: 0x444400
});
const gate = new THREE.Mesh(gateGeometry, gateMaterial);
gate.position.set(0, 0, 0);
gate.userData = { type: 'gate' };
scene.add(gate);
```

#### Token Routing Visualization:
```javascript
function animateTokenRouting(token, gatingScores) {
    const timeline = gsap.timeline();
    
    // 1. Token arrives at gate
    timeline.to(token.position, {
        x: 0, y: 2, z: 0,
        duration: 1,
        ease: "power2.out"
    });
    
    // 2. Gating network computes scores
    timeline.call(() => {
        animateGatingComputation(gate, gatingScores);
    });
    
    // 3. Route to top-k experts
    const topKExperts = getTopKExperts(gatingScores, 2);
    
    topKExperts.forEach((expertId, index) => {
        timeline.call(() => {
            routeToExpert(token.clone(), expertId, gatingScores[expertId]);
        }, null, 2 + index * 0.5);
    });
}

function routeToExpert(token, expertId, weight) {
    const expert = getExpertById(expertId);
    
    // Activate expert based on routing weight
    gsap.to(expert.material, {
        opacity: 0.3 + (weight * 0.7),
        emissiveIntensity: weight * 0.5,
        duration: 0.5
    });
    
    // Create routing line
    const routingLine = createRoutingLine(gate.position, expert.position, weight);
    
    // Animate token to expert
    gsap.to(token.position, {
        x: expert.position.x,
        y: expert.position.y + 1.5,
        z: expert.position.z,
        duration: 1,
        ease: "power2.inOut"
    });
    
    // Process at expert
    setTimeout(() => {
        animateExpertProcessing(expert, token);
    }, 1000);
}

function animateExpertProcessing(expert, token) {
    // Expert processing animation
    const originalScale = expert.scale.clone();
    
    gsap.to(expert.scale, {
        x: 1.2, y: 1.2, z: 1.2,
        duration: 0.3,
        yoyo: true,
        repeat: 1,
        onComplete: () => {
            // Send processed token back
            animateTokenReturn(token, expert);
        }
    });
}
```

#### Load Balancing Visualization:
```javascript
function visualizeLoadBalancing(expertUsage) {
    experts.forEach((expert, i) => {
        const usage = expertUsage[i];
        const targetHeight = 2 * usage; // Scale height by usage
        
        gsap.to(expert.scale, {
            y: targetHeight,
            duration: 1,
            ease: "power2.out"
        });
        
        // Color intensity based on usage
        const intensity = Math.min(usage * 2, 1);
        gsap.to(expert.material, {
            emissiveIntensity: intensity * 0.5,
            duration: 1
        });
    });
}
```

---

## YOLO Guide {#yolo-guide}

### Architecture Overview
YOLO (You Only Look Once) performs object detection in a single forward pass by dividing images into grids and predicting bounding boxes and classes simultaneously.

### 3D Visualization Structure:

#### Grid-Based Detection:
```javascript
// Input Image Grid
const gridSize = 7; // 7x7 grid for YOLO v1
const cellSize = 1;

for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j++) {
        const cellGeometry = new THREE.PlaneGeometry(cellSize, cellSize);
        const cellMaterial = new THREE.MeshBasicMaterial({
            color: 0x444444,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide
        });
        
        const cell = new THREE.Mesh(cellGeometry, cellMaterial);
        cell.position.set(
            i * cellSize - gridSize/2,
            j * cellSize - gridSize/2,
            0
        );
        
        cell.userData = {
            type: 'grid_cell',
            gridX: i,
            gridY: j,
            confidence: 0,
            boxes: []
        };
        
        scene.add(cell);
    }
}

// Backbone Network (Feature Extraction)
const backbonePos = new THREE.Vector3(0, 0, -8);
createCNNBackbone(backbonePos);

// Detection Head
const detectionHeadPos = new THREE.Vector3(0, 0, -4);
createDetectionHead(detectionHeadPos);
```

#### Bounding Box Prediction:
```javascript
function animateBoundingBoxPrediction(gridCell, predictions) {
    // Each cell predicts multiple bounding boxes
    const boxColors = [0xFF0000, 0x00FF00]; // Red and Green for 2 boxes
    
    predictions.boxes.forEach((box, index) => {
        const boxGeometry = new THREE.BoxGeometry(
            box.width, box.height, 0.1
        );
        
        const boxMaterial = new THREE.MeshBasicMaterial({
            color: boxColors[index],
            transparent: true,
            opacity: box.confidence,
            wireframe: true
        });
        
        const boxMesh = new THREE.Mesh(boxGeometry, boxMaterial);
        boxMesh.position.set(
            gridCell.position.x + box.centerX,
            gridCell.position.y + box.centerY,
            gridCell.position.z + 0.1
        );
        
        // Animate confidence with pulsing
        gsap.to(boxMaterial, {
            opacity: box.confidence * 0.8,
            duration: 1,
            repeat: -1,
            yoyo: true
        });
        
        scene.add(boxMesh);
        gridCell.userData.boxes.push(boxMesh);
    });
}

function animateClassPrediction(gridCell, classPredictions) {
    // Show class probabilities as colored indicators
    const classColors = {
        'person': 0xFF0000,
        'car': 0x00FF00,
        'dog': 0x0000FF,
        'bicycle': 0xFFFF00
    };
    
    Object.entries(classPredictions).forEach(([className, probability], index) => {
        if (probability > 0.1) { // Only show significant predictions
            const indicator = createClassIndicator(
                className, 
                probability, 
                classColors[className]
            );
            
            indicator.position.set(
                gridCell.position.x + (index * 0.2 - 0.3),
                gridCell.position.y + 0.6,
                0.2
            );
            
            scene.add(indicator);
        }
    });
}
```

#### Non-Maximum Suppression:
```javascript
function animateNMS(detectedBoxes) {
    const timeline = gsap.timeline();
    
    // Phase 1: Show all detections
    timeline.call(() => {
        detectedBoxes.forEach(box => {
            box.material.opacity = box.userData.confidence;
        });
    });
    
    // Phase 2: Sort boxes by confidence
    timeline.call(() => {
        detectedBoxes.sort((a, b) => b.userData.confidence - a.userData.confidence);
        // Visually reorder or highlight the highest confidence box
    }, null, "+=1");

    // Phase 3: Iterate and discard overlapping boxes
    timeline.call(() => {
        for (let i = 0; i < detectedBoxes.length; i++) {
            if (detectedBoxes[i].userData.isKept === false) continue;
            
            for (let j = i + 1; j < detectedBoxes.length; j++) {
                if (detectedBoxes[j].userData.isKept === false) continue;
                
                const iou = calculateIOU(detectedBoxes[i], detectedBoxes[j]);
                if (iou > 0.5) { // IoU threshold
                    // Fade out the lower confidence box
                    gsap.to(detectedBoxes[j].material, { opacity: 0.1, duration: 0.5 });
                    detectedBoxes[j].userData.isKept = false;
                }
            }
        }
    }, null, "+=1");

    // Phase 4: Show final detections
    timeline.call(() => {
        detectedBoxes.forEach(box => {
            if (box.userData.isKept !== false) {
                // Make final boxes solid and bright
                gsap.to(box.material, { opacity: 1.0, wireframe: false, duration: 0.5 });
            } else {
                // Remove discarded boxes
                scene.remove(box);
            }
        });
    }, null, "+=1.5");
}
```

---

## Animation Techniques {#animation-techniques}

### Overview
This section details reusable animation patterns for visualizing neural network operations.

### 1. Data Flow Particles
Represent data tokens, gradients, or information moving through the network.

```javascript
function createFlowingParticle(startPos, endPos, color = 0x00ffff, speed = 1) {
    const particleGeometry = new THREE.SphereGeometry(0.08, 6, 6);
    const particleMaterial = new THREE.MeshBasicMaterial({
        color: color,
        transparent: true
    });
    
    const particle = new THREE.Mesh(particleGeometry, particleMaterial);
    particle.position.copy(startPos);
    scene.add(particle);
    
    gsap.to(particle.position, {
        x: endPos.x,
        y: endPos.y,
        z: endPos.z,
        duration: 2 / speed,
        ease: "power1.inOut",
        onComplete: () => scene.remove(particle)
    });
}
```

### 2. Activation Visualization
Show neuron or layer activation using light, color, or scale.

```javascript
function animateActivation(targetObject, intensity, color = 0x00ff00) {
    // Method 1: Emissive intensity
    gsap.to(targetObject.material, {
        emissiveIntensity: intensity,
        duration: 0.3,
        yoyo: true,
        repeat: 1
    });
    
    // Method 2: Scaling (pulsing)
    const originalScale = targetObject.scale.clone();
    gsap.to(targetObject.scale, {
        x: originalScale.x * (1 + intensity * 0.5),
        y: originalScale.y * (1 + intensity * 0.5),
        z: originalScale.z * (1 + intensity * 0.5),
        duration: 0.3,
        yoyo: true,
        repeat: 1
    });
}
```

### 3. Weight Update Visualization
Indicate changes in connection weights during backpropagation.

```javascript
function visualizeWeightUpdate(connectionLine, updateMagnitude) {
    const originalColor = connectionLine.material.color.clone();
    const updateColor = updateMagnitude > 0 ? new THREE.Color(0x00ff00) : new THREE.Color(0xff0000);
    
    connectionLine.material.color.lerp(updateColor, Math.abs(updateMagnitude));
    
    // Fade back to original color
    gsap.to(connectionLine.material.color, {
        r: originalColor.r,
        g: originalColor.g,
        b: originalColor.b,
        duration: 1.5
    });
}
```

### 4. Component Highlighting
Draw attention to the currently active part of the network.

```javascript
function highlightComponent(component, duration = 1) {
    // Create a highlight outline or halo effect
    const outlineMaterial = new THREE.MeshBasicMaterial({
        color: 0xffff00,
        side: THREE.BackSide
    });
    const outlineMesh = new THREE.Mesh(component.geometry, outlineMaterial);
    outlineMesh.position.copy(component.position);
    outlineMesh.scale.multiplyScalar(1.1);
    scene.add(outlineMesh);
    
    // Fade out the highlight
    gsap.to(outlineMesh.material, {
        opacity: 0,
        duration: duration,
        onComplete: () => scene.remove(outlineMesh)
    });
}
```

### 5. Using GSAP for Timelines
Orchestrate complex, multi-step animations with precision.

```javascript
// Example of a forward pass animation timeline
const forwardPass = gsap.timeline();

forwardPass
    .addLabel("input")
    .call(() => animateActivation(inputLayer, 1.0), null, "input")
    .call(() => createFlowingParticle(inputLayer.position, hiddenLayer1.position), null, "input")
    
    .addLabel("hidden1", "+=1")
    .call(() => animateActivation(hiddenLayer1, 0.8), null, "hidden1")
    .call(() => createFlowingParticle(hiddenLayer1.position, outputLayer.position), null, "hidden1")
    
    .addLabel("output", "+=1")
    .call(() => animateActivation(outputLayer, 1.0, 0x0000ff), null, "output");
```
