document.addEventListener('DOMContentLoaded', () => {
  // --- STATE MANAGEMENT (from NeuralNetwork3D.tsx) ---
  let state = {
    isTraining: false,
    isTrainingComplete: false,
    activePanel: null,
    cameraMode: 'auto',
    animationSpeed: 1.0,
    epoch: 0,
    batch: 0,
    loss: 1.0,
    accuracy: 0.1,
    architecture: [
      { neurons: 4, activation: 'Linear', name: 'Input' },
      { neurons: 8, activation: 'ReLU', name: 'Hidden 1' },
      { neurons: 6, activation: 'ReLU', name: 'Hidden 2' },
      { neurons: 3, activation: 'Softmax', name: 'Output' }
    ],
    trainingParams: {
      learningRate: 0.01,
      batchSize: 32,
      epochs: 100,
      optimizer: 'Adam',
      lossFunction: 'CrossEntropy'
    }
  };

  // --- DOM ELEMENTS ---
  const canvas3d = document.getElementById('main3d');
  const archPanel = document.getElementById('architecture-panel');
  const paramPanel = document.getElementById('parameters-panel');
  const optPanel = document.getElementById('optimization-panel');
  const panelDetailsContainer = document.getElementById('panel-details-container');

  // --- 3D SETUP ---
  const renderer = new THREE.WebGLRenderer({ canvas: canvas3d, antialias: true, powerPreference: "high-performance" });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0a);

  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

  const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
  scene.add(ambientLight);
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(15, 15, 10);
  directionalLight.castShadow = true;
  scene.add(directionalLight);
  const fillLight = new THREE.DirectionalLight(0x4444ff, 0.3);
  fillLight.position.set(-10, 10, -10);
  scene.add(fillLight);

  const gridHelper = new THREE.GridHelper(40, 40, 0x444444, 0x222222);
  gridHelper.position.y = -8;
  scene.add(gridHelper);

  const lossAccCanvas = document.getElementById('loss-acc-canvas');

  let neurons = [];
  let connections = [];
  let animationProgress = 0;
  let trainingInterval = null;

  // --- CAMERA CONTROLS ---
  let isDragging = false;
  let prevMouse = { x: 0, y: 0 };
  let camState = { distance: 15, angleX: 0, angleY: 0.3, targetDistance: 15, targetAngleX: 0, targetAngleY: 0.3 };

  canvas3d.addEventListener('mousedown', e => { if (state.cameraMode === 'manual') { isDragging = true; prevMouse = { x: e.clientX, y: e.clientY }; } });
  canvas3d.addEventListener('mouseup', () => isDragging = false);
  canvas3d.addEventListener('mouseleave', () => isDragging = false);
  canvas3d.addEventListener('mousemove', e => {
    if (!isDragging || state.cameraMode !== 'manual') return;
    camState.targetAngleX += (e.clientX - prevMouse.x) * 0.008;
    camState.targetAngleY = Math.max(-Math.PI / 2.2, Math.min(Math.PI / 2.2, camState.targetAngleY + (e.clientY - prevMouse.y) * 0.008));
    prevMouse = { x: e.clientX, y: e.clientY };
  });
  canvas3d.addEventListener('wheel', e => {
    if (state.cameraMode !== 'manual') return;
    e.preventDefault();
    camState.targetDistance = Math.max(5, Math.min(80, camState.targetDistance + e.deltaY * 0.05));
  }, { passive: false });

  // --- CORE 3D LOGIC ---
  function getLayerConfig() {
    const colors = [0x4CAF50, 0x2196F3, 0x9C27B0, 0xFF9800, 0xF44336, 0x607D8B];
    return state.architecture.map((layer, idx) => {
      const neurons = layer.neurons;
      let gridSize;
      if (neurons <= 4) gridSize = [2, 2, 1];
      else if (neurons <= 9) gridSize = [3, 3, 1];
      else if (neurons <= 16) gridSize = [4, 4, 1];
      else if (neurons <= 36) gridSize = [6, 6, 1];
      else gridSize = [8, 8, Math.ceil(neurons / 64)];
      return { ...layer, color: colors[idx % colors.length], gridSize };
    });
  }

  function calculateOptimalDistance(layers) {
    const networkWidth = layers.length * 5.0; // Adjusted for new spacing
    const networkHeight = Math.max(...layers.map(layer => layer.gridSize[1])) * 2;
    return Math.max(15, Math.max(networkWidth, networkHeight) * 1.5);
  }

  function createNetwork() {
    neurons.forEach(layer => layer.forEach(n => { scene.remove(n); n.geometry.dispose(); n.material.dispose(); }));
    connections.forEach(c => { scene.remove(c); c.geometry.dispose(); c.material.dispose(); });
    neurons = [];
    connections = [];

    const layers = getLayerConfig();
    const layerSpacing = 5.0; // Increased spacing between layers
    const neuronGeo = new THREE.SphereGeometry(0.15, 8, 6);

    layers.forEach((layer, layerIndex) => {
      const layerNeurons = [];
      const [gridX, gridY, gridZ] = layer.gridSize;
      // Increased intra-layer spacing
      const spacingX = gridX > 1 ? 3.0 / (gridX - 1) : 0;
      const spacingY = gridY > 1 ? 3.0 / (gridY - 1) : 0;
      const spacingZ = gridZ > 1 ? 2.0 / (gridZ - 1) : 0;

      let neuronIndex = 0;
      for (let z = 0; z < gridZ && neuronIndex < layer.neurons; z++) {
        for (let y = 0; y < gridY && neuronIndex < layer.neurons; y++) {
          for (let x = 0; x < gridX && neuronIndex < layer.neurons; x++) {
            const material = new THREE.MeshPhongMaterial({ color: layer.color, transparent: true, opacity: 0.8, shininess: 100 });
            const neuron = new THREE.Mesh(neuronGeo, material);

            neuron.position.set(
              layerIndex * layerSpacing - (layers.length - 1) * layerSpacing / 2,
              (y * spacingY) - (gridY - 1) * spacingY / 2,
              (z * spacingZ) - (gridZ - 1) * spacingZ / 2 + (x * spacingX) - (gridX - 1) * spacingX / 2
            );

            neuron.userData = { fromLayer: layerIndex };
            scene.add(neuron);
            layerNeurons.push(neuron);
            neuronIndex++;
          }
        }
      }
      neurons.push(layerNeurons);
    });

    for (let i = 0; i < layers.length - 1; i++) {
      const currentLayer = neurons[i];
      const nextLayer = neurons[i+1];
      currentLayer.forEach(n1 => {
        nextLayer.forEach(n2 => {
          const density = Math.min(1.0, 50 / (currentLayer.length * nextLayer.length));
          if (Math.random() < density) {
            const geo = new THREE.BufferGeometry().setFromPoints([n1.position, n2.position]);
            const mat = new THREE.LineBasicMaterial({ color: 0x666666, transparent: true, opacity: 0.15 });
            const line = new THREE.Line(geo, mat);
            line.userData = { fromLayer: i, toLayer: i + 1 };
            scene.add(line);
            connections.push(line);
          }
        });
      });
    }
    const optimalDistance = calculateOptimalDistance(layers);
    camState.targetDistance = optimalDistance;
    camState.distance = optimalDistance;
    updateUI();
  }

  // --- UI RENDERING ---
  function updateUI() {
    const statusBadge = document.getElementById('status-badge');
    if (state.isTrainingComplete) { statusBadge.textContent = 'TRAINED'; statusBadge.style.backgroundColor = 'var(--blue)'; }
    else if (state.isTraining) { statusBadge.textContent = 'TRAINING'; statusBadge.style.backgroundColor = 'var(--green)'; }
    else { statusBadge.textContent = 'IDLE'; statusBadge.style.backgroundColor = '#4b5563'; }
    document.getElementById('network-stats').textContent = `${state.architecture.length} layers • ${state.architecture.reduce((s, l) => s + l.neurons, 0)} neurons`;

    document.getElementById('start-btn').classList.toggle('hidden', state.isTraining);
    document.getElementById('stop-btn').classList.toggle('hidden', !state.isTraining);
    document.getElementById('start-btn').disabled = state.isTrainingComplete;
    document.getElementById('cam-auto-btn').style.backgroundColor = state.cameraMode === 'auto' ? 'var(--purple)' : '#4b5563';
    document.getElementById('cam-manual-btn').style.backgroundColor = state.cameraMode === 'manual' ? 'var(--purple)' : '#4b5563';
    document.getElementById('speed-value').textContent = `${state.animationSpeed.toFixed(1)}x`;

    document.getElementById('epoch-value').textContent = state.epoch;
    document.getElementById('batch-value').textContent = state.batch;
    document.getElementById('loss-value').textContent = state.loss.toFixed(4);
    document.getElementById('accuracy-value').textContent = `${(state.accuracy * 100).toFixed(1)}%`;
    document.getElementById('accuracy-label').innerHTML = `Accuracy ${state.accuracy >= 0.9 ? '🎉' : ''}`;

    ['arch', 'param', 'opt'].forEach(p => document.getElementById(`${p}-btn`).classList.toggle('active', state.activePanel === p));
    archPanel.classList.toggle('visible', state.activePanel === 'architecture');
    paramPanel.classList.toggle('visible', state.activePanel === 'parameters');
    optPanel.classList.toggle('visible', state.activePanel === 'optimization');
    panelDetailsContainer.style.display = state.activePanel ? 'block' : 'none';
  }

  function renderArchitecturePanel() {
    let html = `<h3 style="color: var(--purple);">Network Architecture</h3><div class="space-y-2" style="display: flex; flex-direction: column; gap: 0.5rem;">`;
    state.architecture.forEach((layer, index) => {
      html += `<div class="layer-item">
        <div class="flex items-center" style="justify-content: space-between; margin-bottom: 0.5rem;">
          <span class="text-xs font-bold">${layer.name}</span>
          ${index > 0 && index < state.architecture.length - 1 ? `<button class="remove-layer-btn" data-index="${index}" style="color: var(--red); background: none; border: none; font-size: 1.2rem; cursor: pointer;">&times;</button>` : ''}
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
          <div><label>Neurons:</label><input type="number" min="1" max="64" value="${layer.neurons}" class="update-layer-input" data-index="${index}" data-field="neurons"></div>
          <div><label>Activation:</label><select class="update-layer-input" data-index="${index}" data-field="activation" ${index === 0 || index === state.architecture.length - 1 ? 'disabled' : ''}>
            ${['ReLU', 'Sigmoid', 'Tanh', 'Linear', 'Softmax'].map(opt => `<option value="${opt}" ${layer.activation === opt ? 'selected' : ''}>${opt}</option>`).join('')}
          </select></div>
        </div>
      </div>`;
    });
    html += `</div><button id="add-layer-btn" class="btn" style="width: 100%; margin-top: 0.75rem; background-color: var(--purple);">Add Hidden Layer</button>`;
    archPanel.innerHTML = html;
  }

  function renderParametersPanel() {
    paramPanel.innerHTML = `<h3 style="color: var(--blue);">Training Parameters</h3>
      <div style="display: flex; flex-direction: column; gap: 0.75rem;">
        <div><label>Epochs:</label><input type="number" step="10" min="10" max="1000" value="${state.trainingParams.epochs}" class="update-param-input" data-field="epochs"></div>
        <div><label>Learning Rate:</label><input type="number" step="0.001" min="0.001" max="1" value="${state.trainingParams.learningRate}" class="update-param-input" data-field="learningRate"></div>
        <div>
          <label>Batch Size:</label>
          <select class="update-param-input" data-field="batchSize" value="${state.trainingParams.batchSize}">
            <option>16</option><option>32</option><option>64</option><option>128</option>
          </select>
        </div>
        <div><label>Optimizer:</label><select class="update-param-input" data-field="optimizer" value="${state.trainingParams.optimizer}">
          <option>Adam</option><option>SGD</option><option>RMSprop</option><option>AdaGrad</option>
        </select></div>
        <div>
          <label>Loss Function:</label>
          <select class="update-param-input" data-field="lossFunction" value="${state.trainingParams.lossFunction}">
            <option>CrossEntropy</option><option>MSE</option>
          </select>
        </div>
      </div>`;
  }
  function renderOptimizationPanel() {
    if (state.activePanel !== 'optimization') return;
    document.getElementById('opt-info-optimizer').textContent = state.trainingParams.optimizer;
    document.getElementById('opt-info-loss-func').textContent = state.trainingParams.lossFunction;
  }

  let lossHistory = [], accHistory = [];
  function drawLossAccGraph() {
      const ctx = lossAccCanvas.getContext('2d');
      const w = lossAccCanvas.width, h = lossAccCanvas.height;
      ctx.clearRect(0, 0, w, h);

      // Draw Loss (Red)
      ctx.strokeStyle = 'var(--red)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      lossHistory.forEach((v, i) => {
          const x = (i / Math.max(1, lossHistory.length - 1)) * w;
          const y = h - v * h;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Draw Accuracy (Green)
      ctx.strokeStyle = 'var(--green)';
      ctx.beginPath();
      accHistory.forEach((v, i) => {
          const x = (i / Math.max(1, accHistory.length - 1)) * w;
          const y = h - v * h;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
  }

  // --- EVENT HANDLING ---
  function handlePanelToggle(panel) {
    state.activePanel = state.activePanel === panel ? null : panel;
    if (state.activePanel === 'architecture') renderArchitecturePanel();
    if (state.activePanel === 'parameters') renderParametersPanel();
    if (state.activePanel === 'optimization') renderOptimizationPanel();
    updateUI();
  }
  document.getElementById('arch-btn').addEventListener('click', () => handlePanelToggle('architecture'));
  document.getElementById('param-btn').addEventListener('click', () => handlePanelToggle('parameters'));
  document.getElementById('opt-btn').addEventListener('click', () => handlePanelToggle('optimization'));

  document.getElementById('cam-auto-btn').addEventListener('click', () => { state.cameraMode = 'auto'; updateUI(); });
  document.getElementById('cam-manual-btn').addEventListener('click', () => { state.cameraMode = 'manual'; updateUI(); });
  document.getElementById('speed-slider').addEventListener('input', e => { state.animationSpeed = parseFloat(e.target.value); updateUI(); });

  panelDetailsContainer.addEventListener('click', e => {
    if (e.target.id === 'add-layer-btn') {
      state.architecture.splice(-1, 0, { neurons: 8, activation: 'ReLU', name: `Hidden ${state.architecture.length - 1}` });
      createNetwork();
      renderArchitecturePanel();
    }
    if (e.target.classList.contains('remove-layer-btn')) {
      state.architecture.splice(parseInt(e.target.dataset.index), 1);
      createNetwork();
      renderArchitecturePanel();
    }
  });
  panelDetailsContainer.addEventListener('change', e => {
    if (e.target.classList.contains('update-layer-input')) {
      const { index, field } = e.target.dataset;
      const value = e.target.type === 'number' ? parseInt(e.target.value) : e.target.value;
      state.architecture[index][field] = value;
      createNetwork();
    }
    if (e.target.classList.contains('update-param-input')) {
      const { field } = e.target.dataset;
      const value = e.target.type === 'number' ? parseFloat(e.target.value) : e.target.value;
      state.trainingParams[field] = value;
    }
  });

  function startTraining() {
    Object.assign(state, { isTraining: true, isTrainingComplete: false, epoch: 0, batch: 0, loss: 1.0, accuracy: 0.1 });
    lossHistory = []; accHistory = [];
    animationProgress = 0;
    if (trainingInterval) clearInterval(trainingInterval);
    trainingInterval = setInterval(() => {
      if (!state.isTraining) return;
      state.batch++;
      state.loss = Math.max(0.001, state.loss * (1 - state.trainingParams.learningRate * 20) + (Math.random() - 0.5) * 0.01);
      const newAcc = Math.min(0.92, state.accuracy + (0.9 - state.accuracy) * state.trainingParams.learningRate * 15);

      lossHistory.push(state.loss);
      accHistory.push(newAcc);

      if (state.epoch >= state.trainingParams.epochs || (newAcc >= 0.9 && !state.isTrainingComplete)) {
        state.isTrainingComplete = true;
        state.isTraining = false;
        setTimeout(() => {
            neurons.flat().forEach(n => n.material.emissive.setHex(0x0099ff));
            connections.forEach(c => c.material.color.setHex(0x0099ff));
        }, 100);
      }
      state.accuracy = newAcc;
      if (state.batch % 10 === 0) state.epoch++;
      updateUI();
      if (state.activePanel === 'optimization') {
        renderOptimizationPanel();
        drawLossAccGraph();
      }
    }, 200 / state.animationSpeed);
    updateUI();
  }
  function stopTraining() {
    state.isTraining = false;
    if (trainingInterval) clearInterval(trainingInterval);
    updateUI();
  }
  function resetNetwork() {
    stopTraining();
    Object.assign(state, { isTrainingComplete: false, epoch: 0, batch: 0, loss: 1.0, accuracy: 0.1 });
    lossHistory = []; accHistory = [];
    createNetwork();
    updateUI();
  }
  document.getElementById('start-btn').addEventListener('click', startTraining);
  document.getElementById('stop-btn').addEventListener('click', stopTraining);
  document.getElementById('reset-btn').addEventListener('click', resetNetwork);

  // --- ANIMATION LOOP ---
  const clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    const deltaTime = clock.getElapsedTime();

    if (state.cameraMode === 'auto') {
      camState.distance = THREE.MathUtils.lerp(camState.distance, camState.targetDistance, 0.02);
      camera.position.set(Math.cos(deltaTime * 0.3) * camState.distance, camState.distance * 0.4, Math.sin(deltaTime * 0.3) * camState.distance);
      camera.lookAt(0, 0, 0);
    } else {
      const lerp = 0.08;
      camState.distance = THREE.MathUtils.lerp(camState.distance, camState.targetDistance, lerp);
      camState.angleX = THREE.MathUtils.lerp(camState.angleX, camState.targetAngleX, lerp);
      camState.angleY = THREE.MathUtils.lerp(camState.angleY, camState.targetAngleY, lerp);
      camera.position.set(
        Math.cos(camState.angleX) * Math.cos(camState.angleY) * camState.distance,
        Math.sin(camState.angleY) * camState.distance,
        Math.sin(camState.angleX) * Math.cos(camState.angleY) * camState.distance
      );
      camera.lookAt(0, 0, 0);
    }

    if (state.isTraining && !state.isTrainingComplete) {
      animationProgress = (deltaTime * state.animationSpeed * 0.5) % (state.architecture.length * 2 + 2);
      const progress = animationProgress;

      neurons.flat().forEach(n => {
        n.material.emissive.setHex(0);
        n.scale.set(1, 1, 1);
      });
      connections.forEach(c => {
        c.material.color.setHex(0x666666);
        c.material.opacity = 0.15;
      });

      // Refined wave animation from React component
      if (progress < state.architecture.length) { // FORWARD PASS (Green)
        for (let i = 0; i < state.architecture.length; i++) {
          const layerProgress = Math.max(0, Math.min(1, progress - i));
          if (layerProgress > 0) {
            const intensity = Math.sin(Math.PI * layerProgress) * 0.8 + 0.2;
            neurons[i]?.forEach(n => {
              n.material.emissive.setHex(0x00cc00).multiplyScalar(intensity);
              n.scale.setScalar(1 + intensity * 0.5);
            });
            connections.filter(c => c.userData.fromLayer === i).forEach(c => {
              c.material.color.setHex(0x00ff00);
              c.material.opacity = 0.2 + intensity * 0.6;
            });
          }
        }
      } else if (progress < state.architecture.length * 2) { // BACKWARD PASS (Red)
        const backProgress = progress - state.architecture.length;
        for (let i = state.architecture.length - 1; i >= 0; i--) {
          const reverseIdx = state.architecture.length - 1 - i;
          const layerProgress = Math.max(0, Math.min(1, backProgress - reverseIdx));
          if (layerProgress > 0) {
            const intensity = Math.sin(Math.PI * layerProgress) * 0.8 + 0.2;
            neurons[i]?.forEach(n => {
              n.material.emissive.setHex(0xcc0000).multiplyScalar(intensity);
              n.scale.setScalar(1 + intensity * 0.5);
            });
            connections.filter(c => c.userData.toLayer === i).forEach(c => {
              c.material.color.setHex(0xff0000);
              c.material.opacity = 0.2 + intensity * 0.6;
            });
          }
        }
      } else { // WEIGHT UPDATE (Blue)
        const updateIntensity = Math.sin((progress - state.architecture.length * 2) * Math.PI * 4);
        if (updateIntensity > 0) {
          const popScale = 1 + updateIntensity * 0.5;
          neurons.flat().forEach(n => {
            n.material.emissive.setHex(0x0099ff).multiplyScalar(updateIntensity * 0.8);
            n.scale.setScalar(popScale);
          });
          connections.forEach(c => {
            c.material.color.setHex(0x0099ff);
            c.material.opacity = 0.3 + updateIntensity * 0.5;
          });
        }
      }
    }

    renderer.render(scene, camera);
  }

  // --- INITIALIZATION ---
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  createNetwork();
  updateUI();
  animate();
});
