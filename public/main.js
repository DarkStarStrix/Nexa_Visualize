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
    architecture: {
      fnn: [
        { neurons: 4, activation: 'Linear', name: 'Input' },
        { neurons: 8, activation: 'ReLU', name: 'Hidden 1' },
        { neurons: 6, activation: 'ReLU', name: 'Hidden 2' },
        { neurons: 3, activation: 'Softmax', name: 'Output' }
      ],
      transformer: { encoderBlocks: 2, decoderBlocks: 2 },
      cnn: { variant: 'LeNet', convLayers: 2, fcLayers: 2 }, // Default to LeNet
      rnn: { timeSteps: 5, hiddenSize: 4 },
      lstm: { timeSteps: 4 },
      gru: { timeSteps: 4 },
      moe: { experts: 4 },
      gan: { generatorLayers: 3, discriminatorLayers: 3 },
    },
    trainingParams: {
      learningRate: 0.01,
      batchSize: 32,
      epochs: 100,
      optimizer: 'Adam',
      lossFunction: 'CrossEntropy'
    },
    theme: localStorage.getItem('theme') || 'dark',
    archType: 'fnn', // NEW: selected architecture type
    animationStep: 0, // For multi-step animations
  };

  // --- DOM ELEMENTS ---
  const canvas3d = document.getElementById('main3d');
  const archPanel = document.getElementById('architecture-panel');
  const paramPanel = document.getElementById('parameters-panel');
  const optPanel = document.getElementById('optimization-panel');
  const settingsPanel = document.getElementById('settings-panel');
  const panelDetailsContainer = document.getElementById('panel-details-container');
  const archTypeSelect = document.getElementById('arch-type-select');
  const archLoadingSpinner = document.getElementById('arch-loading-spinner');

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

  let gridHelper = null;

  const lossAccCanvas = document.getElementById('loss-acc-canvas');

  let neurons = [];
  let connections = [];
  let animationProgress = 0;
  let trainingInterval = null;
  let transformerObjects = []; // For custom transformer meshes
  let particles = []; // For data flow animation

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

  // --- ARCHITECTURE VISUALIZATION HELPERS ---

  function createTextLabel(text, position, size = 0.8, color = '#ffffff') {
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      const font = `bold ${size * 20}px Arial`;
      context.font = font;
      const metrics = context.measureText(text);
      const textWidth = metrics.width;

      canvas.width = textWidth + 8; // Add padding for stroke
      canvas.height = size * 25 + 8;
      context.font = font;
      context.textAlign = 'center';
      context.textBaseline = 'middle';

      // Draw outline
      context.strokeStyle = 'black';
      context.lineWidth = 4;
      context.strokeText(text, canvas.width / 2, canvas.height / 2);

      // Fill text
      context.fillStyle = color;
      context.fillText(text, canvas.width / 2, canvas.height / 2);

      const texture = new THREE.CanvasTexture(canvas);
      texture.minFilter = THREE.LinearFilter;
      texture.wrapS = THREE.ClampToEdgeWrapping;
      texture.wrapT = THREE.ClampToEdgeWrapping;

      const spriteMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true });
      const sprite = new THREE.Sprite(spriteMaterial);
      sprite.scale.set((size * canvas.width) / 20, (size * canvas.height) / 20, 1.0);
      sprite.position.copy(position);
      transformerObjects.push(sprite); // Add labels to be cleared
      scene.add(sprite);
      return sprite;
  }

  function getFNNLayerConfig() {
    // ...existing code for FNN...
    // (same as getLayerConfig, but renamed for clarity)
    // ...existing code...
    const colors = [0x4CAF50, 0x2196F3, 0x9C27B0, 0xFF9800, 0xF44336, 0x607D8B];
    return state.architecture.fnn.map((layer, idx) => {
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

  function getMoEConfig() {
    // Example: Input, gating, several experts, combine, output
    const numExperts = state.architecture.moe.experts;
    const baseLayers = [
      { name: 'Input', neurons: 6, activation: 'Linear' },
      { name: 'Gating', neurons: numExperts, activation: 'Softmax' },
      ...Array(numExperts).fill().map((_, i) => ({ name: `Expert ${i+1}`, neurons: 6, activation: 'ReLU' })),
      { name: 'Combine', neurons: 6, activation: 'Linear' },
      { name: 'Output', neurons: 3, activation: 'Softmax' }
    ];
    const colors = [0x22c55e, 0xF59E42, 0x3b82f6, 0xa855f7, 0xF44336, 0x607D8B];
    return baseLayers.map((layer, idx) => ({
      ...layer,
      color: colors[idx % colors.length],
      gridSize: [2, 3, 1]
    }));
  }

  function getUNetConfig() {
    // Example: Input, down blocks, bottleneck, up blocks, output
    const numBlocks = state.architecture.unet.depth;
    const baseLayers = [
      { name: 'Input', neurons: 4, activation: 'Linear' },
      ...Array(numBlocks).fill().map((_, i) => ({ name: `Down ${i+1}`, neurons: 8, activation: 'ReLU' })),
      { name: 'Bottleneck', neurons: 16, activation: 'ReLU' },
      ...Array(numBlocks).fill().map((_, i) => ({ name: `Up ${i+1}`, neurons: 8, activation: 'ReLU' })),
      { name: 'Output', neurons: 2, activation: 'Sigmoid' }
    ];
    const colors = [0x3b82f6, 0x22d3ee, 0x22c55e, 0xa855f7, 0xF44336, 0x607D8B];
    return baseLayers.map((layer, idx) => ({
      ...layer,
      color: colors[idx % colors.length],
      gridSize: [2, Math.ceil(layer.neurons/2), 1]
    }));
  }

  function getCurrentLayerConfig() {
    switch (state.archType) {
      case 'transformer':
      case 'cnn':
      case 'rnn':
      case 'lstm':
      case 'gru':
      case 'gan':
        return []; // These have their own builders
      case 'moe': return getMoEConfig();
      case 'unet': return getUNetConfig();
      case 'fnn':
      default: return getFNNLayerConfig();
    }
  }

  function calculateOptimalDistance(layers) {
    if (state.archType === 'transformer' || state.archType === 'gan') {
        return 50;
    }
    if (state.archType === 'cnn') {
        return 50;
    }
    if (state.archType === 'rnn' || state.archType === 'lstm' || state.archType === 'gru') {
        return 30;
    }
    if (state.archType === 'moe') {
        return 20;
    }
    const networkWidth = layers.length * 5.0; // Adjusted for new spacing
    const networkHeight = Math.max(...layers.map(layer => layer.gridSize[1])) * 2;
    return Math.max(15, Math.max(networkWidth, networkHeight) * 1.5);
  }

  function createGenericNetwork(layers) {
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
              (y * spacingY) - (gridY - 1) * spacingY / 2, // Centered at y=0 to float
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

  // --- TRANSFORMER VISUALIZATION ---

  function createTransformerComponent(name, position, geometry, color, userData = {}) {
      const mat = new THREE.MeshPhongMaterial({ color, transparent: true, opacity: 0.85, shininess: 50 });
      const mesh = new THREE.Mesh(geometry, mat);
      // Ensure bottom of mesh is at position.y
      const height = geometry.parameters.height || 0;
      mesh.position.set(position.x, position.y + height / 2, position.z);

      mesh.userData = { name, ...userData, base_y: position.y };
      scene.add(mesh);
      transformerObjects.push(mesh);
      if (name) {
          const labelPos = position.clone();
          labelPos.y += height + 0.5;
          createTextLabel(name, labelPos, 0.6);
      }
      return mesh;
  }

  function createCurvedConnection(start, end, color, offset = -3) {
      const curve = new THREE.QuadraticBezierCurve3(
          start,
          new THREE.Vector3((start.x + end.x) / 2 + offset, (start.y + end.y) / 2, (start.z + end.z) / 2),
          end
      );
      const points = curve.getPoints(20);
      const geo = new THREE.BufferGeometry().setFromPoints(points);
      const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7, linewidth: 2 });
      const line = new THREE.Line(geo, mat);
      scene.add(line);
      transformerObjects.push(line);
  }

  function createEncoderLayer(x, y, index) {
      const y_base = y;
      // Multi-head attention
      const attention = createTransformerComponent('Multi-Head Attention', new THREE.Vector3(x, y_base, 0), new THREE.BoxGeometry(2.5, 1.5, 0.8), 0xffa500, { type: 'attention', stack: 'encoder', layer: index });
      // Add & Norm
      createCurvedConnection(new THREE.Vector3(x, y_base - 0.75, 0), new THREE.Vector3(x, y_base + 0.75, 0), 0xffff00, 2);
      // Feed Forward
      const ff_y = y_base + 2.5;
      const ff = createTransformerComponent('Feed Forward', new THREE.Vector3(x, ff_y, 0), new THREE.BoxGeometry(2.5, 1, 0.8), 0x87ceeb, { type: 'ff', stack: 'encoder', layer: index });
      // Add & Norm
      createCurvedConnection(new THREE.Vector3(x, ff_y - 0.5, 0), new THREE.Vector3(x, ff_y + 0.5, 0), 0xffff00, 2);
  }

  function createDecoderLayer(x, y, index) {
      const y_base = y;
      // Masked Multi-head attention
      const masked_y = y_base - 1;
      const maskedAttention = createTransformerComponent('Masked MHA', new THREE.Vector3(x, masked_y, 0), new THREE.BoxGeometry(2.5, 1.2, 0.8), 0x9932cc, { type: 'masked_attention', stack: 'decoder', layer: index });
      createCurvedConnection(new THREE.Vector3(x, masked_y - 0.6, 0), new THREE.Vector3(x, masked_y + 0.6, 0), 0xffff00, 2);

      // Cross attention
      const cross_y = y_base + 1.5;
      const crossAttention = createTransformerComponent('Cross-Attention', new THREE.Vector3(x, cross_y, 0), new THREE.BoxGeometry(2.5, 1.2, 0.8), 0xdc143c, { type: 'cross_attention', stack: 'decoder', layer: index });
      createCurvedConnection(new THREE.Vector3(x, cross_y - 0.6, 0), new THREE.Vector3(x, cross_y + 0.6, 0), 0xffff00, 2);

      // Feed Forward
      const ff_y = y_base + 4;
      const ff = createTransformerComponent('Feed Forward', new THREE.Vector3(x, ff_y, 0), new THREE.BoxGeometry(2.5, 1, 0.8), 0x87ceeb, { type: 'ff', stack: 'decoder', layer: index });
      createCurvedConnection(new THREE.Vector3(x, ff_y - 0.5, 0), new THREE.Vector3(x, ff_y + 0.5, 0), 0xffff00, 2);
  }

  function createTransformerNetwork() {
      const { encoderBlocks, decoderBlocks } = state.architecture.transformer;
      const enc_x = -8;
      const dec_x = 8;
      const y_base = 0; // Float above grid

      // Embeddings
      createTransformerComponent('Input Embedding', new THREE.Vector3(enc_x, y_base - 6, 0), new THREE.BoxGeometry(3, 1, 1), 0xffb6c1, { type: 'embedding', stack: 'encoder' });
      createTransformerComponent('Output Embedding', new THREE.Vector3(dec_x, y_base - 6, 0), new THREE.BoxGeometry(3, 1, 1), 0xffb6c1, { type: 'embedding', stack: 'decoder' });

      // Encoder Stack
      for (let i = 0; i < encoderBlocks; i++) {
          createEncoderLayer(enc_x, y_base - 2 + i * 5, i);
      }

      // Decoder Stack
      for (let i = 0; i < decoderBlocks; i++) {
          createDecoderLayer(dec_x, y_base - 2 + i * 7, i);
      }

      // Cross-Attention Connections
      for (let i = 0; i < Math.min(encoderBlocks, decoderBlocks); i++) {
          const start_y = y_base - 2 + i * 5;
          const end_y = y_base - 2 + i * 7 + 1.5;
          const start = new THREE.Vector3(enc_x + 1.25, start_y, 0);
          const end = new THREE.Vector3(dec_x - 1.25, end_y, 0);
          const geo = new THREE.BufferGeometry().setFromPoints([start, end]);
          const mat = new THREE.LineBasicMaterial({ color: 0xff0000, transparent: true, opacity: 0.6 });
          const line = new THREE.Line(geo, mat);
          scene.add(line);
          transformerObjects.push(line);
      }

      // Output Layers
      const last_decoder_y = y_base - 2 + (decoderBlocks - 1) * 7 + 4;
      createTransformerComponent('Linear', new THREE.Vector3(dec_x, last_decoder_y + 2, 0), new THREE.BoxGeometry(2, 0.8, 0.8), 0x808080, { type: 'linear' });
      createTransformerComponent('Softmax', new THREE.Vector3(dec_x, last_decoder_y + 3.5, 0), new THREE.SphereGeometry(0.6, 16, 12), 0x32cd32, { type: 'softmax' });
  }

  // --- REFINED CNN ARCHITECTURE BUILDERS ---

  function createLeNet() {
      let z = -15;
      const y = -8; // Bedrock
      let prevComp = null;
      let layer = 0;

      prevComp = createTransformerComponent('Input (32x32)', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(6, 6, 0.5), 0x3b82f6, {type: 'conv', layer: layer++});
      z += 6;
      let currentComp = createTransformerComponent('C1: Conv', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(5.5, 5.5, 1), 0xa855f7, {type: 'conv', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 5;
      currentComp = createTransformerComponent('S2: Pool', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(5, 5, 1), 0x22d3ee, {type: 'pool', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 6;
      currentComp = createTransformerComponent('C3: Conv', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(4, 4, 1.5), 0xa855f7, {type: 'conv', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 5;
      currentComp = createTransformerComponent('S4: Pool', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(3.5, 3.5, 1.5), 0x22d3ee, {type: 'pool', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 6;
      currentComp = createTransformerComponent('C5: Conv', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(3, 3, 2), 0xa855f7, {type: 'conv', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 5;
      currentComp = createTransformerComponent('F6: FC', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(1, 1, 3), 0x22c55e, {type: 'fc', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 4;
      currentComp = createTransformerComponent('Output: FC', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(1, 1, 2), 0xef4444, {type: 'output', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0);
  }

  function createAlexNet() {
      let z = -25;
      const y = -8;
      let prevComp = null;
      let layer = 0;

      prevComp = createTransformerComponent('Input (227x227)', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(8, 8, 0.5), 0x3b82f6, {type: 'conv', layer: layer++});
      z += 7;
      let currentComp = createTransformerComponent('Conv 1', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(7, 7, 2), 0xa855f7, {type: 'conv', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 5;
      currentComp = createTransformerComponent('Pool 1', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(6, 6, 2), 0x22d3ee, {type: 'pool', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 6;
      currentComp = createTransformerComponent('Conv 2', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(5, 5, 3), 0xa855f7, {type: 'conv', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 5;
      currentComp = createTransformerComponent('Pool 2', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(4, 4, 3), 0x22d3ee, {type: 'pool', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 5;
      currentComp = createTransformerComponent('Conv 3', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(3.5, 3.5, 4), 0xa855f7, {type: 'conv', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 4;
      currentComp = createTransformerComponent('Conv 4', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(3, 3, 4.5), 0xa855f7, {type: 'conv', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 4;
      currentComp = createTransformerComponent('Conv 5', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(2.5, 2.5, 5), 0xa855f7, {type: 'conv', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 5;
      currentComp = createTransformerComponent('Pool 3', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(2, 2, 5), 0x22d3ee, {type: 'pool', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 5;
      currentComp = createTransformerComponent('FC 6', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(1, 1, 5), 0x22c55e, {type: 'fc', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 5;
      currentComp = createTransformerComponent('FC 7', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(1, 1, 4), 0x22c55e, {type: 'fc', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 5;
      currentComp = createTransformerComponent('Output', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(1, 1, 3), 0xef4444, {type: 'output', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0);
  }

  function createVGGNet() {
      let z = -30;
      const y = -8;
      let layer = 0;
      let prevComp = null;
      let currentComp = null;

      const addConvBlock = (count, size, depth, pool = true) => {
          for (let i = 0; i < count; i++) {
              currentComp = createTransformerComponent(`Conv`, new THREE.Vector3(0, y, z), new THREE.BoxGeometry(size, size, depth), 0xa855f7, {type: 'conv', layer: layer++});
              if(prevComp) createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0);
              prevComp = currentComp;
              z += 3;
          }
          if (pool) {
              currentComp = createTransformerComponent('Pool', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(size * 0.8, size * 0.8, depth), 0x22d3ee, {type: 'pool', layer: layer++});
              createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0);
              prevComp = currentComp;
              z += 5;
          }
      };

      prevComp = createTransformerComponent('Input (224x224)', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(8, 8, 0.5), 0x3b82f6, {type: 'conv', layer: layer++});
      z += 6;

      addConvBlock(2, 8, 1); // Block 1
      addConvBlock(2, 7, 1.5); // Block 2
      addConvBlock(3, 6, 2); // Block 3

      // FC Layers
      currentComp = createTransformerComponent('FC-1', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(1, 1, 5), 0x22c55e, {type: 'fc', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 5;
      currentComp = createTransformerComponent('FC-2', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(1, 1, 4), 0x22c55e, {type: 'fc', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 5;
      currentComp = createTransformerComponent('Output', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(1, 1, 3), 0xef4444, {type: 'output', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0);
  }

  function createResNet() {
      let z = -20;
      const y = -8;
      const input = createTransformerComponent('Input', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(8, 8, 0.5), 0x3b82f6, {type: 'conv', layer: 0});
      z += 6;
      let prevBlockOut = input;
      for (let i = 0; i < 3; i++) {
          const resBlock = createTransformerComponent(`ResBlock ${i+1}`, new THREE.Vector3(0, y, z), new THREE.BoxGeometry(7-i, 7-i, 2+i), 0xa855f7, {type: 'resblock', layer: i+1});
          createCurvedConnection(prevBlockOut.position, resBlock.position, 0xcccccc, 0); // Main path
          createCurvedConnection(prevBlockOut.position, resBlock.position, 0xffa500, 8); // Skip connection
          prevBlockOut = resBlock;
          z += 6;
      }
      createTransformerComponent('Global Pool', new THREE.Vector3(0, y, z), new THREE.SphereGeometry(1.5), 0x22d3ee, {type: 'pool', layer: 4});
      z += 4;
      createTransformerComponent('FC/Output', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(1, 1, 3), 0xef4444, {type: 'output', layer: 5});
  }

  function createGoogLeNet() {
      let z = -25;
      const y = -8;
      const input = createTransformerComponent('Input', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(8, 8, 0.5), 0x3b82f6, {type: 'conv', layer: 0});
      z += 8;
      let prev_z = input.position.z;
      let layer = 1;

      for (let i = 0; i < 2; i++) {
          const inceptionModule = new THREE.Group();
          inceptionModule.position.z = z;
          // Main path for connection
          const mainBody = createTransformerComponent(null, new THREE.Vector3(0, y, 0), new THREE.BoxGeometry(0.1, 0.1, 0.1), 0xcccccc, {layer: layer++});
          // 1x1 Conv
          const branch1 = createTransformerComponent('1x1', new THREE.Vector3(-3.5, y, 0), new THREE.BoxGeometry(1, 6-i*2, 1), 0x22c55e, {});
          // 1x1 -> 3x3 Conv
          const branch2 = createTransformerComponent('3x3', new THREE.Vector3(-1.5, y, 0), new THREE.BoxGeometry(1.5, 6-i*2, 1.5), 0x3b82f6, {});
          // 1x1 -> 5x5 Conv
          const branch3 = createTransformerComponent('5x5', new THREE.Vector3(1, y, 0), new THREE.BoxGeometry(2, 6-i*2, 2), 0xa855f7, {});
          // 3x3 Pool -> 1x1 Conv
          const branch4 = createTransformerComponent('Pool', new THREE.Vector3(3.5, y, 0), new THREE.BoxGeometry(1, 6-i*2, 1), 0xef4444, {});

          inceptionModule.add(mainBody, branch1, branch2, branch3, branch4);
          scene.add(inceptionModule);
          transformerObjects.push(inceptionModule, mainBody, branch1, branch2, branch3, branch4);
          createTextLabel(`Inception ${i+1}`, new THREE.Vector3(0, y + 8, z));
          createCurvedConnection(new THREE.Vector3(0, y, prev_z), new THREE.Vector3(0, y, z), 0xcccccc, 0);
          prev_z = z;
          z += 10;
      }
      createTransformerComponent('Global Avg Pool', new THREE.Vector3(0, y, z), new THREE.SphereGeometry(1.5), 0x22d3ee, {type: 'pool', layer: layer++});
      createCurvedConnection(new THREE.Vector3(0, y, prev_z), new THREE.Vector3(0, y, z), 0xcccccc, 0);
      prev_z = z;
      z += 5;
      createTransformerComponent('Output', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(1, 1, 3), 0xef4444, {type: 'output', layer: layer++});
      createCurvedConnection(new THREE.Vector3(0, y, prev_z), new THREE.Vector3(0, y, z), 0xcccccc, 0);
  }

  function createYOLONetwork() {
      let z = -20;
      const y = -8;
      let layer = 0;
      let prevComp = null;
      let currentComp = null;

      prevComp = createTransformerComponent('Input', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(8, 8, 0.5), 0x3b82f6, {type: 'conv', layer: layer++});
      z += 7;
      currentComp = createTransformerComponent('Backbone (CSPDarknet53)', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(6, 6, 4), 0xa855f7, {type: 'conv', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 7;
      currentComp = createTransformerComponent('Neck (C2f)', new THREE.Vector3(0, y, z), new THREE.BoxGeometry(4, 4, 2), 0x22d3ee, {type: 'conv', layer: layer++});
      createCurvedConnection(prevComp.position, currentComp.position, 0xcccccc, 0); prevComp = currentComp;
      z += 6;
      const { gridSize } = state.architecture.cnn; // Use CNN state for this now
      const cellSize = 2.5;
      const gridGroup = new THREE.Group();
      gridGroup.position.z = z;
      for (let i = 0; i < gridSize; i++) {
          for (let j = 0; j < gridSize; j++) {
              const x = (i - (gridSize-1)/2) * cellSize;
              const y_pos = y + (j - (gridSize-1)/2) * cellSize;
              const cell = createTransformerComponent(null, new THREE.Vector3(x, y_pos, 0), new THREE.PlaneGeometry(cellSize, cellSize), 0x444444, {type: 'grid_cell', gridX: i, gridY: j, layer: layer});
              gridGroup.add(cell);
          }
      }
      scene.add(gridGroup);
      transformerObjects.push(gridGroup);
      createTextLabel("Detection Head (Anchor-Free)", new THREE.Vector3(0, y + 5, z));
  }

  function createCNNNetwork() {
      const variant = state.architecture.cnn.variant;
      switch(variant) {
          case 'LeNet': createLeNet(); break;
          case 'AlexNet': createAlexNet(); break;
          case 'VGGNet': createVGGNet(); break;
          case 'ResNet': createResNet(); break;
          case 'GoogLeNet': createGoogLeNet(); break;
          case 'YOLO': createYOLONetwork(); break;
          // VGG, MobileNet etc. would be added here
          default: createLeNet(); // Fallback
      }
  }

  function createRNNNetwork() {
    const { timeSteps } = state.architecture.rnn;
    const y_base = 0; // Float above grid
    for (let t = 0; t < timeSteps; t++) {
        const x = t * 4 - (timeSteps-1)*2;
        const input = createTransformerComponent(`Input x_${t}`, new THREE.Vector3(x, y_base - 4, 0), new THREE.SphereGeometry(0.5), 0x3b82f6, {type: 'input', step: t});
        const hidden = createTransformerComponent(`Cell h_${t}`, new THREE.Vector3(x, y_base, 0), new THREE.SphereGeometry(0.8), 0x22c55e, {type: 'hidden', step: t});
        const output = createTransformerComponent(`Output y_${t}`, new THREE.Vector3(x, y_base + 4, 0), new THREE.SphereGeometry(0.5), 0xef4444, {type: 'output', step: t});

        // Connections within a time step
        createCurvedConnection(input.position, hidden.position, 0xcccccc, 0);
        createCurvedConnection(hidden.position, output.position, 0xcccccc, 0);

        if (t > 0) {
            const prev_x = (t-1) * 4 - (timeSteps-1)*2;
            createCurvedConnection(new THREE.Vector3(prev_x, y_base, 0), new THREE.Vector3(x, y_base, 0), 0xffa500, 1.5);
        }
    }
  }

  function createLSTMNetwork() {
    const { timeSteps } = state.architecture.lstm;
    const y_base = 0; // Float above grid
    // Cell State "Conveyor Belt"
    const cellStateGeo = new THREE.CylinderGeometry(0.3, 0.3, timeSteps * 4, 16);
    const cellStateMesh = createTransformerComponent(null, new THREE.Vector3(0, y_base, 0), cellStateGeo, 0x2196F3, {type: 'cell_state'});
    cellStateMesh.rotation.z = Math.PI / 2;
    createTextLabel("Cell State (Memory)", new THREE.Vector3(0, y_base - 1, 0), 0.7);

    for (let t = 0; t < timeSteps; t++) {
        const x = t * 4 - (timeSteps-1)*2;
        createTransformerComponent(`h_${t}`, new THREE.Vector3(x, y_base + 2.5, 0), new THREE.SphereGeometry(0.6), 0x4CAF50, {type: 'hidden', step: t});
        // Gates
        createTransformerComponent(null, new THREE.Vector3(x - 0.8, y_base + 1, 0), new THREE.BoxGeometry(0.6, 0.6, 0.2), 0xFF5722, {type: 'gate', gateType: 'forget', step: t});
        createTransformerComponent(null, new THREE.Vector3(x, y_base + 1, 0), new THREE.BoxGeometry(0.6, 0.6, 0.2), 0x9C27B0, {type: 'gate', gateType: 'input', step: t});
        createTransformerComponent(null, new THREE.Vector3(x + 0.8, y_base + 1, 0), new THREE.BoxGeometry(0.6, 0.6, 0.2), 0xFF9800, {type: 'gate', gateType: 'output', step: t});
        if (t === 0) {
            createTextLabel("F", new THREE.Vector3(x - 0.8, y_base + 1.8, 0), 0.5);
            createTextLabel("I", new THREE.Vector3(x, y_base + 1.8, 0), 0.5);
            createTextLabel("O", new THREE.Vector3(x + 0.8, y_base + 1.8, 0), 0.5);
        }
    }
  }

  function createGRUNetwork() {
      const { timeSteps } = state.architecture.gru;
      const y_base = 0; // Float above grid
      for (let t = 0; t < timeSteps; t++) {
          const x = t * 4 - (timeSteps - 1) * 2;
          const hidden = createTransformerComponent(`h_${t}`, new THREE.Vector3(x, y_base, 0), new THREE.SphereGeometry(0.8), 0x607D8B, { type: 'hidden', step: t });
          // Gates
          const resetGate = createTransformerComponent(null, new THREE.Vector3(x - 1.2, y_base, 0), new THREE.BoxGeometry(0.6, 0.6, 0.2), 0xF44336, { type: 'gate', gateType: 'reset', step: t });
          const updateGate = createTransformerComponent(null, new THREE.Vector3(x, y_base + 1.2, 0), new THREE.TorusGeometry(0.4, 0.15, 8, 12), 0x2196F3, { type: 'gate', gateType: 'update', step: t });
          updateGate.rotation.x = Math.PI / 2;
          if (t === 0) {
              createTextLabel("Reset", resetGate.position.clone().add(new THREE.Vector3(0, 0.8, 0)), 0.5);
              createTextLabel("Update", updateGate.position.clone().add(new THREE.Vector3(0, 0.8, 0)), 0.5);
          }
          if (t > 0) {
              const prev_x = (t - 1) * 4 - (timeSteps - 1) * 2;
              createCurvedConnection(new THREE.Vector3(prev_x, y_base, 0), new THREE.Vector3(x, y_base, 0), 0xffa500, 1.5);
          }
      }
  }

  function createMoENetwork() {
      const { experts } = state.architecture.moe;
      const radius = 6;
      const y_base = 0; // Float above grid
      // Central Gate
      const gate = createTransformerComponent('Gating Network', new THREE.Vector3(0, y_base + 2, 0), new THREE.SphereGeometry(1), 0xFFD700, {type: 'gate'});
      // Experts
      for (let i = 0; i < experts; i++) {
          const angle = (i / experts) * Math.PI * 2;
          const pos = new THREE.Vector3(Math.cos(angle) * radius, y_base, Math.sin(angle) * radius);
          const expert = createTransformerComponent(`Expert ${i+1}`, pos, new THREE.CylinderGeometry(0.8, 0.8, 2, 8), 0x3b82f6, {type: 'expert', id: i});
          createCurvedConnection(gate.position, expert.position, 0xcccccc, 0);
      }
  }

  function createGANNetwork() {
      const y_base = 0; // Float above grid
      createTransformerComponent('Generator', new THREE.Vector3(-8, y_base, 0), new THREE.ConeGeometry(2, 6, 8), 0x22c55e, {type: 'generator'});
      createTransformerComponent('Discriminator', new THREE.Vector3(8, y_base, 0), new THREE.CylinderGeometry(2, 2, 4, 8), 0xef4444, {type: 'discriminator'});
      createTextLabel("Latent Noise", new THREE.Vector3(-8, y_base - 2, 0), 0.7);
      createTextLabel("Real/Fake Decision", new THREE.Vector3(8, y_base - 2, 0), 0.7);
  }

  function createNetwork() {
    if (gridHelper) {
        scene.remove(gridHelper);
        gridHelper.dispose();
        gridHelper = null;
    }
    neurons.forEach(layer => layer.forEach(n => { scene.remove(n); n.geometry.dispose(); n.material.dispose(); }));
    connections.forEach(c => { scene.remove(c); c.geometry.dispose(); c.material.dispose(); });
    transformerObjects.forEach(obj => { scene.remove(obj); if(obj.geometry) obj.geometry.dispose(); if(obj.material) obj.material.dispose(); });
    particles.forEach(p => scene.remove(p));
    neurons = [];
    connections = [];
    transformerObjects = [];
    particles = [];

    if (state.archType === 'transformer') createTransformerNetwork();
    else if (state.archType === 'cnn') createCNNNetwork();
    else if (state.archType === 'rnn') createRNNNetwork();
    else if (state.archType === 'lstm') createLSTMNetwork();
    else if (state.archType === 'gru') createGRUNetwork();
    else if (state.archType === 'moe') createMoENetwork();
    else if (state.archType === 'gan') createGANNetwork();
    else {
        const layers = getCurrentLayerConfig();
        createGenericNetwork(layers);
    }

    // Dynamically adjust grid size
    const box = new THREE.Box3().setFromObject(scene);
    const size = box.getSize(new THREE.Vector3());
    const gridSize = Math.max(size.x, size.z, 40) * 1.2;
    gridHelper = new THREE.GridHelper(gridSize, gridSize / 2, 0x444444, 0x222222);
    gridHelper.position.y = -8;
    scene.add(gridHelper);
    applyTheme(); // Re-apply theme to new grid

    const layers = getCurrentLayerConfig(); // Used for camera distance calculation
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
    const currentArch = state.architecture[state.archType];
    const layerCount = Array.isArray(currentArch) ? currentArch.length : (currentArch.encoderBlocks || 0) + (currentArch.decoderBlocks || 0) + (currentArch.experts || 0) + (currentArch.depth || 0);
    const neuronCount = Array.isArray(currentArch) ? currentArch.reduce((s, l) => s + l.neurons, 0) : 'N/A';
    document.getElementById('network-stats').textContent = `${layerCount} components • ${neuronCount} neurons`;

    document.getElementById('start-btn').classList.toggle('hidden', state.isTraining);
    document.getElementById('stop-btn').classList.toggle('hidden', !state.isTraining);
    document.getElementById('settings-btn')?.classList.toggle('active', state.activePanel === 'settings');
    document.getElementById('start-btn').disabled = state.isTrainingComplete;
    document.getElementById('speed-value').textContent = `${state.animationSpeed.toFixed(1)}x`;

    document.getElementById('epoch-value').textContent = state.epoch;
    document.getElementById('batch-value').textContent = state.batch;
    document.getElementById('loss-value').textContent = state.loss.toFixed(4);
    document.getElementById('accuracy-value').textContent = `${(state.accuracy * 100).toFixed(1)}%`;
    document.getElementById('accuracy-label').innerHTML = `Accuracy ${state.accuracy >= 0.9 ? '🎉' : ''}`;

    ['arch', 'param', 'opt', 'settings'].forEach(p => {
      const btn = document.getElementById(`${p}-btn`);
      if (btn) btn.classList.toggle('active', state.activePanel === p);
    });
    archPanel.classList.toggle('visible', state.activePanel === 'architecture');
    paramPanel.classList.toggle('visible', state.activePanel === 'parameters');
    optPanel.classList.toggle('visible', state.activePanel === 'optimization');
    settingsPanel.classList.toggle('visible', state.activePanel === 'settings');
    panelDetailsContainer.style.display = state.activePanel ? 'block' : 'none';
    document.getElementById('camera-mode-toggle').checked = state.cameraMode === 'manual';
  }

function applyTheme() {
    const isLight = state.theme === 'light';

    const gridColor = isLight ? 0xaaaaaa : 0x444444;
    const subGridColor = isLight ? 0xdddddd : 0x222222;
    const sceneBgColor = isLight ? 0xf0f0f0 : 0x0a0a0a;

    document.body.classList.toggle('light-mode', isLight);

    scene.background.set(sceneBgColor);

    if (gridHelper) {
        gridHelper.material.color.set(gridColor);
        // The grid helper in three.js uses two materials. We can't easily access the second one.
        // A full solution would be to create a custom grid. For now, this is a good approximation.
    }
  }

  function renderArchitecturePanel() {
    let html = '';
    switch (state.archType) {
      case 'transformer':
        html = `<h3 style="color: var(--purple);">Transformer Architecture</h3>
          <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div><label>Encoder Blocks:</label><input type="number" min="1" max="6" value="${state.architecture.transformer.encoderBlocks}" class="update-arch-input" data-field="encoderBlocks"></div>
            <div><label>Decoder Blocks:</label><input type="number" min="1" max="6" value="${state.architecture.transformer.decoderBlocks}" class="update-arch-input" data-field="decoderBlocks"></div>
          </div>`;
        break;
      case 'cnn':
        const cnnOpts = ['LeNet', 'AlexNet', 'VGGNet', 'ResNet', 'GoogLeNet', 'YOLO'];
        html = `<h3 style="color: var(--purple);">CNN Architecture</h3>
          <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div>
              <label>Variant:</label>
              <select id="cnn-variant-select" class="update-cnn-variant">
                ${cnnOpts.map(opt => `<option value="${opt}" ${state.architecture.cnn.variant === opt ? 'selected' : ''}>${opt}</option>`).join('')}
              </select>
            </div>
            ${state.architecture.cnn.variant === 'YOLO' ? `
            <div><label>Grid Size:</label><input type="number" min="2" max="7" value="${state.architecture.cnn.gridSize || 3}" class="update-arch-input" data-field="gridSize"></div>
            ` : ''}
          </div>`;
        break;
      case 'rnn':
      case 'lstm':
      case 'gru':
        const archName = state.archType.toUpperCase();
        html = `<h3 style="color: var(--purple);">${archName} Architecture</h3>
          <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div><label>Time Steps:</label><input type="number" min="2" max="10" value="${state.architecture[state.archType].timeSteps}" class="update-arch-input" data-field="timeSteps"></div>
          </div>`;
        break;
      case 'moe':
        html = `<h3 style="color: var(--purple);">Mixture of Experts Architecture</h3>
          <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div><label>Number of Experts:</label><input type="number" min="2" max="8" value="${state.architecture.moe.experts}" class="update-arch-input" data-field="experts"></div>
          </div>`;
        break;
      case 'gan':
        html = `<h3 style="color: var(--purple);">GAN Architecture</h3>
          <p class="text-xs" style="color: var(--text-med);">Structure is fixed for visualization.</p>`;
        break;
      case 'yolo': // This case is now removed from dropdown, but we keep it clean.
        html = ``;
        break;
      case 'unet':
        html = `<h3 style="color: var(--purple);">U-Net Architecture</h3>
          <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div><label>Down/Up-sampling Depth:</label><input type="number" min="1" max="4" value="${state.architecture.unet.depth}" class="update-arch-input" data-field="depth"></div>
          </div>`;
        break;
      case 'fnn':
      default:
        html = `<h3 style="color: var(--purple);">Network Architecture</h3><div class="space-y-2" style="display: flex; flex-direction: column; gap: 0.5rem;">`;
        state.architecture.fnn.forEach((layer, index) => {
          html += `<div class="layer-item">
            <div class="flex items-center" style="justify-content: space-between; margin-bottom: 0.5rem;">
              <span class="text-xs font-bold">${layer.name}</span>
              ${index > 0 && index < state.architecture.fnn.length - 1 ? `<button class="remove-layer-btn" data-index="${index}" style="color: var(--red); background: none; border: none; font-size: 1.2rem; cursor: pointer;">&times;</button>` : ''}
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
              <div><label>Neurons:</label><input type="number" min="1" max="64" value="${layer.neurons}" class="update-layer-input" data-index="${index}" data-field="neurons"></div>
              <div><label>Activation:</label><select class="update-layer-input" data-index="${index}" data-field="activation" ${index === 0 || index === state.architecture.fnn.length - 1 ? 'disabled' : ''}>
                ${['ReLU', 'Sigmoid', 'Tanh', 'Linear', 'Softmax'].map(opt => `<option value="${opt}" ${layer.activation === opt ? 'selected' : ''}>${opt}</option>`).join('')}
              </select></div>
            </div>
          </div>`;
        });
        html += `</div><button id="add-layer-btn" class="btn" style="width: 100%; margin-top: 0.75rem; background-color: var(--purple); color: #fff;">Add Hidden Layer</button>`;
    }
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

  function renderSettingsPanel() {
      const themeToggle = document.getElementById('theme-toggle');
      if (themeToggle) {
        themeToggle.checked = state.theme === 'light';
        themeToggle.onchange = (e) => {
            state.theme = e.target.checked ? 'light' : 'dark';
            localStorage.setItem('theme', state.theme);
            applyTheme();
            updateUI();
        };
      }
    }

    function renderArchSpecificSettings() {
        const archSpecificSettingsContainer = document.getElementById('arch-specific-settings-container');
        if (!archSpecificSettingsContainer) return;
        archSpecificSettingsContainer.innerHTML = '';
        archSpecificSettingsContainer.style.display = 'none';
    }

  let lossHistory = [], accHistory = [];
function drawLossAccGraph() {
      const ctx = lossAccCanvas.getContext('2d');
      const w = lossAccCanvas.width, h = lossAccCanvas.height;
      ctx.clearRect(0, 0, w, h);

      const computedStyle = getComputedStyle(document.documentElement);
      const redColor = computedStyle.getPropertyValue('--red').trim();
      const greenColor = computedStyle.getPropertyValue('--green').trim();

      // Draw Loss (Red)
      ctx.strokeStyle = redColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      lossHistory.forEach((v, i) => {
          const x = (i / Math.max(1, lossHistory.length - 1)) * w;
          const y = h - v * h;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Draw Accuracy (Green)
      ctx.strokeStyle = greenColor;
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
    if (state.activePanel === 'settings') renderSettingsPanel();
    updateUI();
  }
  document.getElementById('arch-btn').addEventListener('click', () => handlePanelToggle('architecture'));
  document.getElementById('param-btn').addEventListener('click', () => handlePanelToggle('parameters'));
  document.getElementById('opt-btn').addEventListener('click', () => handlePanelToggle('optimization'));
	document.getElementById('settings-btn')?.addEventListener('click', () => handlePanelToggle('settings'));
  document.getElementById('speed-slider').addEventListener('input', e => { state.animationSpeed = parseFloat(e.target.value); updateUI(); });

  document.getElementById('camera-mode-toggle').addEventListener('change', e => {
    state.cameraMode = e.target.checked ? 'manual' : 'auto';
  });


  const autoBtn = document.getElementById('cam-auto-btn');
  const manualBtn = document.getElementById('cam-manual-btn');

  panelDetailsContainer.addEventListener('click', e => {
    if (state.archType !== 'fnn') return; // Layer add/remove only for FNN
    if (e.target.id === 'add-layer-btn') {
      state.architecture.fnn.splice(-1, 0, { neurons: 8, activation: 'ReLU', name: `Hidden ${state.architecture.fnn.length - 1}` });
      createNetwork();
      renderArchitecturePanel();
    }
    if (e.target.classList.contains('remove-layer-btn')) {
      state.architecture.fnn.splice(parseInt(e.target.dataset.index), 1);
      createNetwork();
      renderArchitecturePanel();
    }
  });
  panelDetailsContainer.addEventListener('change', e => {
    // FNN specific layer updates
    if (state.archType === 'fnn' && e.target.classList.contains('update-layer-input')) {
      const { index, field } = e.target.dataset;
      const value = e.target.type === 'number' ? parseInt(e.target.value) : e.target.value;
      state.architecture.fnn[index][field] = value;
      createNetwork();
    }
    // Generic architecture config updates
    if (e.target.classList.contains('update-arch-input')) {
        const { field } = e.target.dataset;
        const value = parseInt(e.target.value);
        if (state.architecture[state.archType]) {
            // Special case for CNN properties which are now nested
            if (state.archType === 'cnn') {
                state.architecture.cnn[field] = value;
            } else if (['rnn', 'lstm', 'gru'].includes(state.archType)) {
                state.architecture.rnn[field] = value;
                state.architecture.lstm[field] = value;
                state.architecture.gru[field] = value;
            } else {
                state.architecture[state.archType][field] = value;
            }
            createNetwork();
        }
    }
    // CNN Variant change
    if (e.target.id === 'cnn-variant-select') {
        state.architecture.cnn.variant = e.target.value;
        if (!state.architecture.cnn.gridSize) state.architecture.cnn.gridSize = 3; // Ensure gridSize exists for YOLO
        createNetwork();
        renderArchitecturePanel(); // Re-render to show/hide variant-specific options
    }
    // Training param updates
    if (e.target.classList.contains('update-param-input')) {
      const { field } = e.target.dataset;
      const value = e.target.type === 'number' ? parseFloat(e.target.value) : e.target.value;
      state.trainingParams[field] = value;
    }
  });

  // --- ARCHITECTURE DROPDOWN & LOADING SPINNER LOGIC ---
  archTypeSelect.addEventListener('change', e => {
    const newType = e.target.value;
    if (state.archType === newType) return;
    archLoadingSpinner.classList.remove('hidden');
    setTimeout(() => {
      state.archType = newType;
      createNetwork();
      if (state.activePanel === 'architecture') renderArchitecturePanel();
      archLoadingSpinner.classList.add('hidden');
    }, 700); // Simulate loading delay
  });

  function startTraining() {
    Object.assign(state, { isTraining: true, isTrainingComplete: false, epoch: 0, batch: 0, loss: 1.0, accuracy: 0.1, animationStep: 0 });
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
      if (state.batch % 10 === 0) {
        state.epoch++;
        if (state.archType === 'transformer') {
            state.animationStep = (state.animationStep + 1) % (state.architecture.transformer.encoderBlocks + state.architecture.transformer.decoderBlocks + 2);
        }
        if (['rnn', 'lstm', 'gru'].includes(state.archType)) {
            state.animationStep = (state.animationStep + 1) % state.architecture[state.archType].timeSteps;
        }
        if (state.archType === 'cnn') {
            state.animationStep = (state.animationStep + 1) % 12; // Generic 12 steps for longer CNNs
        }
        if (state.archType === 'gan') {
            state.animationStep = (state.animationStep + 1) % 2; // 0 for D, 1 for G
        }
      }
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
    Object.assign(state, { isTrainingComplete: false, epoch: 0, batch: 0, loss: 1.0, accuracy: 0.1, animationStep: 0 });
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
    const tick = clock.getDelta();

    if (state.cameraMode === 'auto') {
      camState.distance = THREE.MathUtils.lerp(camState.distance, camState.targetDistance, 0.02);
      camera.position.set(Math.cos(deltaTime * 0.15) * camState.distance, camState.distance * 0.4, Math.sin(deltaTime * 0.15) * camState.distance);
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

    // Update particles
    for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.userData.life -= tick;
        if (p.userData.life <= 0) {
            scene.remove(p);
            particles.splice(i, 1);
        } else {
            p.position.add(p.userData.velocity.clone().multiplyScalar(tick));
            p.material.opacity = p.userData.life;
        }
    }

    if (state.isTraining && !state.isTrainingComplete) {
      const currentArchConfig = state.architecture[state.archType];
      const animLength = Array.isArray(currentArchConfig) ? currentArchConfig.length : 4; // Simplified anim length for non-FNN
      animationProgress = (deltaTime * state.animationSpeed * 0.5) % (animLength * 2 + 2);
      const progress = animationProgress;

      // Reset all visuals
      const allObjects = [...neurons.flat(), ...transformerObjects];
      allObjects.forEach(obj => {
          if (obj.material && obj.material.emissive) obj.material.emissive.setHex(0);
          if (obj.scale && obj.userData.fromLayer !== undefined) obj.scale.set(1, 1, 1);
      });
      connections.forEach(c => {
        c.material.color.setHex(0x666666);
        c.material.opacity = 0.15;
      });

      if (state.archType === 'fnn') {
          // Refined wave animation from React component
          if (progress < state.architecture.fnn.length) { // FORWARD PASS (Green)
            for (let i = 0; i < state.architecture.fnn.length; i++) {
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
          } else if (progress < state.architecture.fnn.length * 2) { // BACKWARD PASS (Red)
            const backProgress = progress - state.architecture.fnn.length;
            for (let i = state.architecture.fnn.length - 1; i >= 0; i--) {
              const reverseIdx = state.architecture.fnn.length - 1 - i;
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
            const updateIntensity = Math.sin((progress - state.architecture.fnn.length * 2) * Math.PI * 4);
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
      } else if (state.archType === 'transformer') {
          const step = state.animationStep;
          const encBlocks = state.architecture.transformer.encoderBlocks;
          const decBlocks = state.architecture.transformer.decoderBlocks;

          transformerObjects.forEach(obj => {
              if (obj.material && obj.material.emissive) obj.material.emissive.setHex(0);
          });

          let activeComponent;
          if (step === 0) activeComponent = 'embedding';
          else if (step <= encBlocks) activeComponent = { stack: 'encoder', layer: step - 1 };
          else if (step <= encBlocks + decBlocks) activeComponent = { stack: 'decoder', layer: step - 1 - encBlocks };
          else activeComponent = 'output';

          transformerObjects.forEach(obj => {
              const ud = obj.userData;
              let is_active = false;
              if (typeof activeComponent === 'string') {
                  if (activeComponent === 'embedding' && ud.type === 'embedding') is_active = true;
                  if (activeComponent === 'output' && (ud.type === 'linear' || ud.type === 'softmax')) is_active = true;
              } else if (ud.stack === activeComponent.stack && ud.layer === activeComponent.layer) {
                  is_active = true;
              }

              if (is_active) {
                  if (obj.material && obj.material.emissive) {
                      obj.material.emissive.setHex(0x00ff00);
                  }
                  // Create flowing particles from active components
                  if (Math.random() < 0.2) {
                      const particle = new THREE.Mesh(new THREE.SphereGeometry(0.1, 4, 4), new THREE.MeshBasicMaterial({ color: 0x00ff88, transparent: true }));
                      particle.position.copy(obj.position);
                      const next_y = obj.position.y + (ud.stack === 'encoder' ? 5 : 7);
                      particle.userData = {
                          velocity: new THREE.Vector3(0, 2, 0),
                          life: 2.5
                      };
                      particles.push(particle);
                      scene.add(particle);
                  }
              }
          });
      } else if (state.archType === 'rnn') {
          const step = state.animationStep;
          transformerObjects.forEach(obj => {
              if (obj.material.emissive) obj.material.emissive.setHex(0);
              if (obj.userData.step === step) {
                  obj.material.emissive.setHex(0x00ff00);
              }
          });
      } else if (state.archType === 'lstm') {
          const step = state.animationStep;
          const sinTime = Math.sin(deltaTime * state.animationSpeed * 2);
          transformerObjects.forEach(obj => {
              if (obj.material.emissive) obj.material.emissive.setHex(0);
              if (obj.userData.step === step) {
                  obj.material.emissive.setHex(0x00ff00);
                  // Gate animation
                  if (obj.userData.type === 'gate') {
                      const gateActivation = (sinTime + 1) / 2; // 0 to 1
                      obj.scale.y = gateActivation;
                      obj.material.opacity = 0.5 + gateActivation * 0.5;
                  }
              }
          });
      } else if (state.archType === 'gru') {
          const step = state.animationStep;
          const sinTime = Math.sin(deltaTime * state.animationSpeed * 2);
          transformerObjects.forEach(obj => {
              if (obj.material.emissive) obj.material.emissive.setHex(0);
              if (obj.userData.step === step) {
                  obj.material.emissive.setHex(0x00ff00);
                  if (obj.userData.gateType === 'reset') {
                      obj.scale.x = (sinTime + 1) / 2;
                  }
                  if (obj.userData.gateType === 'update') {
                      obj.rotation.y += 0.05 * sinTime;
                  }
              }
          });
      } else if (state.archType === 'cnn') {
          const step = state.animationStep;
          transformerObjects.forEach(obj => {
              if (obj.material && obj.material.emissive) obj.material.emissive.setHex(0);
          });
          const activeObjs = transformerObjects.filter(o => o.userData.layer === step);
          if (activeObjs.length > 0) {
              activeObjs.forEach(o => {
                  if (o.material && o.material.emissive) o.material.emissive.setHex(0x00ff00);
              });
          }
      } else if (state.archType === 'moe') {
          const gate = transformerObjects.find(o => o.userData.type === 'gate');
          const experts = transformerObjects.filter(o => o.userData.type === 'expert');
          gate.material.emissive.setHex(0xffff00);
          const activeExpert = Math.floor(deltaTime * state.animationSpeed * 2) % experts.length;
          experts.forEach((exp, i) => {
              exp.material.emissive.setHex(i === activeExpert ? 0x00ff00 : 0);
          });
      } else if (state.archType === 'gan') {
          const generator = transformerObjects.find(o => o.userData.type === 'generator');
          const discriminator = transformerObjects.find(o => o.userData.type === 'discriminator');
          // Step 0: Discriminator trains
          if (state.animationStep === 0) {
              discriminator.material.emissive.setHex(0x00ff00);
              generator.material.emissive.setHex(0);
              if (Math.random() < 0.1) { // Fake data flow
                  const fakeData = new THREE.Mesh(new THREE.SphereGeometry(0.2), new THREE.MeshBasicMaterial({color: 0xff8888}));
                  fakeData.position.copy(generator.position);
                  scene.add(fakeData);
                  setTimeout(() => scene.remove(fakeData), 1000);
              }
          } else { // Step 1: Generator trains
              generator.material.emissive.setHex(0x00ff00);
              discriminator.material.emissive.setHex(0);
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

  archTypeSelect.value = state.archType;
  createNetwork();
  updateUI();
  applyTheme();
  animate();
});
