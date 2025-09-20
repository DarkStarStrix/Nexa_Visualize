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
      transformer: {
        variant: 'encoder-decoder', // NEW: encoder-only, decoder-only, encoder-decoder
        encoderBlocks: 2,
        decoderBlocks: 2
      },
      cnn: { variant: 'LeNet', convLayers: 2, fcLayers: 2 },
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

      // Handle different geometry types for height calculation
      let height = 0;
      if (geometry.parameters) {
          height = geometry.parameters.height || geometry.parameters.radius || 1;
      } else {
          height = 1; // Default fallback
      }

      mesh.position.set(position.x, position.y + height / 2, position.z);

      mesh.userData = { name, ...userData, base_y: position.y };
      scene.add(mesh);
      transformerObjects.push(mesh);

      if (name) {
          // Position label above the component's actual mesh position
          const labelPos = mesh.position.clone();
          labelPos.y += height / 2 + 0.4;
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

  // --- OUTLINE & DOTTED LINE HELPERS ---
  function addOutline(mesh, color = 0xffff00, thickness = 0.08) {
      // Add a slightly larger mesh with a basic material for outline
      const outlineMat = new THREE.MeshBasicMaterial({
          color,
          side: THREE.BackSide,
          transparent: true,
          opacity: 0.7,
          depthWrite: false
      });
      const outline = new THREE.Mesh(mesh.geometry.clone(), outlineMat);
      outline.position.copy(mesh.position);
      outline.scale.copy(mesh.scale).multiplyScalar(1 + thickness);
      outline.renderOrder = 1;
      scene.add(outline);
      transformerObjects.push(outline);
      return outline;
  }

  function addDottedLine(start, end, color = 0xffffff, dashSize = 0.4, gapSize = 0.25) {
      const points = [start, end];
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      // Use LineDashedMaterial for dotted/dashed lines
      const material = new THREE.LineDashedMaterial({
          color,
          dashSize,
          gapSize,
          linewidth: 2,
          transparent: true,
          opacity: 0.7
      });
      const line = new THREE.Line(geometry, material);
      line.computeLineDistances();
      scene.add(line);
      transformerObjects.push(line);
      return line;
  }

  function addDottedBox(center, size, color, dashSize = 0.2, gapSize = 0.2) {
      const halfSize = { x: size.x / 2, y: size.y / 2, z: size.z / 2 };
      const corners = [
          new THREE.Vector3(center.x - halfSize.x, center.y - halfSize.y, center.z - halfSize.z),
          new THREE.Vector3(center.x + halfSize.x, center.y - halfSize.y, center.z - halfSize.z),
          new THREE.Vector3(center.x + halfSize.x, center.y + halfSize.y, center.z - halfSize.z),
          new THREE.Vector3(center.x - halfSize.x, center.y + halfSize.y, center.z - halfSize.z),
          new THREE.Vector3(center.x - halfSize.x, center.y - halfSize.y, center.z + halfSize.z),
          new THREE.Vector3(center.x + halfSize.x, center.y - halfSize.y, center.z + halfSize.z),
          new THREE.Vector3(center.x + halfSize.x, center.y + halfSize.y, center.z + halfSize.z),
          new THREE.Vector3(center.x - halfSize.x, center.y + halfSize.y, center.z + halfSize.z)
      ];

      // Edges
      addDottedLine(corners[0], corners[1], color, dashSize, gapSize);
      addDottedLine(corners[1], corners[2], color, dashSize, gapSize);
      addDottedLine(corners[2], corners[3], color, dashSize, gapSize);
      addDottedLine(corners[3], corners[0], color, dashSize, gapSize);
      addDottedLine(corners[4], corners[5], color, dashSize, gapSize);
      addDottedLine(corners[5], corners[6], color, dashSize, gapSize);
      addDottedLine(corners[6], corners[7], color, dashSize, gapSize);
      addDottedLine(corners[7], corners[4], color, dashSize, gapSize);
      addDottedLine(corners[0], corners[4], color, dashSize, gapSize);
      addDottedLine(corners[1], corners[5], color, dashSize, gapSize);
      addDottedLine(corners[2], corners[6], color, dashSize, gapSize);
      addDottedLine(corners[3], corners[7], color, dashSize, gapSize);
  }

  function createPositionalEncodingVisualization(centerX, baseY, stackType, stackHeight) {
      // Single horizontal branch extending from the trunk
      const branchLength = 2.5; // Length of the branch
      const branchY = baseY - (stackHeight / 2) - 0.5; // Position at bottom of stack
      const branchEndX = centerX + (stackType === 'encoder' ? -branchLength : branchLength); // Left for encoder, right for decoder

      // Create the horizontal branch line
      const lineGeometry = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(centerX, branchY, 0),
          new THREE.Vector3(branchEndX, branchY, 0)
      ]);
      const lineMaterial = new THREE.LineBasicMaterial({
          color: 0xffd700,
          linewidth: 3,
          transparent: true,
          opacity: 0.8
      });
      const line = new THREE.Line(lineGeometry, lineMaterial);
      scene.add(line);
      transformerObjects.push(line);

      // Plus symbol at connection point (where branch meets trunk)
      createTextLabel("+", new THREE.Vector3(centerX + (stackType === 'encoder' ? -0.3 : 0.3), branchY, 0), 0.5, "#ffd700");

      // Marble at the end of the branch
      const marbleGeometry = new THREE.SphereGeometry(0.15, 8, 6);
      const marbleMaterial = new THREE.MeshPhongMaterial({
          color: 0xffd700,
          shininess: 100,
          transparent: true,
          opacity: 0.9
      });
      const marble = new THREE.Mesh(marbleGeometry, marbleMaterial);
      marble.position.set(branchEndX, branchY, 0);
      scene.add(marble);
      transformerObjects.push(marble);
      addOutline(marble, 0xffd700, 0.05);

      // Label for positional encoding
      createTextLabel("Positional Encoding", new THREE.Vector3(branchEndX, branchY - 0.5, 0), 0.5, "#ffd700");
  }

  // --- REFINED TRANSFORMER VISUALIZATION ---

  function createEncoderOnlyTransformer() {
      // Clear previous logic removed from here. createNetwork() handles it.

      const { encoderBlocks } = state.architecture.transformer;
      const enc_x = 0; // Centered
      const y_base = 0;

      // Input Embedding
      const inputEmbed = createTransformerComponent('Input Embedding', new THREE.Vector3(enc_x, y_base - 7, 0), new THREE.BoxGeometry(3, 1, 1), 0xffb6c1, { type: 'embedding', stack: 'encoder' });
      addOutline(inputEmbed, 0x22d3ee, 0.09);

      // --- Unified Encoder Stack Outline ---
      const encBlockHeight = 2.5;
      const encBlockSpacing = 3.5;
      const encStackHeight = (encoderBlocks - 1) * encBlockSpacing + encBlockHeight;
      const encStackCenterY = y_base + ((encoderBlocks - 1) * encBlockSpacing) / 2;

      addDottedBox(new THREE.Vector3(enc_x, encStackCenterY + 0.5, 0), {x: 3.2, y: encStackHeight + 1.5, z: 1.6}, 0x2196f3);
      createTextLabel("Encoder Stack", new THREE.Vector3(enc_x, encStackCenterY + (encStackHeight + 1.5) / 2 + 1.0, 0), 0.8, "#22d3ee");

      // Positional Encoding for Encoder
      createPositionalEncodingVisualization(enc_x, encStackCenterY, 'encoder', encStackHeight);

      // Encoder Stack
      let prevEncBlock = inputEmbed;
      for (let i = 0; i < encoderBlocks; i++) {
          const y = y_base + i * encBlockSpacing;

          const encBlockPos = new THREE.Vector3(enc_x, y + encBlockHeight / 2, 0);

          // Solid connection from previous block
          createCurvedConnection(prevEncBlock.position, encBlockPos, 0xffffff, 0);
          prevEncBlock = { position: encBlockPos };

          // Sub-components inside encoder block
          const attn = createTransformerComponent('Multi-Head Self-Attention', new THREE.Vector3(enc_x, y + 0.3, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0xffa500, { type: 'attention', stack: 'encoder', layer: i });
          const ff = createTransformerComponent('Feed Forward', new THREE.Vector3(enc_x, y + 1.5, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0x87ceeb, { type: 'ff', stack: 'encoder', layer: i });

          attn.material.depthWrite = false;
          ff.material.depthWrite = false;
          addOutline(attn, 0xffa500, 0.05);
          addOutline(ff, 0x87ceeb, 0.05);

          // Internal flow line
          createCurvedConnection(attn.position, ff.position, 0xcccccc, 1.0);
      }

      // Output Layers for Classification/Embedding
      const last_encoder_y = y_base + (encoderBlocks - 1) * encBlockSpacing + encBlockHeight;
      const pooler = createTransformerComponent('Pooler', new THREE.Vector3(enc_x, last_encoder_y + 2.0, 0), new THREE.BoxGeometry(2, 0.8, 0.8), 0x808080, { type: 'pooler' });
      const classifier = createTransformerComponent('Classification Head', new THREE.Vector3(enc_x, last_encoder_y + 4.0, 0), new THREE.SphereGeometry(0.6, 16, 12), 0x32cd32, { type: 'classifier' });

      createCurvedConnection(prevEncBlock.position, pooler.position, 0xffffff, 0);
      createCurvedConnection(pooler.position, classifier.position, 0xffffff, 0);

      addOutline(pooler, 0x808080, 0.06);
      addOutline(classifier, 0x32cd32, 0.06);

      createTextLabel("Classification Output", new THREE.Vector3(enc_x, last_encoder_y + 5.5, 0), 0.7, "#22c55e");
  }

  function createDecoderOnlyTransformer() {
      // Clear previous logic removed from here. createNetwork() handles it.

      const { decoderBlocks } = state.architecture.transformer;
      const dec_x = 0; // Centered
      const y_base = 0;

      // Input/Output Embedding
      const outputEmbed = createTransformerComponent('Token Embedding', new THREE.Vector3(dec_x, y_base - 7, 0), new THREE.BoxGeometry(3, 1, 1), 0xffb6c1, { type: 'embedding', stack: 'decoder' });
      addOutline(outputEmbed, 0x22d3ee, 0.09);

      // --- Unified Decoder Stack Outline ---
      const decBlockHeight = 3.0;
      const decBlockSpacing = 4.0;
      const decStackHeight = (decoderBlocks - 1) * decBlockSpacing + decBlockHeight;
      const decStackCenterY = y_base + ((decoderBlocks - 1) * decBlockSpacing) / 2;

      addDottedBox(new THREE.Vector3(dec_x, decStackCenterY + 0.5, 0), {x: 3.2, y: decStackHeight + 1.5, z: 1.6}, 0xa855f7);
      createTextLabel("Decoder Stack", new THREE.Vector3(dec_x, decStackCenterY + (decStackHeight + 1.5) / 2 + 1.0, 0), 0.8, "#a855f7");

      // Positional Encoding for Decoder
      createPositionalEncodingVisualization(dec_x, decStackCenterY, 'decoder', decStackHeight);

      // Decoder Stack (Decoder-only has only masked self-attention)
      let decoderTops = []; // Bug fix: decoderTops was not defined
      let prevDecBlock = outputEmbed;
      for (let i = 0; i < decoderBlocks; i++) {
          const y = y_base + i * decBlockSpacing;

          const decBlockPos = new THREE.Vector3(dec_x, y + decBlockHeight / 2, 0);
          const decBlockTop = decBlockPos.clone().add(new THREE.Vector3(0, decBlockHeight / 2, 0));
          decoderTops.push(decBlockTop);

          // Solid connection from previous block
          createCurvedConnection(prevDecBlock.position, decBlockPos, 0xffffff, 0);
          prevDecBlock = { position: decBlockPos };

          // Sub-components inside decoder block (No cross-attention in decoder-only)
          const maskedAttn = createTransformerComponent('Masked Multi-Head Self-Attention', new THREE.Vector3(dec_x, y + 0.3, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0x9932cc, { type: 'masked_attention', stack: 'decoder', layer: i });
          const ff = createTransformerComponent('Feed Forward', new THREE.Vector3(dec_x, y + 1.8, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0x87ceeb, { type: 'ff', stack: 'decoder', layer: i });

          maskedAttn.material.depthWrite = false;
          ff.material.depthWrite = false;
          addOutline(maskedAttn, 0x9932cc, 0.05);
          addOutline(ff, 0x87ceeb, 0.05);

          // Internal flow lines
          createCurvedConnection(maskedAttn.position, ff.position, 0xcccccc, 1.0);
      }

      // Output Layers for Language Modeling
      const last_decoder_y = y_base + (decoderBlocks - 1) * decBlockSpacing + decBlockHeight;
      const lmHead = createTransformerComponent('Language Model Head', new THREE.Vector3(dec_x, last_decoder_y + 2.0, 0), new THREE.BoxGeometry(2, 0.8, 0.8), 0x808080, { type: 'lm_head' });
      const softmax = createTransformerComponent('Softmax', new THREE.Vector3(dec_x, last_decoder_y + 4.0, 0), new THREE.SphereGeometry(0.6, 16, 12), 0x32cd32, { type: 'softmax' });

      createCurvedConnection(prevDecBlock.position, lmHead.position, 0xffffff, 0);
      createCurvedConnection(lmHead.position, softmax.position, 0xffffff, 0);

      addOutline(lmHead, 0x808080, 0.06);
      addOutline(softmax, 0x32cd32, 0.06);

      createTextLabel("Next Token Prediction", new THREE.Vector3(dec_x, last_decoder_y + 5.5, 0), 0.7, "#22c55e");
  }

  function createTransformerNetwork() {
      const variant = state.architecture.transformer.variant;

      switch(variant) {
          case 'encoder-only':
              createEncoderOnlyTransformer();
              break;
          case 'decoder-only':
              createDecoderOnlyTransformer();
              break;
          case 'encoder-decoder':
          default:
              createEncoderDecoderTransformer();
              break;
      }
  }

  function createEncoderDecoderTransformer() {
      // Clear previous logic removed from here. createNetwork() handles it.

      const { encoderBlocks, decoderBlocks } = state.architecture.transformer;
      const enc_x = -4;
      const dec_x = 4;
      const y_base = 0;

      // Embeddings
      const inputEmbed = createTransformerComponent('Input Embedding', new THREE.Vector3(enc_x, y_base - 7, 0), new THREE.BoxGeometry(3, 1, 1), 0xffb6c1, { type: 'embedding', stack: 'encoder' });
      const outputEmbed = createTransformerComponent('Output Embedding', new THREE.Vector3(dec_x, y_base - 7, 0), new THREE.BoxGeometry(3, 1, 1), 0xffb6c1, { type: 'embedding', stack: 'decoder' });

      addOutline(inputEmbed, 0x22d3ee, 0.09);
      addOutline(outputEmbed, 0x22d3ee, 0.09);

      // Draw dotted vertical lines to separate encoder/decoder
      addDottedLine(
          new THREE.Vector3(0, y_base - 12, -4),
          new THREE.Vector3(0, y_base + 20, 4),
          0xffffff, 0.3, 0.3
      );

      // --- Unified Encoder Stack Outline ---
      const encBlockHeight = 2.5;
      const encBlockSpacing = 3.5;
      const encStackHeight = (encoderBlocks - 1) * encBlockSpacing + encBlockHeight;
      const encStackCenterY = y_base + ((encoderBlocks - 1) * encBlockSpacing) / 2;

      addDottedBox(new THREE.Vector3(enc_x, encStackCenterY + 0.5, 0), {x: 3.2, y: encStackHeight + 1.5, z: 1.6}, 0x2196f3);
      createTextLabel("Encoder", new THREE.Vector3(enc_x, encStackCenterY + (encStackHeight + 1.5) / 2 + 1.0, 0), 0.8, "#22d3ee");

      // Positional Encoding for Encoder
      createPositionalEncodingVisualization(enc_x, encStackCenterY, 'encoder', encStackHeight);

      // Encoder Stack
      let encoderTops = [];
      let prevEncBlock = inputEmbed;
      for (let i = 0; i < encoderBlocks; i++) {
          const y = y_base + i * encBlockSpacing;

          const encBlockPos = new THREE.Vector3(enc_x, y + encBlockHeight / 2, 0);
          const encBlockTop = encBlockPos.clone().add(new THREE.Vector3(0, encBlockHeight / 2, 0));
          encoderTops.push(encBlockTop);

          // Solid connection from previous block
          createCurvedConnection(prevEncBlock.position, encBlockPos, 0xffffff, 0);
          prevEncBlock = { position: encBlockPos };

          const attn = createTransformerComponent('Multi-Head Attention', new THREE.Vector3(enc_x, y + 0.3, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0xffa500, { type: 'attention', stack: 'encoder', layer: i });
          const ff = createTransformerComponent('Feed Forward', new THREE.Vector3(enc_x, y + 1.5, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0x87ceeb, { type: 'ff', stack: 'encoder', layer: i });

          attn.material.depthWrite = false;
          ff.material.depthWrite = false;
          addOutline(attn, 0xffa500, 0.05);
          addOutline(ff, 0x87ceeb, 0.05);

          // Internal flow line
          createCurvedConnection(attn.position, ff.position, 0xcccccc, 1.0);
      }

      // --- Unified Decoder Stack Outline ---
      const decBlockHeight = 4.0;
      const decBlockSpacing = 5.0;
      const decStackHeight = (decoderBlocks - 1) * decBlockSpacing + decBlockHeight;
      const decStackCenterY = y_base + ((decoderBlocks - 1) * decBlockSpacing) / 2;

      addDottedBox(new THREE.Vector3(dec_x, decStackCenterY + 0.5, 0), {x: 3.2, y: decStackHeight + 1.5, z: 1.6}, 0xa855f7);
      createTextLabel("Decoder", new THREE.Vector3(dec_x, decStackCenterY + (decStackHeight + 1.5) / 2 + 1.0, 0), 0.8, "#a855f7");

      // Positional Encoding for Decoder
      createPositionalEncodingVisualization(dec_x, decStackCenterY, 'decoder', decStackHeight);

      // Decoder Stack
      let decoderTops = [];
      let prevDecBlock = outputEmbed;
      for (let i = 0; i < decoderBlocks; i++) {
          const y = y_base + i * decBlockSpacing;

          const decBlockPos = new THREE.Vector3(dec_x, y + decBlockHeight / 2, 0);
          const decBlockTop = decBlockPos.clone().add(new THREE.Vector3(0, decBlockHeight / 2, 0));
          decoderTops.push(decBlockTop);

          // Solid connection from previous block
          createCurvedConnection(prevDecBlock.position, decBlockPos, 0xffffff, 0);
          prevDecBlock = { position: decBlockPos };

          const maskedAttn = createTransformerComponent('Masked MHA', new THREE.Vector3(dec_x, y + 0.3, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0x9932cc, { type: 'masked_attention', stack: 'decoder', layer: i });
          const crossAttn = createTransformerComponent('Cross-Attention', new THREE.Vector3(dec_x, y + 1.3, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0xdc143c, { type: 'cross_attention', stack: 'decoder', layer: i });
          const ff = createTransformerComponent('Feed Forward', new THREE.Vector3(dec_x, y + 2.3, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0x87ceeb, { type: 'ff', stack: 'decoder', layer: i });

          maskedAttn.material.depthWrite = false;
          crossAttn.material.depthWrite = false;
          ff.material.depthWrite = false;
          addOutline(maskedAttn, 0x9932cc, 0.05);
          addOutline(crossAttn, 0xdc143c, 0.05);
          addOutline(ff, 0x87ceeb, 0.05);

          createCurvedConnection(maskedAttn.position, crossAttn.position, 0xcccccc, 1.0);
          createCurvedConnection(crossAttn.position, ff.position, 0xcccccc, 1.0);
      }

      // Cross-Attention Connections
      for (let i = 0; i < Math.min(encoderBlocks, decoderBlocks); i++) {
          const encTop = encoderTops[i];
          const decCrossAttn = transformerObjects.find(o => o.userData.type === 'cross_attention' && o.userData.layer === i);
          if (decCrossAttn) {
              createCurvedConnection(
                  new THREE.Vector3(encTop.x + 1.5, encTop.y, encTop.z),
                  new THREE.Vector3(decCrossAttn.position.x - 1.5, decCrossAttn.position.y, decCrossAttn.position.z),
                  0xff0000, -2
              );
          }
      }

      // Output Layers
      const last_decoder_y = y_base + (decoderBlocks - 1) * decBlockSpacing + decBlockHeight;
      const linear = createTransformerComponent('Linear', new THREE.Vector3(dec_x, last_decoder_y + 2.0, 0), new THREE.BoxGeometry(2, 0.8, 0.8), 0x808080, { type: 'linear' });
      const softmax = createTransformerComponent('Softmax', new THREE.Vector3(dec_x, last_decoder_y + 4.0, 0), new THREE.SphereGeometry(0.6, 16, 12), 0x32cd32, { type: 'softmax' });

      createCurvedConnection(prevDecBlock.position, linear.position, 0xffffff, 0);
      createCurvedConnection(linear.position, softmax.position, 0xffffff, 0);

      addOutline(linear, 0x808080, 0.06);
      addOutline(softmax, 0x32cd32, 0.06);

      createTextLabel("Output", new THREE.Vector3(dec_x, last_decoder_y + 5.5, 0), 0.7, "#22c55e");
  }

  // --- NETWORK CREATION DISPATCH ---
  function createNetwork() {
    // Clear existing network
    neurons.forEach(layer => layer.forEach(n => { scene.remove(n); n.geometry.dispose(); n.material.dispose(); }));
    connections.forEach(c => { scene.remove(c); c.geometry.dispose(); c.material.dispose(); });
    transformerObjects.forEach(obj => { scene.remove(obj); if(obj.geometry) obj.geometry.dispose(); if(obj.material) obj.material.dispose(); });
    particles.forEach(p => scene.remove(p));

    neurons = [];
    connections = [];
    transformerObjects = [];
    particles = [];

    // Get current architecture config
    const currentArch = state.architecture[state.archType];

    // Create standard flexible grid for all models
    const gridSize = createStandardGrid(state.archType, currentArch);

    // Dispatch to appropriate architecture builder
    switch (state.archType) {
      case 'transformer':
        createTransformerNetwork();
        break;
      case 'cnn':
        createCNNNetwork();
        break;
      case 'rnn':
        createRNNNetwork();
        break;
      case 'lstm':
        createLSTMNetwork();
        break;
      case 'gru':
        createGRUNetwork();
        break;
      case 'moe':
        createMoENetwork();
        break;
      case 'gan':
        createGANNetwork();
        break;
      case 'fnn':
      default:
        const layers = getCurrentLayerConfig();
        createGenericNetwork(layers);
        break;
    }

    updateUI();
  }

  function calculateGridSize(archType, currentArch) {
    let minSize = 20; // Default minimum
    let maxDimension = 0;

    switch(archType) {
      case 'transformer':
        // Calculate based on encoder/decoder positions and blocks
        const encBlocks = currentArch.encoderBlocks || 0;
        const decBlocks = currentArch.decoderBlocks || 0;
        const maxBlocks = Math.max(encBlocks, decBlocks);
        const width = currentArch.variant === 'encoder-decoder' ? 16 : 8; // Two sides or one
        const height = maxBlocks * 5 + 15; // Block spacing + embeddings + output
        maxDimension = Math.max(width, height);
        break;

      case 'cnn':
        // CNN layers spread horizontally
        const layerCount = getCNNLayerConfig(currentArch.variant).length;
        const width_cnn = (layerCount + 2) * 3 + 10; // Layer spacing + input + padding
        maxDimension = Math.max(width_cnn, 20);
        break;

      case 'rnn':
      case 'lstm':
      case 'gru':
        // RNN unrolled across time steps
        const timeSteps = currentArch.timeSteps || 5;
        const width_rnn = timeSteps * 5 + 10; // Time step spacing + padding
        maxDimension = Math.max(width_rnn, 20);
        break;

      case 'moe':
        // Circular expert arrangement
        const numExperts = currentArch.experts || 4;
        const radius = Math.max(5, numExperts * 0.8);
        maxDimension = Math.max(radius * 3, 20); // Diameter + padding
        break;

      case 'gan':
        // Two networks side by side
        maxDimension = Math.max(25, 20);
        break;

      case 'fnn':
      default:
        // FNN layers in sequence
        const layers = getCurrentLayerConfig();
        if (layers && layers.length > 0) {
          const layerSpacing = layers.length * 5 + 10;
          const maxNeuronGrid = Math.max(...layers.map(l => {
            if (l && l.gridSize && Array.isArray(l.gridSize)) {
              return Math.max(l.gridSize[0] || 1, l.gridSize[1] || 1) * 3;
            }
            return 3; // Default fallback
          }));
          maxDimension = Math.max(layerSpacing, maxNeuronGrid, 20);
        } else {
          maxDimension = 20; // Fallback if no layers
        }
        break;
    }

    // Ensure grid is always larger than content with padding
    const gridSize = Math.ceil(maxDimension * 1.4);
    return Math.max(gridSize, minSize);
  }

  function createStandardGrid(archType, currentArch) {
    // Remove existing grid
    if (gridHelper) {
      scene.remove(gridHelper);
      gridHelper.dispose();
      gridHelper = null;
    }

    // Calculate appropriate grid size for the current architecture
    const gridSize = calculateGridSize(archType, currentArch);

    // Apply theme-appropriate colors
    const isLight = state.theme === 'light';
    const gridColor = isLight ? 0xaaaaaa : 0x444444;
    const subGridColor = isLight ? 0xdddddd : 0x222222;

    // Create new grid helper
    gridHelper = new THREE.GridHelper(gridSize, Math.max(20, gridSize / 2), gridColor, subGridColor);
    gridHelper.position.y = -8;
    scene.add(gridHelper);

    return gridSize;
  }

  // --- CNN ARCHITECTURE BUILDERS ---
  function createCNNNetwork() {
    const variant = state.architecture.cnn.variant;
    const y_base = -5; // Lower the model to be closer to the grid

    // Input image
    const inputImage = createTransformerComponent('Input Image', new THREE.Vector3(-15, y_base, 0), new THREE.BoxGeometry(4, 4, 0.3), 0x3b82f6, { type: 'input', layer: 0 });
    addOutline(inputImage, 0x3b82f6, 0.05);

    // CNN layers based on variant
    let layerConfigs = getCNNLayerConfig(variant);
    let lastPosition = inputImage.position.clone();

    layerConfigs.forEach((config, index) => {
      const x = -10 + index * 4; // Increase spacing
      // Create a funnel effect by lowering the y position for deeper layers
      const y = y_base - index * 0.5;
      const component = createTransformerComponent(config.name, new THREE.Vector3(x, y, 0), config.geometry, config.color, { type: config.type, layer: index });
      addOutline(component, config.color, 0.05);

      createCurvedConnection(lastPosition, component.position, 0xffffff, 0);
      lastPosition = component.position;
    });

    // Set camera distance based on grid size
    const gridSize = calculateGridSize('cnn', state.architecture.cnn);
    camState.targetDistance = Math.max(40, gridSize * 0.8);
    camState.distance = camState.targetDistance;
  }

  function getCNNLayerConfig(variant) {
    const configs = {
      'LeNet': [
        { name: 'Conv1 (6@28x28)', geometry: new THREE.BoxGeometry(3, 3, 1), color: 0xff6b6b, type: 'conv' },
        { name: 'Pool1 (6@14x14)', geometry: new THREE.BoxGeometry(2, 2, 1), color: 0x4ecdc4, type: 'pool' },
        { name: 'Conv2 (16@10x10)', geometry: new THREE.BoxGeometry(2.5, 2.5, 1.5), color: 0xff6b6b, type: 'conv' },
        { name: 'Pool2 (16@5x5)', geometry: new THREE.BoxGeometry(1.5, 1.5, 1.5), color: 0x4ecdc4, type: 'pool' },
        { name: 'FC1 (120)', geometry: new THREE.SphereGeometry(0.8), color: 0xfeca57, type: 'fc' },
        { name: 'FC2 (84)', geometry: new THREE.SphereGeometry(0.6), color: 0xfeca57, type: 'fc' },
        { name: 'Output (10)', geometry: new THREE.SphereGeometry(0.5), color: 0x48dbfb, type: 'output' }
      ],
      'AlexNet': [
        { name: 'Conv1 (96@55x55)', geometry: new THREE.BoxGeometry(4, 4, 1), color: 0xff6b6b, type: 'conv' },
        { name: 'Pool1 (96@27x27)', geometry: new THREE.BoxGeometry(3, 3, 1), color: 0x4ecdc4, type: 'pool' },
        { name: 'Conv2 (256@27x27)', geometry: new THREE.BoxGeometry(3.5, 3.5, 1.5), color: 0xff6b6b, type: 'conv' },
        { name: 'Pool2 (256@13x13)', geometry: new THREE.BoxGeometry(2.5, 2.5, 1.5), color: 0x4ecdc4, type: 'pool' },
        { name: 'Conv3 (384@13x13)', geometry: new THREE.BoxGeometry(3, 3, 2), color: 0xff6b6b, type: 'conv' },
        { name: 'Conv4 (384@13x13)', geometry: new THREE.BoxGeometry(3, 3, 2), color: 0xff6b6b, type: 'conv' },
        { name: 'Conv5 (256@13x13)', geometry: new THREE.BoxGeometry(2.5, 2.5, 2), color: 0xff6b6b, type: 'conv' },
        { name: 'Pool3 (256@6x6)', geometry: new THREE.BoxGeometry(2, 2, 2), color: 0x4ecdc4, type: 'pool' },
        { name: 'FC1 (4096)', geometry: new THREE.SphereGeometry(1), color: 0xfeca57, type: 'fc' },
        { name: 'FC2 (4096)', geometry: new THREE.SphereGeometry(0.8), color: 0xfeca57, type: 'fc' },
        { name: 'Output (1000)', geometry: new THREE.SphereGeometry(0.6), color: 0x48dbfb, type: 'output' }
      ],
      'VGGNet': [
        { name: 'Conv1 (64@224x224)', geometry: new THREE.BoxGeometry(3.5, 3.5, 0.5), color: 0xff6b6b, type: 'conv' },
        { name: 'Conv2 (64@224x224)', geometry: new THREE.BoxGeometry(3.5, 3.5, 0.5), color: 0xff6b6b, type: 'conv' },
        { name: 'Pool1 (64@112x112)', geometry: new THREE.BoxGeometry(2.5, 2.5, 0.5), color: 0x4ecdc4, type: 'pool' },
        { name: 'Conv3 (128@112x112)', geometry: new THREE.BoxGeometry(3, 3, 1), color: 0xff6b6b, type: 'conv' },
        { name: 'Conv4 (128@112x112)', geometry: new THREE.BoxGeometry(3, 3, 1), color: 0xff6b6b, type: 'conv' },
        { name: 'Pool2 (128@56x56)', geometry: new THREE.BoxGeometry(2, 2, 1), color: 0x4ecdc4, type: 'pool' },
        { name: 'FC1 (4096)', geometry: new THREE.SphereGeometry(1), color: 0xfeca57, type: 'fc' },
        { name: 'FC2 (4096)', geometry: new THREE.SphereGeometry(0.8), color: 0xfeca57, type: 'fc' },
        { name: 'Output (1000)', geometry: new THREE.SphereGeometry(0.6), color: 0x48dbfb, type: 'output' }
      ],
      'ResNet': [
        { name: 'Conv1 (64@112x112)', geometry: new THREE.BoxGeometry(3.5, 3.5, 0.8), color: 0xff6b6b, type: 'conv' },
        { name: 'Pool1 (64@56x56)', geometry: new THREE.BoxGeometry(2.5, 2.5, 0.8), color: 0x4ecdc4, type: 'pool' },
        { name: 'ResBlock1 (64@56x56)', geometry: new THREE.CylinderGeometry(1.5, 1.5, 2), color: 0x9c88ff, type: 'resblock' },
        { name: 'ResBlock2 (128@28x28)', geometry: new THREE.CylinderGeometry(1.3, 1.3, 2), color: 0x9c88ff, type: 'resblock' },
        { name: 'ResBlock3 (256@14x14)', geometry: new THREE.CylinderGeometry(1.1, 1.1, 2), color: 0x9c88ff, type: 'resblock' },
        { name: 'ResBlock4 (512@7x7)', geometry: new THREE.CylinderGeometry(0.9, 0.9, 2), color: 0x9c88ff, type: 'resblock' },
        { name: 'GAP (512)', geometry: new THREE.OctahedronGeometry(0.8), color: 0x4ecdc4, type: 'gap' },
        { name: 'FC (1000)', geometry: new THREE.SphereGeometry(0.6), color: 0x48dbfb, type: 'output' }
      ],
      'GoogLeNet': [
        { name: 'Conv1 (64@112x112)', geometry: new THREE.BoxGeometry(3.5, 3.5, 0.8), color: 0xff6b6b, type: 'conv' },
        { name: 'Pool1 (64@56x56)', geometry: new THREE.BoxGeometry(2.5, 2.5, 0.8), color: 0x4ecdc4, type: 'pool' },
        { name: 'Inception1', geometry: new THREE.DodecahedronGeometry(1.2), color: 0xffd93d, type: 'inception' },
        { name: 'Inception2', geometry: new THREE.DodecahedronGeometry(1.1), color: 0xffd93d, type: 'inception' },
        { name: 'Inception3', geometry: new THREE.DodecahedronGeometry(1.0), color: 0xffd93d, type: 'inception' },
        { name: 'GAP (1024)', geometry: new THREE.OctahedronGeometry(0.8), color: 0x4ecdc4, type: 'gap' },
        { name: 'FC (1000)', geometry: new THREE.SphereGeometry(0.6), color: 0x48dbfb, type: 'output' }
      ],
      'YOLO': [
        { name: 'Backbone CNN', geometry: new THREE.BoxGeometry(4, 4, 2), color: 0xff6b6b, type: 'backbone' },
        { name: 'Feature Pyramid', geometry: new THREE.ConeGeometry(1.5, 3), color: 0x9c88ff, type: 'fpn' },
        { name: 'Detection Head', geometry: new THREE.CylinderGeometry(1.2, 1.2, 1.5), color: 0xffd93d, type: 'detection' },
        { name: 'NMS', geometry: new THREE.OctahedronGeometry(0.8), color: 0x4ecdc4, type: 'nms' },
        { name: 'Bounding Boxes', geometry: new THREE.BoxGeometry(2, 1.5, 0.5), color: 0x48dbfb, type: 'bbox' }
      ]
    };
    return configs[variant] || configs['LeNet'];
  }

  function applyTheme() {
    const isLight = state.theme === 'light';

    const gridColor = isLight ? 0xaaaaaa : 0x444444;
    const subGridColor = isLight ? 0xdddddd : 0x222222;
    const sceneBgColor = isLight ? 0xf0f0f0 : 0x0a0a0a;

    document.body.classList.toggle('light-mode', isLight);

    scene.background.set(sceneBgColor);

    // Update grid helper colors if it exists
    if (gridHelper) {
        const currentArch = state.architecture[state.archType];
        createStandardGrid(state.archType, currentArch);
    }
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
    document.getElementById('network-stats').textContent = `${layerCount} components ��� ${neuronCount} neurons`;

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

  function getTransformerVariantDescription(variant) {
    switch(variant) {
      case 'encoder-only':
        return '<strong>Encoder-only:</strong> Used for tasks like text classification and sentiment analysis (e.g., BERT). It processes the entire input sequence at once using bidirectional attention.';
      case 'decoder-only':
        return '<strong>Decoder-only:</strong> Used for generative tasks like text completion (e.g., GPT). It generates text sequentially, one token at a time, using masked self-attention.';
      case 'encoder-decoder':
        return '<strong>Encoder-Decoder:</strong> The original Transformer model, used for sequence-to-sequence tasks like machine translation. The encoder processes the input, and the decoder generates the output while attending to the encoder\'s output.';
      default:
        return '';
    }
  }

  // --- PANEL RENDERING FUNCTIONS ---
  function renderArchitecturePanel() {
    let html = '';
    switch (state.archType) {
      case 'transformer':
        const transformerOpts = ['encoder-decoder', 'encoder-only', 'decoder-only'];
        html = `<h3 style="color: var(--purple);">Transformer Architecture</h3>
          <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div>
              <label>Variant:</label>
              <select id="transformer-variant-select" class="update-transformer-variant">
                ${transformerOpts.map(opt => `<option value="${opt}" ${state.architecture.transformer.variant === opt ? 'selected' : ''}>${opt.charAt(0).toUpperCase() + opt.slice(1).replace('-', ' ')}</option>`).join('')}
              </select>
            </div>
            ${state.architecture.transformer.variant === 'encoder-decoder' || state.architecture.transformer.variant === 'encoder-only' ? `
            <div><label>Encoder Blocks:</label><input type="number" min="1" max="6" value="${state.architecture.transformer.encoderBlocks}" class="update-arch-input" data-field="encoderBlocks"></div>
            ` : ''}
            ${state.architecture.transformer.variant === 'encoder-decoder' || state.architecture.transformer.variant === 'decoder-only' ? `
            <div><label>Decoder Blocks:</label><input type="number" min="1" max="6" value="${state.architecture.transformer.decoderBlocks}" class="update-arch-input" data-field="decoderBlocks"></div>
            ` : ''}
            <div style="margin-top: 0.5rem; padding: 0.5rem; background: var(--layer-item-bg); border-radius: 4px; font-size: 0.75rem;">
              ${getTransformerVariantDescription(state.architecture.transformer.variant)}
            </div>
          </div>`;
        break;
      case 'cnn':
        const cnnVariants = ['LeNet', 'AlexNet', 'VGGNet', 'ResNet', 'GoogLeNet', 'YOLO'];
        html = `<h3 style="color: var(--red);">CNN Architecture</h3>
          <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div>
              <label>Variant:</label>
              <select id="cnn-variant-select" class="update-cnn-variant">
                ${cnnVariants.map(variant => `<option value="${variant}" ${state.architecture.cnn.variant === variant ? 'selected' : ''}>${variant}</option>`).join('')}
              </select>
            </div>
          </div>`;
        break;
      case 'rnn':
      case 'lstm':
      case 'gru':
        html = `<h3 style="color: var(--purple);">RNN Architecture</h3>
          <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div><label>Time Steps:</label><input type="number" min="3" max="10" value="${state.architecture[state.archType].timeSteps}" class="update-arch-input" data-field="timeSteps"></div>
          </div>`;
        break;
      case 'moe':
        html = `<h3 style="color: var(--green);">Mixture of Experts</h3>
          <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div><label>Experts:</label><input type="number" min="2" max="8" value="${state.architecture.moe.experts}" class="update-arch-input" data-field="experts"></div>
          </div>`;
        break;
      case 'gan':
        html = `<h3 style="color: var(--cyan);">GAN Architecture</h3>
          <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div><label>Generator Layers:</label><input type="number" min="2" max="5" value="${state.architecture.gan.generatorLayers}" class="update-arch-input" data-field="generatorLayers"></div>
            <div><label>Discriminator Layers:</label><input type="number" min="2" max="5" value="${state.architecture.gan.discriminatorLayers}" class="update-arch-input" data-field="discriminatorLayers"></div>
          </div>`;
        break;
      case 'fnn':
      default:
        html = `<h3 style="color: var(--blue);">Feedforward Network</h3>
          <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            ${state.architecture.fnn.map((layer, i) => `
              <div class="layer-item">
                <label>Layer ${i + 1}:</label>
                <div style="display: flex; gap: 0.5rem; align-items: center;">
                  <input type="number" min="1" max="128" value="${layer.neurons}" data-index="${i}" data-field="neurons" class="update-layer-input" style="width: 4rem;">
                  <select data-index="${i}" data-field="activation" class="update-layer-input">
                    <option value="Linear" ${layer.activation === 'Linear' ? 'selected' : ''}>Linear</option>
                    <option value="ReLU" ${layer.activation === 'ReLU' ? 'selected' : ''}>ReLU</option>
                    <option value="Sigmoid" ${layer.activation === 'Sigmoid' ? 'selected' : ''}>Sigmoid</option>
                    <option value="Tanh" ${layer.activation === 'Tanh' ? 'selected' : ''}>Tanh</option>
                    <option value="Softmax" ${layer.activation === 'Softmax' ? 'selected' : ''}>Softmax</option>
                  </select>
                  <button class="remove-layer-btn" data-index="${i}" style="background: var(--red); color: white; border: none; padding: 0.25rem 0.5rem; border-radius: 4px; cursor: pointer;">Remove</button>
                </div>
              </div>
            `).join('')}
            <button id="add-layer-btn" style="background: var(--green); color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; margin-top: 0.5rem;">Add Layer</button>
          </div>`;
        break;
    }
    archPanel.innerHTML = html;
  }

  function renderParametersPanel() {
    let html = `<h3 style="color: var(--blue);">Training Parameters</h3>
      <div style="display: flex; flex-direction: column; gap: 0.75rem;">
        <div><label>Learning Rate:</label><input type="number" min="0.0001" max="0.1" step="0.0001" value="${state.trainingParams.learningRate}" data-field="learningRate" class="update-param-input"></div>
        <div><label>Batch Size:</label><input type="number" min="8" max="128" value="${state.trainingParams.batchSize}" data-field="batchSize" class="update-param-input"></div>
        <div><label>Epochs:</label><input type="number" min="10" max="500" value="${state.trainingParams.epochs}" data-field="epochs" class="update-param-input"></div>
        <div>
          <label>Optimizer:</label>
          <select data-field="optimizer" class="update-param-input">
            <option value="Adam" ${state.trainingParams.optimizer === 'Adam' ? 'selected' : ''}>Adam</option>
            <option value="SGD" ${state.trainingParams.optimizer === 'SGD' ? 'selected' : ''}>SGD</option>
            <option value="RMSprop" ${state.trainingParams.optimizer === 'RMSprop' ? 'selected' : ''}>RMSprop</option>
          </select>
        </div>
        <div>
          <label>Loss Function:</label>
          <select data-field="lossFunction" class="update-param-input">
            <option value="CrossEntropy" ${state.trainingParams.lossFunction === 'CrossEntropy' ? 'selected' : ''}>CrossEntropy</option>
            <option value="MSE" ${state.trainingParams.lossFunction === 'MSE' ? 'selected' : ''}>MSE</option>
            <option value="MAE" ${state.trainingParams.lossFunction === 'MAE' ? 'selected' : ''}>MAE</option>
          </select>
        </div>
      </div>`;
    paramPanel.innerHTML = html;
  }

  function renderOptimizationPanel() {
    if (state.activePanel !== 'optimization') return;
    document.getElementById('opt-info-optimizer').textContent = state.trainingParams.optimizer;
    document.getElementById('opt-info-loss-func').textContent = state.trainingParams.lossFunction;
  }

  function renderSettingsPanel() {
    let html = `
      <div style="margin-bottom: 0.5rem;">
        <label>Theme:</label>
        <div class="flex gap-2 items-center">
          <span>Dark</span>
          <label class="switch">
            <input type="checkbox" id="theme-toggle" ${state.theme === 'light' ? '' : 'checked'}>
            <span class="slider round"></span>
          </label>
          <span>Light</span>
        </div>
      </div>
    `;
    settingsPanel.innerHTML = html;
  }

  // --- DOM EVENT LISTENERS ---
  // ...existing code...

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
            if (state.archType === 'transformer') {
                state.architecture.transformer[field] = value;
            } else if (state.archType === 'cnn') {
                state.architecture.cnn[field] = value;
            } else if (['rnn', 'lstm', 'gru'].includes(state.archType)) {
                state.architecture.rnn[field] = value;
                state.architecture.lstm[field] = value;
                state.architecture.gru[field] = value;
            } else {
                state.architecture[state.archType][field] = value;
            }
            createNetwork();
            renderArchitecturePanel(); // Re-render panel after change
        }
    }
    // CNN Variant change
    if (e.target.id === 'cnn-variant-select') {
        state.architecture.cnn.variant = e.target.value;
        if (!state.architecture.cnn.gridSize) state.architecture.cnn.gridSize = 3; // Ensure gridSize exists for YOLO
        createNetwork();
        renderArchitecturePanel(); // Re-render to show/hide variant-specific options
    }
    // Transformer Variant change
    if (e.target.id === 'transformer-variant-select') {
        state.architecture.transformer.variant = e.target.value;
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
