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
      operator: {
        spectralBlocks: 3,
        width: 64,
        projectionDim: 64,
        mlpDim: 128,
        inputDim: 3,
        outputDim: 3
      },
      moe: {
        variant: 'ST-MoE',
        expertLayers: [
          { experts: 4, name: 'Layer 1' }
        ],
        top_k: 2,
        expertUsage: [[0,0,0,0]],
        gateValues: [[0,0,0,0]]
      },
      autoencoder: { encodingDim: 3 },
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
    viewMode: localStorage.getItem('viewMode') || '3d',
    selectedOperatorKey: null,
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
  const operatorLegend = document.getElementById('operator-legend');
  const operatorDetails = document.getElementById('operator-details');
  const operatorDetailsType = document.getElementById('operator-details-type');
  const operatorDetailsTitle = document.getElementById('operator-details-title');
  const operatorDetailsMeta = document.getElementById('operator-details-meta');
  const operatorDetailsDescription = document.getElementById('operator-details-description');
  const view3dBtn = document.getElementById('view-3d-btn');
  const view2dBtn = document.getElementById('view-2d-btn');

  // --- 3D SETUP ---
  const renderer = new THREE.WebGLRenderer({ canvas: canvas3d, antialias: true, powerPreference: "high-performance" });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0a);

  const perspectiveCamera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 1200);
  const orthographicCamera = new THREE.OrthographicCamera(-20, 20, 20, -20, 0.1, 1200);
  let camera = perspectiveCamera;

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
  let lossHistory = [];
  let accHistory = [];

  let neurons = [];
  let connections = [];
  let animationProgress = 0;
  let trainingInterval = null;
  let transformerObjects = []; // For custom transformer meshes
  let particles = []; // For data flow animation
  let interactiveObjects = [];
  let operatorModuleMeta = [];
  let hoveredOperatorObject = null;
  let selectedOperatorObject = null;
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();

  // --- CAMERA CONTROLS ---
  let isDragging = false;
  let dragDistance = 0;
  let lastDragWasPan = false;
  let prevMouse = { x: 0, y: 0 };
  let camState = {
    distance: 42,
    angleX: -0.7,
    angleY: 0.32,
    targetDistance: 42,
    targetAngleX: -0.7,
    targetAngleY: 0.32,
    panX: 0,
    panY: 0,
    targetPanX: 0,
    targetPanY: 0
  };

  canvas3d.addEventListener('mousedown', e => {
    if (state.cameraMode !== 'manual') return;
    isDragging = true;
    dragDistance = 0;
    lastDragWasPan = false;
    prevMouse = { x: e.clientX, y: e.clientY };
    if (state.viewMode === '2d') e.preventDefault();
  });
  canvas3d.addEventListener('mouseup', () => {
    isDragging = false;
    lastDragWasPan = dragDistance > 6;
  });
  canvas3d.addEventListener('mouseleave', () => {
    isDragging = false;
    lastDragWasPan = false;
    hoveredOperatorObject = null;
    updateOperatorDetailsPanel();
  });
  canvas3d.addEventListener('mousemove', e => {
    if (isDragging && state.cameraMode === 'manual') {
      const dx = e.clientX - prevMouse.x;
      const dy = e.clientY - prevMouse.y;
      dragDistance += Math.abs(dx) + Math.abs(dy);
      if (state.viewMode === '2d') {
        const panScale = camState.distance * 0.0024;
        camState.targetPanX -= dx * panScale;
        camState.targetPanY += dy * panScale;
      } else {
        camState.targetAngleX += dx * 0.008;
        camState.targetAngleY = Math.max(-Math.PI / 2.2, Math.min(Math.PI / 2.2, camState.targetAngleY + dy * 0.008));
      }
      prevMouse = { x: e.clientX, y: e.clientY };
    }
    handlePointerMove(e);
  });
  canvas3d.addEventListener('wheel', e => {
    if (state.cameraMode !== 'manual') return;
    e.preventDefault();
    const zoomScale = state.viewMode === '2d' ? 0.03 : 0.05;
    const minZoom = state.viewMode === '2d' ? 18 : 12;
    const maxZoom = state.viewMode === '2d' ? 120 : 110;
    camState.targetDistance = Math.max(minZoom, Math.min(maxZoom, camState.targetDistance + e.deltaY * zoomScale));
  }, { passive: false });
  canvas3d.addEventListener('click', handleOperatorClick);

  function updateOrthographicFrustum() {
    const aspect = window.innerWidth / window.innerHeight;
    const frustumSize = camState.distance * 0.32;
    orthographicCamera.left = -frustumSize * aspect;
    orthographicCamera.right = frustumSize * aspect;
    orthographicCamera.top = frustumSize;
    orthographicCamera.bottom = -frustumSize;
    orthographicCamera.updateProjectionMatrix();
  }

  function setActiveCamera() {
    camera = state.viewMode === '2d' ? orthographicCamera : perspectiveCamera;
    perspectiveCamera.aspect = window.innerWidth / window.innerHeight;
    perspectiveCamera.updateProjectionMatrix();
    updateOrthographicFrustum();
  }

  function updateGridVisibility() {
    if (gridHelper) {
      gridHelper.visible = state.viewMode === '3d';
    }
  }

  function applyViewModeClass() {
    document.body.classList.toggle('view-mode-2d', state.viewMode === '2d');
    document.body.classList.toggle('view-mode-3d', state.viewMode !== '2d');
  }

  function switchViewMode(mode) {
    if (state.viewMode === mode) return;
    state.viewMode = mode;
    localStorage.setItem('viewMode', mode);
    if (mode === '2d') {
      camState.targetPanX = 0;
      camState.targetPanY = 0;
    }
    setActiveCamera();
    applyViewModeClass();
    updateGridVisibility();
    createNetwork();
    updateUI();
  }

  // --- CORE 3D LOGIC ---

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

  function getAutoencoderConfig() {
      const encodingDim = state.architecture.autoencoder.encodingDim;
      const baseLayers = [
          { name: 'Input', neurons: 12, activation: 'Linear' },
          { name: 'Encoder 1', neurons: 8, activation: 'ReLU' },
          { name: 'Encoding', neurons: encodingDim, activation: 'Linear' },
          { name: 'Decoder 1', neurons: 8, activation: 'ReLU' },
          { name: 'Output', neurons: 12, activation: 'Sigmoid' }
      ];
      const colors = [0x3b82f6, 0x22d3ee, 0xa855f7, 0x22d3ee, 0x3b82f6];
      return baseLayers.map((layer, idx) => ({
          ...layer,
          color: colors[idx % colors.length],
          gridSize: [2, Math.ceil(layer.neurons / 2), 1]
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
      case 'operator':
        return []; // These have their own builders
      case 'moe': return getMoEConfig();
      case 'autoencoder': return getAutoencoderConfig();
      case 'unet': return getUNetConfig();
      case 'fnn':
      default: return getFNNLayerConfig();
    }
  }

  function calculateOptimalDistance(layers) {
    const adjustForViewMode = value => state.viewMode === '2d' ? value * 1.2 : value;
    if (state.archType === 'transformer') {
        return adjustForViewMode(50);
    }
    if (state.archType === 'cnn') {
        return adjustForViewMode(50);
    }
    if (state.archType === 'operator') {
        return adjustForViewMode(55);
    }
    if (state.archType === 'moe') {
        return adjustForViewMode(25);
    }
    const networkWidth = layers.length * 5.0; // Adjusted for new spacing
    const networkHeight = Math.max(...layers.map(layer => layer.gridSize[1])) * 2;
    const baseDistance = Math.max(15, Math.max(networkWidth, networkHeight) * 1.5);
    return adjustForViewMode(baseDistance);
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

  function createEncoderDecoderTransformer() {
      const { encoderBlocks, decoderBlocks } = state.architecture.transformer;
      const enc_x = -2; // Moved closer
      const dec_x = 5;  // Moved closer
      const y_base = 0;

      // --- ENCODER ---
      const inputEmbed = createTransformerComponent('Input Embedding', new THREE.Vector3(enc_x, y_base - 7, 0), new THREE.BoxGeometry(3, 1, 1), 0xffb6c1, { type: 'embedding', stack: 'encoder' });
      addOutline(inputEmbed, 0x22d3ee, 0.09);

      const encBlockHeight = 2.5;
      const encBlockSpacing = 3.5;
      const encStackHeight = (encoderBlocks - 1) * encBlockSpacing + encBlockHeight;
      const encStackCenterY = y_base + ((encoderBlocks - 1) * encBlockSpacing) / 2;
      addDottedBox(new THREE.Vector3(enc_x, encStackCenterY + 0.5, 0), {x: 3.2, y: encStackHeight + 1.5, z: 1.6}, 0x2196f3);
      createTextLabel("Encoder Stack", new THREE.Vector3(enc_x, encStackCenterY + (encStackHeight + 1.5) / 2 + 1.0, 0), 0.8, "#22d3ee");
      createPositionalEncodingVisualization(enc_x, encStackCenterY, 'encoder', encStackHeight);

      let encoderTops = [];
      let prevEncBlock = inputEmbed;
      for (let i = 0; i < encoderBlocks; i++) {
          const y = y_base + i * encBlockSpacing;
          const encBlockPos = new THREE.Vector3(enc_x, y + encBlockHeight / 2, 0);
          const encBlockTop = encBlockPos.clone().add(new THREE.Vector3(0, encBlockHeight / 2, 0));
          encoderTops.push(encBlockTop);
          createCurvedConnection(prevEncBlock.position, encBlockPos, 0xffffff, 0);
          prevEncBlock = { position: encBlockPos };

          const attn = createTransformerComponent('Multi-Head Self-Attention', new THREE.Vector3(enc_x, y + 0.3, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0xffa500, { type: 'attention', stack: 'encoder', layer: i });
          const ff = createTransformerComponent('Feed Forward', new THREE.Vector3(enc_x, y + 1.5, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0x87ceeb, { type: 'ff', stack: 'encoder', layer: i });
          addOutline(attn, 0xffa500, 0.05);
          addOutline(ff, 0x87ceeb, 0.05);
          createCurvedConnection(attn.position, ff.position, 0xcccccc, 1.0);
      }

      // --- DECODER ---
      const outputEmbed = createTransformerComponent('Output Embedding', new THREE.Vector3(dec_x, y_base - 7, 0), new THREE.BoxGeometry(3, 1, 1), 0xffb6c1, { type: 'embedding', stack: 'decoder' });
      addOutline(outputEmbed, 0xa855f7, 0.09);

      const decBlockHeight = 3.0;
      const decBlockSpacing = 4.0;
      const decStackHeight = (decoderBlocks - 1) * decBlockSpacing + decBlockHeight;
      const decStackCenterY = y_base + ((decoderBlocks - 1) * decBlockSpacing) / 2;
      addDottedBox(new THREE.Vector3(dec_x, decStackCenterY + 0.5, 0), {x: 3.2, y: decStackHeight + 1.5, z: 1.6}, 0xa855f7);
      createTextLabel("Decoder Stack", new THREE.Vector3(dec_x, decStackCenterY + (decStackHeight + 1.5) / 2 + 1.0, 0), 0.8, "#a855f7");
      createPositionalEncodingVisualization(dec_x, decStackCenterY, 'decoder', decStackHeight);

      let prevDecBlock = outputEmbed;
      for (let i = 0; i < decoderBlocks; i++) {
          const y = y_base + i * decBlockSpacing;
          const decBlockPos = new THREE.Vector3(dec_x, y + decBlockHeight / 2, 0);
          createCurvedConnection(prevDecBlock.position, decBlockPos, 0xffffff, 0);

          // Repositioned components to fit within the block outline
          const maskedAttn = createTransformerComponent('Masked MHA', new THREE.Vector3(dec_x, y + 0.3, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0x9932cc, { type: 'masked_attention', stack: 'decoder', layer: i });
          const crossAttn = createTransformerComponent('Cross-Attention', new THREE.Vector3(dec_x, y + 1.3, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0xdc143c, { type: 'cross_attention', stack: 'decoder', layer: i });
          const ff = createTransformerComponent('Feed Forward', new THREE.Vector3(dec_x, y + 2.3, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0x87ceeb, { type: 'ff', stack: 'decoder', layer: i });

          // The next connection should start from the top of the current block
          prevDecBlock = { position: new THREE.Vector3(dec_x, y + decBlockHeight, 0) };

          addOutline(maskedAttn, 0x9932cc, 0.05);
          addOutline(crossAttn, 0xdc143c, 0.05);
          addOutline(ff, 0x87ceeb, 0.05);

          createCurvedConnection(maskedAttn.position, crossAttn.position, 0xcccccc, 1.0);
          createCurvedConnection(crossAttn.position, ff.position, 0xcccccc, 1.0);

          // Cross-attention connections
          const encoderTop = encoderTops[Math.min(i, encoderTops.length - 1)];
          addDottedLine(encoderTop, crossAttn.position, 0xffffff, 0.3, 0.2);
      }

      // --- OUTPUT ---
      const last_decoder_y = y_base + (decoderBlocks - 1) * decBlockSpacing + decBlockHeight;
      const linear = createTransformerComponent('Linear', new THREE.Vector3(dec_x, last_decoder_y + 2.0, 0), new THREE.BoxGeometry(2, 0.8, 0.8), 0x808080, { type: 'linear' });
      const softmax = createTransformerComponent('Softmax', new THREE.Vector3(dec_x, last_decoder_y + 4.0, 0), new THREE.SphereGeometry(0.6, 16, 12), 0x32cd32, { type: 'softmax' });
      createCurvedConnection(prevDecBlock.position, linear.position, 0xffffff, 0);
      createCurvedConnection(linear.position, softmax.position, 0xffffff, 0);
      addOutline(linear, 0x808080, 0.06);
      addOutline(softmax, 0x32cd32, 0.06);
      createTextLabel("Output", new THREE.Vector3(dec_x, last_decoder_y + 5.5, 0), 0.7, "#22c55e");
  }

  // --- REFINED TRANSFORMER VISUALIZATION ---
  function createTransformerNetwork() {
      const { variant } = state.architecture.transformer;
      switch (variant) {
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
      camState.targetDistance = calculateOptimalDistance([]);
      camState.distance = camState.targetDistance;
  }

  function createEncoderOnlyTransformer() {
      const { encoderBlocks } = state.architecture.transformer;
      const enc_x = 0; // Centered
      const y_base = 0;

      const inputEmbed = createTransformerComponent('Input Embedding', new THREE.Vector3(enc_x, y_base - 7, 0), new THREE.BoxGeometry(3, 1, 1), 0xffb6c1, { type: 'embedding', stack: 'encoder' });
      addOutline(inputEmbed, 0x22d3ee, 0.09);

      const encBlockHeight = 2.5;
      const encBlockSpacing = 3.5;
      const encStackHeight = (encoderBlocks - 1) * encBlockSpacing + encBlockHeight;
      const encStackCenterY = y_base + ((encoderBlocks - 1) * encBlockSpacing) / 2;

      addDottedBox(new THREE.Vector3(enc_x, encStackCenterY + 0.5, 0), {x: 3.2, y: encStackHeight + 1.5, z: 1.6}, 0x2196f3);
      createTextLabel("Encoder Stack", new THREE.Vector3(enc_x, encStackCenterY + (encStackHeight + 1.5) / 2 + 1.0, 0), 0.8, "#22d3ee");
      createPositionalEncodingVisualization(enc_x, encStackCenterY, 'encoder', encStackHeight);

      let prevEncBlock = inputEmbed;
      for (let i = 0; i < encoderBlocks; i++) {
          const y = y_base + i * encBlockSpacing;
          const encBlockPos = new THREE.Vector3(enc_x, y + encBlockHeight / 2, 0);
          createCurvedConnection(prevEncBlock.position, encBlockPos, 0xffffff, 0);
          prevEncBlock = { position: encBlockPos };

          const attn = createTransformerComponent('Multi-Head Self-Attention', new THREE.Vector3(enc_x, y + 0.3, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0xffa500, { type: 'attention', stack: 'encoder', layer: i });
          const ff = createTransformerComponent('Feed Forward', new THREE.Vector3(enc_x, y + 1.5, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0x87ceeb, { type: 'ff', stack: 'encoder', layer: i });
          addOutline(attn, 0xffa500, 0.05);
          addOutline(ff, 0x87ceeb, 0.05);
          createCurvedConnection(attn.position, ff.position, 0xcccccc, 1.0);
      }

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
      const { decoderBlocks } = state.architecture.transformer;
      const dec_x = 0; // Centered
      const y_base = 0;

      const outputEmbed = createTransformerComponent('Token Embedding', new THREE.Vector3(dec_x, y_base - 7, 0), new THREE.BoxGeometry(3, 1, 1), 0xffb6c1, { type: 'embedding', stack: 'decoder' });
      addOutline(outputEmbed, 0x22d3ee, 0.09);

      const decBlockHeight = 3.0;
      const decBlockSpacing = 4.0;
      const decStackHeight = (decoderBlocks - 1) * decBlockSpacing + decBlockHeight;
      const decStackCenterY = y_base + ((decoderBlocks - 1) * decBlockSpacing) / 2;

      addDottedBox(new THREE.Vector3(dec_x, decStackCenterY + 0.5, 0), {x: 3.2, y: decStackHeight + 1.5, z: 1.6}, 0xa855f7);
      createTextLabel("Decoder Stack", new THREE.Vector3(dec_x, decStackCenterY + (decStackHeight + 1.5) / 2 + 1.0, 0), 0.8, "#a855f7");
      createPositionalEncodingVisualization(dec_x, decStackCenterY, 'decoder', decStackHeight);

      let prevDecBlock = outputEmbed;
      for (let i = 0; i < decoderBlocks; i++) {
          const y = y_base + i * decBlockSpacing;
          const decBlockPos = new THREE.Vector3(dec_x, y + decBlockHeight / 2, 0);
          createCurvedConnection(prevDecBlock.position, decBlockPos, 0xffffff, 0);
          prevDecBlock = { position: decBlockPos };

          const maskedAttn = createTransformerComponent('Masked Multi-Head Self-Attention', new THREE.Vector3(dec_x, y + 0.3, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0x9932cc, { type: 'masked_attention', stack: 'decoder', layer: i });
          const ff = createTransformerComponent('Feed Forward', new THREE.Vector3(dec_x, y + 1.8, 0), new THREE.BoxGeometry(1.8, 0.45, 0.5), 0x87ceeb, { type: 'ff', stack: 'decoder', layer: i });
          addOutline(maskedAttn, 0x9932cc, 0.05);
          addOutline(ff, 0x87ceeb, 0.05);
          createCurvedConnection(maskedAttn.position, ff.position, 0xcccccc, 1.0);
      }

      const last_decoder_y = y_base + (decoderBlocks - 1) * decBlockSpacing + decBlockHeight;
      const lmHead = createTransformerComponent('Language Model Head', new THREE.Vector3(dec_x, last_decoder_y + 2.0, 0), new THREE.BoxGeometry(2, 0.8, 0.8), 0x808080, { type: 'lm_head' });
      const softmax = createTransformerComponent('Softmax', new THREE.Vector3(dec_x, last_decoder_y + 4.0, 0), new THREE.SphereGeometry(0.6, 16, 12), 0x32cd32, { type: 'softmax' });
      createCurvedConnection(prevDecBlock.position, lmHead.position, 0xffffff, 0);
      createCurvedConnection(lmHead.position, softmax.position, 0xffffff, 0);
      addOutline(lmHead, 0x808080, 0.06);
      addOutline(softmax, 0x32cd32, 0.06);
      createTextLabel("Next Token Prediction", new THREE.Vector3(dec_x, last_decoder_y + 5.5, 0), 0.7, "#22c55e");
  }

  // --- MoE ARCHITECTURE BUILDER ---
  function createMoENetwork() {
      const { expertLayers } = state.architecture.moe;
      const layer = expertLayers[0]; // Reverted to simple, single-layer logic
      const expert_spacing = 8.0;

      // --- Vertical Layout Positions (Corrected for proper spacing and to be above grid) ---
      const router_y = 22;
      const prob_y = 12;
      const experts_y = 0;
      const usage_y = -10;
      const combiner_y = -18;

      // --- 1. Router (Top) ---
      const router_fnn = createTransformerComponent('Gating FNN', new THREE.Vector3(0, router_y, 0), new THREE.BoxGeometry(5, 3, 2), 0xffd700, { type: 'moe_router_fnn' });
      const router_softmax = createTransformerComponent('Softmax', new THREE.Vector3(0, router_y - 4, 0), new THREE.SphereGeometry(1.2), 0xfeca57, { type: 'moe_router_softmax' });
      createCurvedConnection(router_fnn.position, router_softmax.position, 0xffffff, 0);
      addDottedBox(new THREE.Vector3(0, router_y - 1, 0), {x: 8, y: 8, z: 5}, 0xffd700);
      createTextLabel("Router", new THREE.Vector3(0, router_y + 5, 0), 0.8, "#ffd700");

      // --- 2. Probabilities Layer (Gate Values Histogram) ---
      const total_prob_width = (layer.experts - 1) * expert_spacing;
      const prob_start_x = -total_prob_width / 2;
      for (let i = 0; i < layer.experts; i++) {
          const hist_x = prob_start_x + i * expert_spacing;
          const gateBar = createTransformerComponent(null, new THREE.Vector3(hist_x, prob_y, 0), new THREE.BoxGeometry(1.5, 0.1, 0.4), 0x22d3ee, { type: 'moe_gate_bar', layer: 0, expert: i });
          gateBar.userData.base_y = prob_y;
          addOutline(gateBar, 0x22d3ee, 0.05);
          createCurvedConnection(router_softmax.position, gateBar.position, 0xcccccc, 0);
      }
      addDottedBox(new THREE.Vector3(0, prob_y, 0), {x: total_prob_width + 8, y: 4, z: 5}, 0x22d3ee);
      createTextLabel("Probabilities", new THREE.Vector3(0, prob_y + 3, 0), 0.8, "#22d3ee");

      // --- 3. Expert Layer ---
      const total_expert_width = (layer.experts - 1) * expert_spacing;
      const expert_start_x = -total_expert_width / 2;
      for (let i = 0; i < layer.experts; i++) {
          const expert_x = expert_start_x + i * expert_spacing;
          const expert_color = new THREE.Color().setHSL(i / layer.experts, 0.7, 0.6);

          const expert_block = createTransformerComponent(`E${i + 1}`, new THREE.Vector3(expert_x, experts_y, 0), new THREE.BoxGeometry(4, 6, 3), expert_color, { type: 'moe_expert', layer: 0, expert_id: i });
          addOutline(expert_block, expert_color, 0.08);

          createCurvedConnection(new THREE.Vector3(expert_x, prob_y, 0), expert_block.position, 0xcccccc, 0);
      }
      // Corrected Expert Pool Outline position
      addDottedBox(new THREE.Vector3(0, experts_y + 3, 0), {x: total_expert_width + 8, y: 10, z: 6}, 0xa855f7);
      createTextLabel(`Expert Pool - ${layer.name}`, new THREE.Vector3(0, experts_y + 9, 0), 0.8, "#a855f7");

      // --- 4. Expert Usage Histogram ---
      const usageBars = [];
      for (let i = 0; i < layer.experts; i++) {
          const usage_x = expert_start_x + i * expert_spacing;
          const usageBar = createTransformerComponent(null, new THREE.Vector3(usage_x, usage_y, 0), new THREE.BoxGeometry(2, 0.1, 0.6), 0x4ecdc4, { type: 'moe_usage_bar', layer: 0, expert: i });
          usageBar.userData.base_y = usage_y;
          addOutline(usageBar, 0x4ecdc4, 0.05);
          createCurvedConnection(new THREE.Vector3(usage_x, experts_y, 0), usageBar.position, 0xcccccc, 0);
          usageBars.push(usageBar);
      }
      addDottedBox(new THREE.Vector3(0, usage_y, 0), {x: total_expert_width + 8, y: 4, z: 5}, 0x4ecdc4);
      createTextLabel(`Usage`, new THREE.Vector3(0, usage_y + 3, 0), 0.8, "#4ecdc4");

      // --- 5. Combiner (Bottom) ---
      const combiner = createTransformerComponent('Weighted Sum', new THREE.Vector3(0, combiner_y, 0), new THREE.SphereGeometry(2.5), 0x22c55e, { type: 'moe_combiner' });
      addOutline(combiner, 0x22c55e, 0.08);
      addDottedBox(new THREE.Vector3(0, combiner_y, 0), {x: 6, y: 6, z: 6}, 0x22c55e);
      createTextLabel("Output", new THREE.Vector3(0, combiner_y + 4, 0), 0.8, "#22c55e");

      // Connect all usage bars (representing expert outputs) to the combiner
      usageBars.forEach(bar => {
        createCurvedConnection(bar.position, combiner.position, 0xcccccc, 0);
      });

      camState.targetDistance = calculateOptimalDistance([]);
      camState.distance = camState.targetDistance;
  }

  // --- NETWORK CREATION DISPATCH ---
  function createNetwork() {
    const previousOperatorKey = state.selectedOperatorKey;
    // Clear existing network
    neurons.forEach(layer => layer.forEach(n => { scene.remove(n); n.geometry.dispose(); n.material.dispose(); }));
    connections.forEach(c => { scene.remove(c); c.geometry.dispose(); c.material.dispose(); });
    transformerObjects.forEach(obj => { scene.remove(obj); if(obj.geometry) obj.geometry.dispose(); if(obj.material) obj.material.dispose(); });
    particles.forEach(p => scene.remove(p));

    neurons = [];
    connections = [];
    transformerObjects = [];
    particles = [];
    interactiveObjects = [];
    operatorModuleMeta = [];
    hoveredOperatorObject = null;
    selectedOperatorObject = null;
    state.selectedOperatorKey = null;

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
      case 'operator':
        createOperatorNetwork();
        state.animationStep = state.animationStep % getOperatorStepCount();
        applyOperatorSelectionByKey(previousOperatorKey);
        break;
      case 'moe':
        createMoENetwork();
        break;
      case 'autoencoder':
      case 'fnn':
      default:
        const layers = getCurrentLayerConfig();
        createGenericNetwork(layers);
        break;
    }

    updateOperatorLegendVisibility();
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

      case 'operator':
        // Neural operator layout spreads horizontally with stacked blocks
        const opBlocks = currentArch.spectralBlocks || 1;
        const opWidth = (opBlocks + 4) * 5 + 18;
        const opHeight = 26;
        maxDimension = Math.max(opWidth, opHeight, 36);
        break;

      case 'moe':
        // Circular expert arrangement
        const numExperts = currentArch.experts || 4;
        const radius = Math.max(5, numExperts * 2.5); // Wider for pipeline layout
        maxDimension = Math.max(radius * 2.5, 40); // Diameter + padding
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
    // Move grid under the model
    if (archType === 'moe') {
        gridHelper.position.y = -22; // Position grid under the MoE model
    } else {
        gridHelper.position.y = -8;
    }
    gridHelper.visible = state.viewMode === '3d';
    scene.add(gridHelper);

    return gridSize;
  }

  // --- NEURAL OPERATOR ARCHITECTURE BUILDERS ---
  const OPERATOR_COLORS = {
    input: { fill: 0xf59e0b, outline: 0xb45309 },
    projection: { fill: 0x60a5fa, outline: 0x2563eb },
    spectral: { fill: 0xf472b6, outline: 0xdb2777 },
    mlp: { fill: 0xc084fc, outline: 0x7c3aed },
    arrow: 0x94a3b8,
    highlight: 0x34d399,
    hover: 0x38bdf8,
    selected: 0xf8fafc
  };

  function addOperatorOutline(mesh, outlineColor, layerIndex, moduleKey) {
    const edges = new THREE.EdgesGeometry(mesh.geometry);
    const outlineMaterial = new THREE.LineBasicMaterial({
      color: outlineColor,
      transparent: true,
      opacity: 0.9
    });
    const outline = new THREE.LineSegments(edges, outlineMaterial);
    outline.position.copy(mesh.position);
    outline.rotation.copy(mesh.rotation);
    outline.userData = {
      layerIndex,
      moduleKey,
      baseColor: outlineColor,
      type: 'operator_outline',
      kind: 'outline'
    };
    scene.add(outline);
    transformerObjects.push(outline);
    return outline;
  }

  function createLayerStack(config, basePosition, layerIndex) {
    const is2D = state.viewMode === '2d';
    const stackCount = config.stackCount || 1;
    const stackOffset = (config.stackOffset || 0.2) * (is2D ? 0.45 : 1);
    const depthScale = is2D ? 0.38 : 1;
    const rotationY = is2D ? 0 : -0.22;
    const xOffsetScale = is2D ? 0.38 : 0.55;
    const zOffsetScale = is2D ? 0.18 : 1;
    const meshes = [];

    for (let i = stackCount - 1; i >= 0; i--) {
      const depthIndex = stackCount - 1 - i;
      const scale = 1 - depthIndex * 0.03;
      const geometry = new THREE.BoxGeometry(
        config.size.x * scale,
        config.size.y * scale,
        config.size.z * depthScale
      );
      const opacity = Math.min(0.94, 0.58 + i * 0.12);
      const material = new THREE.MeshPhongMaterial({
        color: config.color,
        transparent: true,
        opacity,
        shininess: 90
      });
      const mesh = new THREE.Mesh(geometry, material);
      const offset = depthIndex * stackOffset;
      const offsetX = offset * xOffsetScale;
      const offsetZ = offset * zOffsetScale;
      const isInteractive = depthIndex === 0;
      mesh.position.set(
        basePosition.x - offsetX,
        basePosition.y + (config.size.y * scale) / 2,
        basePosition.z + offsetZ
      );
      mesh.rotation.y = rotationY;
      mesh.userData = {
        ...config.userData,
        layerIndex,
        moduleKey: config.key,
        label: config.label,
        sublabel: config.sublabel,
        description: config.description,
        dims: config.dims,
        baseColor: config.color,
        type: config.type,
        kind: 'block',
        interactive: isInteractive
      };
      scene.add(mesh);
      transformerObjects.push(mesh);
      addOperatorOutline(mesh, config.outlineColor, layerIndex, config.key);
      if (isInteractive) {
        interactiveObjects.push(mesh);
      }
      meshes.push(mesh);
    }

    const labelBaseY = basePosition.y + config.size.y + 0.9;
    const labelZ = basePosition.z + stackCount * 0.12;
    const labelSize = is2D ? 0.72 : 0.7;
    createTextLabel(config.label, new THREE.Vector3(basePosition.x, labelBaseY, labelZ), labelSize, '#e2e8f0');
    if (config.sublabel) {
      createTextLabel(
        config.sublabel,
        new THREE.Vector3(basePosition.x, labelBaseY - 0.65, labelZ),
        is2D ? 0.56 : 0.52,
        '#94a3b8'
      );
    }

    return {
      center: new THREE.Vector3(basePosition.x, basePosition.y + config.size.y / 2, basePosition.z),
      size: config.size,
      layerIndex,
      type: config.type,
      key: config.key,
      dims: config.dims,
      description: config.description,
      label: config.label,
      sublabel: config.sublabel
    };
  }

  function createOperatorArrow(fromMeta, toMeta, label) {
    const is2D = state.viewMode === '2d';
    const midY = (fromMeta.center.y + toMeta.center.y) / 2;
    const start = new THREE.Vector3(fromMeta.center.x + fromMeta.size.x / 2 + 0.65, midY, 0);
    const end = new THREE.Vector3(toMeta.center.x - toMeta.size.x / 2 - 0.65, midY, 0);
    const lineGeometry = new THREE.BufferGeometry().setFromPoints([start, end]);
    const lineMaterial = new THREE.LineBasicMaterial({
      color: OPERATOR_COLORS.arrow,
      transparent: true,
      opacity: 0.85
    });
    const line = new THREE.Line(lineGeometry, lineMaterial);
    line.userData = {
      layerIndex: toMeta.layerIndex,
      moduleKey: toMeta.key,
      baseColor: OPERATOR_COLORS.arrow,
      type: 'operator_arrow',
      kind: 'arrow'
    };
    scene.add(line);
    transformerObjects.push(line);

    const coneGeometry = new THREE.ConeGeometry(is2D ? 0.24 : 0.28, is2D ? 0.7 : 0.8, 12);
    const coneMaterial = new THREE.MeshPhongMaterial({
      color: OPERATOR_COLORS.arrow,
      transparent: true,
      opacity: 0.9,
      shininess: 120
    });
    const cone = new THREE.Mesh(coneGeometry, coneMaterial);
    cone.position.copy(end);
    cone.rotation.z = -Math.PI / 2;
    cone.userData = {
      layerIndex: toMeta.layerIndex,
      moduleKey: toMeta.key,
      baseColor: OPERATOR_COLORS.arrow,
      type: 'operator_arrow_head',
      kind: 'arrow'
    };
    scene.add(cone);
    transformerObjects.push(cone);

    if (label) {
      const labelPos = start.clone().lerp(end, 0.5);
      labelPos.y += 0.75;
      createTextLabel(label, labelPos, is2D ? 0.54 : 0.5, '#94a3b8');
    }
  }

  function updatePointerFromEvent(event) {
    const rect = canvas3d.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  function getOperatorObjectAtPointer() {
    if (state.archType !== 'operator' || interactiveObjects.length === 0) return null;
    raycaster.setFromCamera(pointer, camera);
    const intersections = raycaster.intersectObjects(interactiveObjects, false);
    return intersections.length ? intersections[0].object : null;
  }

  function updateOperatorDetailsPanel() {
    if (!operatorDetails) return;
    const activeObj = hoveredOperatorObject || selectedOperatorObject;
    if (!activeObj || state.archType !== 'operator') {
      operatorDetailsTitle.textContent = 'Select a block';
      operatorDetailsType.textContent = 'Operator Block';
      operatorDetailsMeta.innerHTML = '';
      operatorDetailsDescription.textContent = 'Hover or click a module to inspect its role in the operator pipeline.';
      return;
    }
    const { label, type, sublabel, description, dims, layerIndex, moduleKey } = activeObj.userData;
    const prettyType = type
      .replace('operator_', '')
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
    const isSelected = activeObj === selectedOperatorObject;
    operatorDetailsTitle.textContent = label || 'Operator Block';
    operatorDetailsType.textContent = isSelected ? `Selected • ${prettyType}` : prettyType;

    const metaItems = [
      { label: 'Stage', value: layerIndex + 1 },
      { label: 'View', value: state.viewMode.toUpperCase() }
    ];
    if (sublabel) {
      metaItems.push({ label: 'Shape', value: sublabel });
    }
    if (dims?.channels) {
      metaItems.push({ label: 'Channels', value: dims.channels });
    }
    if (dims?.modes) {
      metaItems.push({ label: 'Modes', value: dims.modes });
    }
    if (dims?.block) {
      metaItems.push({ label: 'Block', value: dims.block });
    }
    operatorDetailsMeta.innerHTML = metaItems
      .map(item => `<div><strong>${item.label}:</strong> ${item.value}</div>`)
      .join('');

    const detailText = description || `Module key: ${moduleKey}`;
    operatorDetailsDescription.textContent = detailText;
  }

  function setOperatorSelection(obj) {
    selectedOperatorObject = obj || null;
    state.selectedOperatorKey = selectedOperatorObject?.userData?.moduleKey || null;
    updateOperatorDetailsPanel();
  }

  function applyOperatorSelectionByKey(key) {
    if (!key || state.archType !== 'operator') {
      setOperatorSelection(null);
      return;
    }
    const match = interactiveObjects.find(obj => obj.userData.moduleKey === key);
    setOperatorSelection(match || null);
  }

  function handlePointerMove(event) {
    updatePointerFromEvent(event);
    if (state.archType !== 'operator') return;
    if (isDragging && state.cameraMode === 'manual') return;
    const nextHover = getOperatorObjectAtPointer();
    if (hoveredOperatorObject !== nextHover) {
      hoveredOperatorObject = nextHover;
      updateOperatorDetailsPanel();
    }
  }

  function handleOperatorClick(event) {
    if (state.archType !== 'operator') return;
    updatePointerFromEvent(event);
    if (lastDragWasPan) {
      lastDragWasPan = false;
      return;
    }
    const clicked = getOperatorObjectAtPointer();
    if (clicked) {
      setOperatorSelection(clicked === selectedOperatorObject ? null : clicked);
    } else {
      setOperatorSelection(null);
    }
  }

  function getOperatorStepCount() {
    const spectralBlocks = state.architecture.operator?.spectralBlocks || 1;
    return spectralBlocks + 4;
  }

  function updateOperatorLegendVisibility() {
    if (!operatorLegend) return;
    const isOperator = state.archType === 'operator';
    operatorLegend.classList.toggle('hidden', !isOperator);
    operatorDetails?.classList.toggle('hidden', !isOperator);
    applyViewModeClass();
    if (!isOperator) {
      hoveredOperatorObject = null;
      selectedOperatorObject = null;
      state.selectedOperatorKey = null;
    }
    updateOperatorDetailsPanel();
  }

  function resetOperatorVisuals() {
    transformerObjects.forEach(obj => {
      const mat = obj.material;
      if (!mat) return;
      if (mat.emissive) {
        mat.emissive.setHex(0x000000);
        mat.emissiveIntensity = 0.85;
      }
      if (obj.userData?.baseColor && mat.color) {
        mat.color.setHex(obj.userData.baseColor);
      }
    });
  }

  function highlightOperatorLayer(layerIndex, color, intensity = 0.95) {
    transformerObjects
      .filter(obj => obj.userData.layerIndex === layerIndex)
      .forEach(obj => {
        const mat = obj.material;
        if (!mat) return;
        if (mat.emissive) {
          mat.emissive.setHex(color);
          mat.emissiveIntensity = intensity;
        } else if (mat.color) {
          mat.color.setHex(color);
        }
      });
  }

  function applyOperatorInteractionHighlights() {
    if (hoveredOperatorObject && hoveredOperatorObject !== selectedOperatorObject) {
      highlightOperatorLayer(hoveredOperatorObject.userData.layerIndex, OPERATOR_COLORS.hover, 1.0);
    }
    if (selectedOperatorObject) {
      highlightOperatorLayer(selectedOperatorObject.userData.layerIndex, OPERATOR_COLORS.selected, 1.05);
    }
  }

  function getOperatorModules() {
    const operatorCfg = state.architecture.operator;
    const is2D = state.viewMode === '2d';
    const spectralBlocks = Math.max(1, operatorCfg.spectralBlocks || 1);
    const baseSpectralHeight = is2D ? 8.4 : 7.8;
    const spectralHeightStep = Math.min(0.6, spectralBlocks * 0.08);
    const widthScale = is2D ? 1.08 : 1;
    const heightScale = is2D ? 1.05 : 1;
    const spectralModes = Math.max(8, Math.round(operatorCfg.width / 4));

    const modules = [
      {
        key: 'input',
        label: 'Input',
        sublabel: `N × ${operatorCfg.inputDim}`,
        size: { x: 2.9 * widthScale, y: 8.6 * heightScale, z: 1.1 },
        color: OPERATOR_COLORS.input.fill,
        outlineColor: OPERATOR_COLORS.input.outline,
        stackCount: 2,
        stackOffset: 0.24,
        type: 'operator_input',
        dims: { channels: operatorCfg.inputDim },
        description: 'Input coordinates or field values lifted into the operator pipeline.'
      },
      {
        key: 'projection',
        label: 'Projection',
        sublabel: `N × ${operatorCfg.projectionDim}`,
        size: { x: 3.3 * widthScale, y: 8.3 * heightScale, z: 1.15 },
        color: OPERATOR_COLORS.projection.fill,
        outlineColor: OPERATOR_COLORS.projection.outline,
        stackCount: 2,
        stackOffset: 0.26,
        type: 'operator_projection',
        dims: { channels: operatorCfg.projectionDim },
        description: 'A learned linear lift maps inputs to a wider latent channel space.'
      }
    ];

    for (let i = 0; i < spectralBlocks; i++) {
      modules.push({
        key: `spectral-${i + 1}`,
        label: `Spectral ${i + 1}`,
        sublabel: `${operatorCfg.width} × N`,
        size: { x: 3.15 * widthScale, y: (baseSpectralHeight - spectralHeightStep + i * 0.1) * heightScale, z: 1.2 },
        color: OPERATOR_COLORS.spectral.fill,
        outlineColor: OPERATOR_COLORS.spectral.outline,
        stackCount: 3,
        stackOffset: 0.21,
        type: 'operator_spectral',
        dims: { channels: operatorCfg.width, modes: spectralModes, block: i + 1 },
        description: `Spectral convolution ${i + 1} mixes global Fourier modes with learned filters.`,
        userData: { block: i + 1 }
      });
    }

    modules.push(
      {
        key: 'mlp',
        label: 'MLP Head',
        sublabel: `N × ${operatorCfg.mlpDim}`,
        size: { x: 4.25 * widthScale, y: 9.4 * heightScale, z: 1.35 },
        color: OPERATOR_COLORS.mlp.fill,
        outlineColor: OPERATOR_COLORS.mlp.outline,
        stackCount: 2,
        stackOffset: 0.2,
        type: 'operator_mlp',
        dims: { channels: operatorCfg.mlpDim },
        description: 'Point-wise MLP refines the latent operator output back into physical space.'
      },
      {
        key: 'output',
        label: 'Output',
        sublabel: `N × ${operatorCfg.outputDim}`,
        size: { x: 2.95 * widthScale, y: 8.6 * heightScale, z: 1.1 },
        color: OPERATOR_COLORS.input.fill,
        outlineColor: OPERATOR_COLORS.input.outline,
        stackCount: 2,
        stackOffset: 0.24,
        type: 'operator_output',
        dims: { channels: operatorCfg.outputDim },
        description: 'Projected solution field returned to the original dimensionality.'
      }
    );

    return modules;
  }

  function createOperatorNetwork() {
    const modules = getOperatorModules();
    const yBase = -8;
    const baseSpacing = state.viewMode === '2d' ? 6.2 : 5.8;
    const spacingDecay = state.viewMode === '2d' ? 0.985 : 0.97;
    let spacing = baseSpacing;
    let totalWidth = 0;

    modules.forEach((module, index) => {
      totalWidth += module.size.x;
      if (index < modules.length - 1) {
        totalWidth += spacing;
        spacing *= spacingDecay;
      }
    });

    let cursorX = -totalWidth / 2;
    spacing = baseSpacing;
    const moduleMeta = [];

    modules.forEach((module, index) => {
      cursorX += module.size.x / 2;
      const meta = createLayerStack(module, new THREE.Vector3(cursorX, yBase, 0), index);
      moduleMeta.push(meta);
      cursorX += module.size.x / 2;
      if (index < modules.length - 1) {
        cursorX += spacing;
        spacing *= spacingDecay;
      }
    });

    operatorModuleMeta = moduleMeta;
    camState.panX = 0;
    camState.panY = 0;
    camState.targetPanX = 0;
    camState.targetPanY = 0;

    const spectralBlocks = Math.max(1, state.architecture.operator?.spectralBlocks || 1);
    const spectralStart = 2;
    const spectralEnd = spectralStart + spectralBlocks - 1;

    for (let i = 0; i < moduleMeta.length - 1; i++) {
      let label = 'Linear';
      if (i === 1) label = 'Permute';
      else if (i >= spectralStart && i < spectralEnd) label = 'Spectral';
      else if (i === spectralEnd) label = 'MLP';
      createOperatorArrow(moduleMeta[i], moduleMeta[i + 1], label);
    }

    const gridSize = calculateGridSize('operator', state.architecture.operator);
    camState.targetDistance = state.viewMode === '2d'
      ? Math.max(54, gridSize * 0.74)
      : Math.max(40, gridSize * 0.7);
    camState.distance = camState.targetDistance;
    updateOrthographicFrustum();
  }

  // --- CNN ARCHITECTURE BUILDERS ---
  function createCNNNetwork() {
    const variant = state.architecture.cnn.variant;
    const y_base = -8; // Set base to the grid level

    let layerConfigs = getCNNLayerConfig(variant);

    // Calculate total width to center the model
    let totalWidth = 0;
    let tempSpacing = 7.0;
    layerConfigs.forEach(() => {
        totalWidth += tempSpacing;
        tempSpacing *= 0.95;
    });
    const startX = -totalWidth / 2;

    // Input image as a thin 3D block
    const inputGeo = new THREE.BoxGeometry(4, 4, 0.2);
    const inputComp = createTransformerComponent('Input Image', new THREE.Vector3(startX, y_base, 0), inputGeo, 0x3b82f6, { type: 'input', layer: 0 });

    let lastPosition = inputComp.position.clone();
    let horizontalSpacing = 7.0;
    let currentScale = 1.0;

    layerConfigs.forEach((config, index) => {
        const x = lastPosition.x + horizontalSpacing;
        currentScale *= 0.92; // Gradually reduce scale for the funnel effect
        horizontalSpacing *= 0.95; // Gradually reduce spacing

        // Clone and scale the original geometry to create the funnel effect
        let funnelGeometry = config.geometry.clone();
        let params = funnelGeometry.parameters;

        if (funnelGeometry instanceof THREE.BoxGeometry) {
            params.width *= currentScale;
            params.height *= currentScale;
            params.depth *= currentScale; // Scale depth for a 3D funnel
        } else if (funnelGeometry instanceof THREE.SphereGeometry) {
            params.radius *= currentScale;
        } else if (funnelGeometry instanceof THREE.CylinderGeometry) {
            params.radiusTop *= currentScale;
            params.radiusBottom *= currentScale;
            params.height *= currentScale;
        } else if (funnelGeometry instanceof THREE.DodecahedronGeometry || funnelGeometry instanceof THREE.OctahedronGeometry || funnelGeometry instanceof THREE.ConeGeometry) {
             params.radius *= currentScale;
             if(params.height) params.height *= currentScale;
        }

        // Re-create the geometry from scaled parameters
        funnelGeometry = new config.geometry.constructor(...Object.values(params));

        const component = createTransformerComponent(config.name, new THREE.Vector3(x, y_base, 0), funnelGeometry, config.color, { type: config.type, layer: index + 1 });

        createCurvedConnection(lastPosition, component.position, 0xffffff, 0);
        lastPosition = component.position;
    });

    // Set camera distance based on grid size
    const gridSize = calculateGridSize('cnn', state.architecture.cnn);
    camState.targetDistance = Math.max(50, gridSize * 0.9);
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
    let layerCount = 0;
    if (Array.isArray(currentArch)) {
      layerCount = currentArch.length;
    } else if (state.archType === 'operator') {
      layerCount = getOperatorStepCount();
    } else {
      layerCount = (currentArch.encoderBlocks || 0) + (currentArch.decoderBlocks || 0) + (currentArch.experts || 0) + (currentArch.depth || 0);
    }
    const neuronValue = Array.isArray(currentArch)
      ? currentArch.reduce((s, l) => s + l.neurons, 0)
      : state.archType === 'operator'
        ? currentArch.width
        : 'N/A';
    const neuronLabel = state.archType === 'operator' ? 'channels' : 'neurons';
    document.getElementById('network-stats').textContent = `${layerCount} components • ${neuronValue} ${neuronLabel}`;

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
    document.getElementById('theme-toggle').checked = state.theme === 'light';
    view3dBtn?.classList.toggle('active', state.viewMode === '3d');
    view2dBtn?.classList.toggle('active', state.viewMode === '2d');
    view3dBtn?.setAttribute('aria-pressed', String(state.viewMode === '3d'));
    view2dBtn?.setAttribute('aria-pressed', String(state.viewMode === '2d'));
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
      case 'operator':
        html = `<h3 style="color: var(--cyan);">Neural Operator</h3>
          <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div><label>Spectral Blocks:</label><input type="number" min="1" max="6" value="${state.architecture.operator.spectralBlocks}" class="update-arch-input" data-field="spectralBlocks"></div>
            <div><label>Width (Channels):</label><input type="number" min="32" max="256" step="16" value="${state.architecture.operator.width}" class="update-arch-input" data-field="width"></div>
            <div><label>Projection Dim:</label><input type="number" min="32" max="256" step="16" value="${state.architecture.operator.projectionDim}" class="update-arch-input" data-field="projectionDim"></div>
            <div><label>MLP Dim:</label><input type="number" min="64" max="512" step="32" value="${state.architecture.operator.mlpDim}" class="update-arch-input" data-field="mlpDim"></div>
            <div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.5rem;">
              <div><label>Input Dim:</label><input type="number" min="1" max="16" value="${state.architecture.operator.inputDim}" class="update-arch-input" data-field="inputDim"></div>
              <div><label>Output Dim:</label><input type="number" min="1" max="16" value="${state.architecture.operator.outputDim}" class="update-arch-input" data-field="outputDim"></div>
            </div>
            <div style="margin-top: 0.25rem; padding: 0.6rem; background: var(--layer-item-bg); border-radius: 4px; font-size: 0.75rem; border-left: 3px solid var(--cyan);">
              Spectral operators lift inputs into a wider latent space, apply repeated spectral convolutions, then project back through an MLP head.
            </div>
          </div>`;
        break;
      case 'moe':
        const moeVariants = ['ST-MoE', 'GLaM', 'Switch'];
        html = `<h3 style="color: var(--green);">Mixture of Experts</h3>
          <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div>
              <label>Variant:</label>
              <select id="moe-variant-select" class="update-moe-variant">
                ${moeVariants.map(opt => `<option value="${opt}" ${state.architecture.moe.variant === opt ? 'selected' : ''}>${opt}</option>`).join('')}
              </select>
            </div>
            <div><label>Top-K:</label><input type="number" min="1" max="4" value="${state.architecture.moe.top_k}" class="update-arch-input" data-field="top_k"></div>
            <div style="margin-top: 0.75rem;">
              <label style="font-weight: bold;">Expert Layer:</label>
              ${state.architecture.moe.expertLayers.map((layer, i) => `
                <div class="expert-layer-item">
                  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-weight: bold;">${layer.name}</span>
                  </div>
                  <div style="display: flex; gap: 0.5rem; align-items: center;">
                    <label style="margin: 0; font-size: 0.7rem;">Experts:</label>
                    <input type="number" min="2" max="8" value="${layer.experts}" data-layer="${i}" data-field="experts" class="update-expert-layer-input" style="width: 4rem; font-size: 0.7rem;">
                    <input type="text" value="${layer.name}" data-layer="${i}" data-field="name" class="update-expert-layer-input" style="flex: 1; font-size: 0.7rem;" placeholder="Layer name">
                  </div>
                </div>
              `).join('')}
            </div>
            <div style="margin-top: 1rem;">
              <h4 style="color: var(--cyan); margin-bottom: 0.5rem;">MoE Variants Explained</h4>
              <div class="moe-variant-info">
                <h5>ST-MoE: The Foundational Blueprint</h5>
                <p>The Sparsely-Gated MoE uses a trainable gating network to select the top-k experts for each token, making the model "sparse." An auxiliary load balancing loss is added to encourage the gating network to distribute tokens evenly across all available experts.</p>
              </div>
              <div class="moe-variant-info">
                <h5>GLaM: Scaling to Trillion-Parameter Models</h5>
                <p>GLaM replaces the FFN with an MoE layer in every other Transformer block. This alternating pattern, combined with Top-2 gating, proved highly effective, allowing GLaM to outperform dense models like GPT-3 with less energy.</p>
              </div>
              <div class="moe-variant-info">
                <h5>Switch Transformers: Simplifying for Speed</h5>
                <p>The Switch Transformer simplifies routing by sending each token to only one expert (k=1). This design reduces communication overhead in distributed training, resulting in significant speedups (up to 7x) compared to dense models.</p>
              </div>
            </div>
          </div>`;
        break;
      case 'autoencoder':
        html = `<h3 style="color: var(--purple);">Autoencoder</h3>
          <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div><label>Encoding Dimension:</label><input type="number" min="2" max="8" value="${state.architecture.autoencoder.encodingDim}" class="update-arch-input" data-field="encodingDim"></div>
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
                  <button class="remove-layer-btn" data-index="${i}" style="background: var(--red); color: white; border: none, padding: 0.25rem 0.5rem; border-radius: 4px; cursor: pointer;">Remove</button>
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

  function drawLossAccGraph() {
    const ctx = lossAccCanvas.getContext('2d');
    const w = lossAccCanvas.width;
    const h = lossAccCanvas.height;

    ctx.clearRect(0, 0, w, h);

    // Draw Loss (Red)
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--red').trim();
    ctx.lineWidth = 2;
    ctx.beginPath();
    lossHistory.forEach((val, i) => {
        const x = (i / Math.max(1, lossHistory.length - 1)) * w;
        const y = h - (val * h); // Assuming loss is 0-1
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw Accuracy (Green)
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--green').trim();
    ctx.lineWidth = 2;
    ctx.beginPath();
    accHistory.forEach((val, i) => {
        const x = (i / Math.max(1, accHistory.length - 1)) * w;
        const y = h - (val * h); // Assuming accuracy is 0-1
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  function renderSettingsPanel() {
    let html = `
      <div style="margin-bottom: 0.5rem;">
        <label>Theme:</label>
        <div class="flex gap-2 items-center">
          <span>Dark</span>
          <label class="switch">
            <input type="checkbox" id="theme-toggle" ${state.theme === 'light' ? 'checked' : ''}>
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
  view3dBtn?.addEventListener('click', () => switchViewMode('3d'));
  view2dBtn?.addEventListener('click', () => switchViewMode('2d'));

  document.getElementById('camera-mode-toggle').addEventListener('change', e => {
    state.cameraMode = e.target.checked ? 'manual' : 'auto';
  });


  const autoBtn = document.getElementById('cam-auto-btn');
  const manualBtn = document.getElementById('cam-manual-btn');

  panelDetailsContainer.addEventListener('click', e => {
    if (e.target.id === 'add-layer-btn' && state.archType === 'fnn') {
      state.architecture.fnn.splice(-1, 0, { neurons: 8, activation: 'ReLU', name: `Hidden ${state.architecture.fnn.length - 1}` });
      createNetwork();
      renderArchitecturePanel();
    }
    if (e.target.classList.contains('remove-layer-btn') && state.archType === 'fnn') {
      state.architecture.fnn.splice(parseInt(e.target.dataset.index), 1);
      createNetwork();
      renderArchitecturePanel();
    }
  });

  // MERGED EVENT LISTENER
  panelDetailsContainer.addEventListener('change', e => {
      // Theme toggle
      if (e.target.id === 'theme-toggle') {
          state.theme = e.target.checked ? 'light' : 'dark';
          localStorage.setItem('theme', state.theme);
          applyTheme();
          return; // Exit after handling
      }

      // FNN specific layer updates
      if (state.archType === 'fnn' && e.target.classList.contains('update-layer-input')) {
        const { index, field } = e.target.dataset;
        const value = e.target.type === 'number' ? parseInt(e.target.value) : e.target.value;
        state.architecture.fnn[index][field] = value;
        createNetwork();
      }
      // MoE Expert Updates
      if (state.archType === 'moe' && e.target.classList.contains('update-expert-layer-input')) {
          const { layer, field } = e.target.dataset;
          const layerIndex = parseInt(layer);
          const value = e.target.type === 'number' ? parseInt(e.target.value) : e.target.value;

          if (field === 'experts') {
              state.architecture.moe.expertLayers[layerIndex].experts = value;
              // Resize usage and gate arrays
              state.architecture.moe.expertUsage[layerIndex] = Array(value).fill(0);
              state.architecture.moe.gateValues[layerIndex] = Array(value).fill(0);
          } else {
              state.architecture.moe.expertLayers[layerIndex][field] = value;
          }
          createNetwork();
          renderArchitecturePanel();
      }
      // Generic architecture config updates
      if (e.target.classList.contains('update-arch-input')) {
          const { field } = e.target.dataset;
          const min = e.target.min ? parseInt(e.target.min, 10) : -Infinity;
          const max = e.target.max ? parseInt(e.target.max, 10) : Infinity;
          let value = parseInt(e.target.value, 10);
          if (Number.isNaN(value)) return;
          value = Math.min(max, Math.max(min, value));
          e.target.value = value;
          if (state.architecture[state.archType]) {
              state.architecture[state.archType][field] = value;
              if (state.archType === 'operator') {
                  state.animationStep = state.animationStep % getOperatorStepCount();
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
      // MoE Variant change
      if (e.target.id === 'moe-variant-select') {
          state.architecture.moe.variant = e.target.value;
          // Adjust top_k for Switch variant
          if (e.target.value === 'Switch') {
              state.architecture.moe.top_k = 1;
          }
          createNetwork();
          renderArchitecturePanel();
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
            const allObjects = [...neurons.flat(), ...transformerObjects];
            allObjects.forEach(obj => {
                if (obj.material) {
                    obj.material.emissive.setHex(0x0099ff);
                }
            });
        }, 100);
      }
      state.accuracy = newAcc;
      if (state.batch % 10 === 0) {
        state.epoch++;
        if (state.archType === 'transformer') {
            const { variant, encoderBlocks, decoderBlocks } = state.architecture.transformer;
            let totalSteps = 2; // input embed + final output
            if (variant === 'encoder-decoder') {
                totalSteps += encoderBlocks + decoderBlocks;
            } else if (variant === 'encoder-only') {
                totalSteps += encoderBlocks;
            } else if (variant === 'decoder-only') {
                totalSteps += decoderBlocks;
            }
            state.animationStep = (state.animationStep + 1) % totalSteps;
        }
        if (state.archType === 'cnn') {
            state.animationStep = (state.animationStep + 1) % 12; // Generic 12 steps for longer CNNs
        }
        if (state.archType === 'operator') {
            state.animationStep = (state.animationStep + 1) % getOperatorStepCount();
        }
        if (state.archType === 'moe') {
            // Enhanced MoE simulation with multiple layers
            const { expertLayers, top_k } = state.architecture.moe;
            let totalUsage = state.architecture.moe.expertUsage;
            let gateValues = state.architecture.moe.gateValues;

            expertLayers.forEach((layer, layerIndex) => {
                // Simulate gate values (selection probabilities)
                const gates = Array(layer.experts).fill(0).map(() => Math.random());
                const gateSum = gates.reduce((a, b) => a + b, 0);
                gateValues[layerIndex] = gates.map(g => g / gateSum); // Normalize to sum to 1

                // Simulate expert selection and usage update
                const activeExperts = [...Array(layer.experts).keys()]
                    .sort(() => 0.5 - Math.random())
                    .slice(0, Math.min(top_k, layer.experts));

                activeExperts.forEach(idx => {
                    if (!totalUsage[layerIndex]) totalUsage[layerIndex] = Array(layer.experts).fill(0);
                    totalUsage[layerIndex][idx]++;
                });

                // Apply load balancing (aux loss simulation)
                if (totalUsage[layerIndex]) {
                    const meanUsage = totalUsage[layerIndex].reduce((a, b) => a + b, 0) / layer.experts;
                    totalUsage[layerIndex] = totalUsage[layerIndex].map(u => THREE.MathUtils.lerp(u, meanUsage, 0.03));
                }
            });

            state.architecture.moe.expertUsage = totalUsage;
            state.architecture.moe.gateValues = gateValues;
            state.animationStep = { activeExperts: expertLayers.map((layer, idx) =>
                [...Array(layer.experts).keys()].sort(() => 0.5 - Math.random()).slice(0, Math.min(top_k, layer.experts))
            )};
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
      camState.targetPanX = 0;
      camState.targetPanY = 0;
    }

    const cameraLerp = state.cameraMode === 'auto' ? 0.04 : 0.12;
    camState.distance = THREE.MathUtils.lerp(camState.distance, camState.targetDistance, cameraLerp);
    camState.panX = THREE.MathUtils.lerp(camState.panX, camState.targetPanX, cameraLerp);
    camState.panY = THREE.MathUtils.lerp(camState.panY, camState.targetPanY, cameraLerp);

    if (state.viewMode === '2d') {
      camera = orthographicCamera;
      updateOrthographicFrustum();
      camera.position.set(camState.panX, camState.panY, camState.distance);
      camera.lookAt(camState.panX, camState.panY, 0);
    } else if (state.cameraMode === 'auto') {
      camera = perspectiveCamera;
      const orbitHeight = state.archType === 'operator' ? 0.34 : 0.4;
      camera.position.set(
        Math.cos(deltaTime * 0.15) * camState.distance + camState.panX,
        camState.distance * orbitHeight + camState.panY,
        Math.sin(deltaTime * 0.15) * camState.distance
      );
      camera.lookAt(camState.panX, camState.panY, 0);
    } else {
      camera = perspectiveCamera;
      const lerp = 0.1;
      camState.angleX = THREE.MathUtils.lerp(camState.angleX, camState.targetAngleX, lerp);
      camState.angleY = THREE.MathUtils.lerp(camState.angleY, camState.targetAngleY, lerp);
      camera.position.set(
        Math.cos(camState.angleX) * Math.cos(camState.angleY) * camState.distance + camState.panX,
        Math.sin(camState.angleY) * camState.distance + camState.panY,
        Math.sin(camState.angleX) * Math.cos(camState.angleY) * camState.distance
      );
      camera.lookAt(camState.panX, camState.panY, 0);
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
          const { variant, encoderBlocks, decoderBlocks } = state.architecture.transformer;

          transformerObjects.forEach(obj => {
              if (obj.material && obj.material.emissive) obj.material.emissive.setHex(0);
          });

          let activeComponent;
          if (variant === 'encoder-decoder') {
              if (step === 0) activeComponent = { type: 'embedding', stack: 'encoder' };
              else if (step <= encoderBlocks) activeComponent = { stack: 'encoder', layer: step - 1 };
              else if (step === encoderBlocks + 1) activeComponent = { type: 'embedding', stack: 'decoder' };
              else if (step <= encoderBlocks + decoderBlocks + 1) activeComponent = { stack: 'decoder', layer: step - 2 - encoderBlocks };
              else activeComponent = { type: 'output' };
          } else if (variant === 'encoder-only') {
              if (step === 0) activeComponent = { type: 'embedding' };
              else if (step <= encoderBlocks) activeComponent = { stack: 'encoder', layer: step - 1 };
              else activeComponent = { type: 'output' };
          } else { // decoder-only
              if (step === 0) activeComponent = { type: 'embedding' };
              else if (step <= decoderBlocks) activeComponent = { stack: 'decoder', layer: step - 1 };
              else activeComponent = { type: 'output' };
          }


          transformerObjects.forEach(obj => {
              const ud = obj.userData;
              let is_active = false;

              if (typeof activeComponent === 'string') { // Legacy or simple cases
                  if (activeComponent === 'embedding' && ud.type && ud.type.includes('embedding')) is_active = true;
                  if (activeComponent === 'output' && (ud.type === 'linear' || ud.type === 'softmax' || ud.type === 'pooler' || ud.type === 'classifier' || ud.type === 'lm_head')) is_active = true;
              } else if (typeof activeComponent === 'object') {
                  if (activeComponent.type === 'embedding') {
                      if (ud.type && ud.type.includes('embedding') && (!activeComponent.stack || ud.stack === activeComponent.stack)) is_active = true;
                  } else if (activeComponent.type === 'output') {
                      if (ud.type === 'linear' || ud.type === 'softmax' || ud.type === 'pooler' || ud.type === 'classifier' || ud.type === 'lm_head') is_active = true;
                  } else if (ud.stack === activeComponent.stack && ud.layer === activeComponent.layer) {
                      is_active = true;
                  }
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
      } else if (state.archType === 'operator') {
          const step = state.animationStep % getOperatorStepCount();
          resetOperatorVisuals();
          highlightOperatorLayer(step, OPERATOR_COLORS.highlight, 1.0);
          applyOperatorInteractionHighlights();
      } else if (state.archType === 'moe') {
          const activeExpertsData = state.animationStep?.activeExperts || [];
          const router_fnn = transformerObjects.find(o => o.userData.type === 'moe_router_fnn');
          const router_softmax = transformerObjects.find(o => o.userData.type === 'moe_router_softmax');
          const expert_blocks = transformerObjects.filter(o => o.userData.type === 'moe_expert');
          const gate_bars = transformerObjects.filter(o => o.userData.type === 'moe_gate_bar');
          const usage_bars = transformerObjects.filter(o => o.userData.type === 'moe_usage_bar');

          // Reset all visuals
          transformerObjects.forEach(obj => {
              if (obj.material && obj.material.emissive) obj.material.emissive.setHex(0);
          });

          // Animate router
          if (router_fnn) router_fnn.material.emissive.setHex(0xffff00);
          if (router_softmax) router_softmax.material.emissive.setHex(0xffff00);

          // Animate active experts across all layers
          activeExpertsData.forEach((layerActiveExperts, layerIndex) => {
              layerActiveExperts.forEach(expertId => {
                  // Highlight expert block
                  const expert_block = expert_blocks.find(e => e.userData.layer === layerIndex && e.userData.expert_id === expertId);
                  if (expert_block) expert_block.material.emissive.setHex(0x00ff00);


                  // Particle flow
                  if (router_softmax && expert_block && Math.random() < 0.15) {
                      const particle = new THREE.Mesh(new THREE.SphereGeometry(0.12, 6, 6), new THREE.MeshBasicMaterial({ color: 0x00ff88, transparent: true }));
                      particle.position.copy(router_softmax.position);
                      const velocity = expert_block.position.clone().sub(router_softmax.position).normalize().multiplyScalar(12);
                      particle.userData = { velocity, life: 1.2 };
                      particles.push(particle);
                      scene.add(particle);
                  }
              });
          });

          // Update gate value histograms
          state.architecture.moe.gateValues.forEach((layerGates, layerIndex) => {
              const maxGate = Math.max(...layerGates);
              layerGates.forEach((gateValue, expertIndex) => {
                  const bar = gate_bars.find(b => b.userData.layer === layerIndex && b.userData.expert === expertIndex);
                  if (bar) {
                      const targetHeight = Math.max(0.1, (gateValue / Math.max(maxGate, 0.1)) * 2);
                      bar.scale.y = THREE.MathUtils.lerp(bar.scale.y, targetHeight, 0.1);
                      bar.position.y = bar.userData.base_y + (bar.geometry.parameters.height * bar.scale.y) / 2;
                  }
              });
          });

          // Update usage histograms
          state.architecture.moe.expertUsage.forEach((layerUsage, layerIndex) => {
              const maxUsage = Math.max(1, ...layerUsage);
              layerUsage.forEach((usage, expertIndex) => {
                  const bar = usage_bars.find(b => b.userData.layer === layerIndex && b.userData.expert === expertIndex);
                  if (bar) {
                      const targetHeight = Math.max(0.1, (usage / maxUsage) * 3);
                      bar.scale.y = THREE.MathUtils.lerp(bar.scale.y, targetHeight, 0.1);
                      bar.position.y = bar.userData.base_y + (bar.geometry.parameters.height * bar.scale.y) / 2;
                  }
              });
          });
      }
    }

    renderer.render(scene, camera);
  }

  // --- INITIALIZATION ---
  window.addEventListener('resize', () => {
    renderer.setSize(window.innerWidth, window.innerHeight);
    setActiveCamera();
    updateGridVisibility();
  });

  setActiveCamera();
  applyViewModeClass();
  applyTheme();
  archTypeSelect.value = state.archType;
  createNetwork();
  updateUI();
  animate();
});
