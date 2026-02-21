import * as THREE from 'three';

const LAYER_SPACING = 3.5;
const NEURON_SIZE = 0.15;
const BASE_MAX_CONNECTIONS = 1800;
const GRID_PLANE_Y = -8;
const GRID_CLEARANCE = 0.25;

const LEGACY_CAMERA_DISTANCE = {
  CNN: 50,
  Transformer: 50,
  GAN: 50,
  RNN: 30,
  LSTM: 30,
  GRU: 30,
  MoE: 24
};

const MODEL_CONNECTION_BUDGET = {
  CNN: 2000,
  Transformer: 2200,
  'Neural Operator': 2000,
  MoE: 1800,
  Autoencoder: 2000,
  MLP: 2200,
  RNN: 1600,
  LSTM: 1800,
  GRU: 1800,
  GAN: 1400,
  Custom: BASE_MAX_CONNECTIONS
};

const MODEL_SPACING = {
  CNN: 4.6,
  Transformer: 5.2,
  'Neural Operator': 4.8,
  MoE: 4.5,
  Autoencoder: 4.4,
  MLP: 3.8,
  RNN: 4.2,
  LSTM: 4.2,
  GRU: 4.2,
  GAN: 6.0,
  Custom: LAYER_SPACING
};

const MODEL_SCALE = {
  CNN: 1.9,
  Transformer: 2.2,
  'Neural Operator': 2.0,
  MoE: 1.9,
  Autoencoder: 1.95,
  MLP: 1.8,
  RNN: 2.0,
  LSTM: 2.0,
  GRU: 2.0,
  GAN: 2.2,
  Custom: 1.8
};

const LEGACY_MODELS = new Set(['CNN', 'Transformer', 'MoE', 'GAN', 'RNN', 'LSTM', 'GRU']);

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const disposeMaterial = (material) => {
  if (!material) return;
  if (Array.isArray(material)) {
    material.forEach(disposeMaterial);
    return;
  }
  material.map?.dispose?.();
  material.dispose?.();
};

const createLabelSprite = (text, color = '#f8fafc') => {
  if (!text || typeof document === 'undefined') return null;
  if (typeof navigator !== 'undefined' && /jsdom/i.test(navigator.userAgent || '')) return null;
  const canvas = document.createElement('canvas');
  let context;
  try {
    context = canvas.getContext('2d');
  } catch {
    context = null;
  }
  if (!context) return null;

  const fontSize = 18;
  const padding = 8;
  context.font = `600 ${fontSize}px sans-serif`;
  const width = Math.ceil(context.measureText(text).width + padding * 2);
  const height = fontSize + padding * 2;
  canvas.width = width;
  canvas.height = height;

  context.font = `600 ${fontSize}px sans-serif`;
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillStyle = 'rgba(15, 23, 42, 0.66)';
  context.fillRect(0, 0, width, height);
  context.strokeStyle = 'rgba(255, 255, 255, 0.24)';
  context.lineWidth = 2;
  context.strokeRect(1, 1, width - 2, height - 2);
  context.fillStyle = color;
  context.fillText(text, width / 2, height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
    depthTest: false
  });
  const sprite = new THREE.Sprite(material);
  const aspect = width / Math.max(1, height);
  sprite.scale.set(1.2 * aspect, 0.95, 1);
  sprite.userData = {
    isLabel: true
  };
  return sprite;
};

const getGeometryHeight = (geometry) => {
  if (!geometry || !geometry.parameters) return 1.2;
  const params = geometry.parameters;
  if (Number.isFinite(params.height)) return params.height;
  if (Number.isFinite(params.radius)) return params.radius * 2;
  if (Number.isFinite(params.radiusTop)) return Math.max(params.radiusTop, params.radiusBottom || 0) * 2;
  if (Number.isFinite(params.depth)) return params.depth;
  return 1.2;
};

const resolveCenterYAboveGrid = (centerY, geometryHeight) => {
  const minCenterY = GRID_PLANE_Y + geometryHeight / 2 + GRID_CLEARANCE;
  return Math.max(centerY, minCenterY);
};

export const calculateOptimalCameraDistance = (layers, selectedModel = 'Custom') => {
  if (LEGACY_CAMERA_DISTANCE[selectedModel]) {
    return LEGACY_CAMERA_DISTANCE[selectedModel];
  }

  if (!layers?.length) return 12;
  const spacing = MODEL_SPACING[selectedModel] || LAYER_SPACING;
  const networkWidth = layers.length * spacing;
  const networkHeight = Math.max(...layers.map((layer) => layer.gridSize[1])) * 2;
  const networkDepth = Math.max(...layers.map((layer) => layer.gridSize[2])) * 2;
  const maxDimension = Math.max(networkWidth, networkHeight, networkDepth);
  const scale = MODEL_SCALE[selectedModel] || 1.8;
  return Math.max(12, maxDimension * scale);
};

export const clearNetwork = ({ scene, neurons = [], connections = [], decorations = [] }) => {
  if (!scene) return;

  neurons.forEach((layer) => {
    layer.forEach((neuron) => {
      scene.remove(neuron);
      neuron.geometry?.dispose();
      neuron.material?.dispose();
    });
  });

  connections.forEach((conn) => {
    scene.remove(conn);
    conn.geometry?.dispose();
    disposeMaterial(conn.material);
  });

  decorations.forEach((item) => {
    scene.remove(item);
    item.geometry?.dispose?.();
    disposeMaterial(item.material);
  });
};

const getConnectionDensity = (fromCount, toCount) => Math.min(1.0, 50 / Math.max(1, fromCount * toCount));

const getTotalPossibleConnections = (layers) => {
  let total = 0;
  for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex++) {
    total += layers[layerIndex].neurons * layers[layerIndex + 1].neurons;
  }
  return total;
};

const resolveConnectionBudget = ({ selectedModel, maxConnections }) => {
  if (Number.isFinite(maxConnections) && maxConnections > 0) return maxConnections;
  return MODEL_CONNECTION_BUDGET[selectedModel] || BASE_MAX_CONNECTIONS;
};

const resolveGridCoordinates = (index, gridX, gridY) => {
  const x = index % gridX;
  const y = Math.floor(index / gridX) % gridY;
  const z = Math.floor(index / (gridX * gridY));
  return { x, y, z };
};

const resolveNeuronOffset = ({
  selectedModel,
  layer,
  layerIndex,
  layerCount,
  neuronIndex
}) => {
  if (selectedModel === 'Neural Operator') {
    const angle = (neuronIndex / Math.max(1, layer.neurons)) * Math.PI * 2;
    const spiral = (neuronIndex / Math.max(1, layer.neurons) - 0.5) * 1.4;
    return {
      y: Math.sin(layerIndex * 0.8) * 0.9 + Math.sin(angle * 1.8) * 1.0,
      z: Math.cos(angle) * 1.5 + spiral * 0.45
    };
  }

  if (selectedModel === 'Autoencoder') {
    const center = (layerCount - 1) / 2;
    const distance = Math.abs(layerIndex - center);
    const layerYOffset = (layerIndex < center ? 1 : -1) * distance * 0.6;
    const widthFactor = 0.7 + distance * 0.7;
    const cols = Math.ceil(Math.sqrt(layer.neurons));
    const rows = Math.ceil(layer.neurons / cols);
    const col = neuronIndex % cols;
    const row = Math.floor(neuronIndex / cols);
    return {
      y: layerYOffset + ((row - (rows - 1) / 2) / Math.max(1, rows - 1)) * widthFactor * 2,
      z: ((col - (cols - 1) / 2) / Math.max(1, cols - 1)) * widthFactor * 1.6
    };
  }

  const [gridX, gridY, gridZ] = layer.gridSize;
  const spacingX = gridX > 1 ? 2.5 / (gridX - 1) : 0;
  const spacingY = gridY > 1 ? 2.5 / (gridY - 1) : 0;
  const spacingZ = gridZ > 1 ? 1.5 / (gridZ - 1) : 0;
  const { x, y, z } = resolveGridCoordinates(neuronIndex, gridX, gridY);

  return {
    y: (y * spacingY) - (gridY - 1) * spacingY / 2,
    z: (z * spacingZ) - (gridZ - 1) * spacingZ / 2 + (x * spacingX) - (gridX - 1) * spacingX / 2
  };
};

const shouldConnectDense = ({
  selectedModel,
  fromIndex,
  toIndex,
  fromCount,
  toCount,
  baseDensity,
  budgetDensity,
  random
}) => {
  const rng = typeof random === 'function' ? random : Math.random;
  const baseProbability = Math.min(1, Math.max(baseDensity * 0.6, budgetDensity));

  if (selectedModel === 'Autoencoder') {
    const mirrored = Math.abs(fromIndex / Math.max(1, fromCount) - toIndex / Math.max(1, toCount));
    const mirrorProb = mirrored < 0.15 ? 0.9 : 0.3;
    return rng() < baseProbability * mirrorProb;
  }

  if (selectedModel === 'Neural Operator') {
    const phase = Math.abs(Math.sin((fromIndex + toIndex) * 0.3));
    return rng() < baseProbability * (0.35 + phase * 0.55);
  }

  return rng() < baseProbability;
};

const createLegacyContext = ({ scene, connectionBudget }) => {
  const neurons = [];
  const connections = [];
  const decorations = [];

  const addStage = (nodeSpecs) => {
    const layerIndex = neurons.length;
    const layerNodes = nodeSpecs.map((spec, neuronIndex) => {
      const material = new THREE.MeshPhongMaterial({
        color: spec.color,
        transparent: true,
        opacity: spec.opacity ?? 0.85,
        shininess: 70
      });
      const geometry = spec.geometry();
      const mesh = new THREE.Mesh(geometry, material);
      const geometryHeight = getGeometryHeight(geometry);
      const centerY = resolveCenterYAboveGrid(spec.y, geometryHeight);
      mesh.position.set(spec.x, centerY, spec.z);
      mesh.userData = {
        originalColor: spec.color,
        activation: 0,
        layerIndex,
        neuronIndex,
        name: spec.name || '',
        legacyType: spec.type || ''
      };
      scene.add(mesh);
      if (spec.name) {
        const label = createLabelSprite(spec.name, spec.labelColor || '#f8fafc');
        if (label) {
          label.position.set(spec.x, centerY + geometryHeight / 2 + 0.28, spec.z);
          scene.add(label);
          decorations.push(label);
        }
      }
      return mesh;
    });
    neurons.push(layerNodes);
    return layerIndex;
  };

  const connect = ({ fromLayer, fromNeuron, toLayer, toNeuron, opacity = 0.18 }) => {
    if (connections.length >= connectionBudget) return;
    const fromNode = neurons[fromLayer]?.[fromNeuron];
    const toNode = neurons[toLayer]?.[toNeuron];
    if (!fromNode || !toNode) return;

    const geometry = new THREE.BufferGeometry().setFromPoints([
      fromNode.position.clone(),
      toNode.position.clone()
    ]);
    const material = new THREE.LineBasicMaterial({
      color: 0x666666,
      transparent: true,
      opacity
    });
    const line = new THREE.Line(geometry, material);
    line.userData = { fromLayer, toLayer, fromNeuron, toNeuron };
    scene.add(line);
    connections.push(line);
  };

  const connectAll = ({ fromLayer, toLayer, opacity = 0.18 }) => {
    const fromNodes = neurons[fromLayer] || [];
    const toNodes = neurons[toLayer] || [];
    fromNodes.forEach((_, fromNeuron) => {
      toNodes.forEach((__, toNeuron) => {
        connect({ fromLayer, fromNeuron, toLayer, toNeuron, opacity });
      });
    });
  };

  return {
    neurons,
    connections,
    decorations,
    addStage,
    connect,
    connectAll,
    finalize: () => ({
      neurons,
      connections,
      decorations,
      stats: {
        neuronCount: neurons.reduce((sum, layer) => sum + layer.length, 0),
        connectionCount: connections.length,
        connectionBudget
      }
    })
  };
};

const buildLegacyCnnNetwork = ({ scene, connectionBudget }) => {
  const ctx = createLegacyContext({ scene, connectionBudget });
  const y = -8;

  const stageSpecs = [
    { name: 'Input (32x32)', type: 'input', geometry: () => new THREE.BoxGeometry(6, 6, 0.5), color: 0x3b82f6, x: 0, y, z: -15 },
    { name: 'C1 Conv', type: 'conv', geometry: () => new THREE.BoxGeometry(5.5, 5.5, 1), color: 0xa855f7, x: 0, y, z: -9 },
    { name: 'S2 Pool', type: 'pool', geometry: () => new THREE.BoxGeometry(5, 5, 1), color: 0x22d3ee, x: 0, y, z: -4 },
    { name: 'C3 Conv', type: 'conv', geometry: () => new THREE.BoxGeometry(4, 4, 1.5), color: 0xa855f7, x: 0, y, z: 2 },
    { name: 'S4 Pool', type: 'pool', geometry: () => new THREE.BoxGeometry(3.5, 3.5, 1.5), color: 0x22d3ee, x: 0, y, z: 7 },
    { name: 'C5 Conv', type: 'conv', geometry: () => new THREE.BoxGeometry(3, 3, 2), color: 0xa855f7, x: 0, y, z: 13 },
    { name: 'F6 FC', type: 'fc', geometry: () => new THREE.BoxGeometry(1, 1, 3), color: 0x22c55e, x: 0, y, z: 18 },
    { name: 'Output', type: 'output', geometry: () => new THREE.BoxGeometry(1, 1, 2), color: 0xef4444, x: 0, y, z: 22 }
  ];

  stageSpecs.forEach((spec) => ctx.addStage([spec]));
  for (let i = 0; i < stageSpecs.length - 1; i += 1) {
    ctx.connect({ fromLayer: i, fromNeuron: 0, toLayer: i + 1, toNeuron: 0, opacity: 0.2 });
  }

  return ctx.finalize();
};

const buildLegacyTransformerNetwork = ({ scene, layers, connectionBudget }) => {
  const ctx = createLegacyContext({ scene, connectionBudget });

  const inferredEncoderBlocks = clamp(
    layers?.filter((layer) => layer?.name?.toLowerCase().includes('self-attention')).length || 2,
    2,
    4
  );
  const encoderBlocks = inferredEncoderBlocks;
  const decoderBlocks = 2;

  const embeddingStage = ctx.addStage([
    { name: 'Input Embedding', type: 'embedding', geometry: () => new THREE.BoxGeometry(3, 1, 1), color: 0xffb6c1, x: -8, y: -6, z: 0 },
    { name: 'Output Embedding', type: 'embedding', geometry: () => new THREE.BoxGeometry(3, 1, 1), color: 0xffb6c1, x: 8, y: -6, z: 0 }
  ]);

  const encoderAttentionStages = [];
  const encoderFeedForwardStages = [];
  for (let i = 0; i < encoderBlocks; i += 1) {
    const baseY = -2 + i * 5;
    encoderAttentionStages.push(
      ctx.addStage([
        {
          name: `Encoder MHA ${i + 1}`,
          type: 'attention',
          geometry: () => new THREE.BoxGeometry(2.5, 1.5, 0.8),
          color: 0xffa500,
          x: -8,
          y: baseY,
          z: 0
        }
      ])
    );
    encoderFeedForwardStages.push(
      ctx.addStage([
        {
          name: `Encoder FF ${i + 1}`,
          type: 'ff',
          geometry: () => new THREE.BoxGeometry(2.5, 1.0, 0.8),
          color: 0x87ceeb,
          x: -8,
          y: baseY + 2.5,
          z: 0
        }
      ])
    );
  }

  const decoderMaskedStages = [];
  const decoderCrossStages = [];
  const decoderFeedForwardStages = [];
  for (let i = 0; i < decoderBlocks; i += 1) {
    const baseY = -2 + i * 7;
    decoderMaskedStages.push(
      ctx.addStage([
        {
          name: `Masked MHA ${i + 1}`,
          type: 'masked_attention',
          geometry: () => new THREE.BoxGeometry(2.5, 1.2, 0.8),
          color: 0x9932cc,
          x: 8,
          y: baseY,
          z: 0
        }
      ])
    );
    decoderCrossStages.push(
      ctx.addStage([
        {
          name: `Cross Attention ${i + 1}`,
          type: 'cross_attention',
          geometry: () => new THREE.BoxGeometry(2.5, 1.2, 0.8),
          color: 0xdc143c,
          x: 8,
          y: baseY + 2,
          z: 0
        }
      ])
    );
    decoderFeedForwardStages.push(
      ctx.addStage([
        {
          name: `Decoder FF ${i + 1}`,
          type: 'ff',
          geometry: () => new THREE.BoxGeometry(2.5, 1.0, 0.8),
          color: 0x87ceeb,
          x: 8,
          y: baseY + 4,
          z: 0
        }
      ])
    );
  }

  const linearStage = ctx.addStage([
    { name: 'Linear', type: 'linear', geometry: () => new THREE.BoxGeometry(2, 0.8, 0.8), color: 0x808080, x: 8, y: 16, z: 0 }
  ]);
  const softmaxStage = ctx.addStage([
    { name: 'Softmax', type: 'softmax', geometry: () => new THREE.SphereGeometry(0.6, 16, 12), color: 0x32cd32, x: 8, y: 19, z: 0 }
  ]);

  ctx.connect({ fromLayer: embeddingStage, fromNeuron: 0, toLayer: encoderAttentionStages[0], toNeuron: 0, opacity: 0.22 });
  ctx.connect({ fromLayer: embeddingStage, fromNeuron: 1, toLayer: decoderMaskedStages[0], toNeuron: 0, opacity: 0.22 });

  for (let i = 0; i < encoderBlocks; i += 1) {
    ctx.connect({ fromLayer: encoderAttentionStages[i], fromNeuron: 0, toLayer: encoderFeedForwardStages[i], toNeuron: 0, opacity: 0.2 });
    if (i < encoderBlocks - 1) {
      ctx.connect({ fromLayer: encoderFeedForwardStages[i], fromNeuron: 0, toLayer: encoderAttentionStages[i + 1], toNeuron: 0, opacity: 0.2 });
    }
  }

  for (let i = 0; i < decoderBlocks; i += 1) {
    ctx.connect({ fromLayer: decoderMaskedStages[i], fromNeuron: 0, toLayer: decoderCrossStages[i], toNeuron: 0, opacity: 0.2 });
    ctx.connect({ fromLayer: decoderCrossStages[i], fromNeuron: 0, toLayer: decoderFeedForwardStages[i], toNeuron: 0, opacity: 0.2 });
    if (i < decoderBlocks - 1) {
      ctx.connect({ fromLayer: decoderFeedForwardStages[i], fromNeuron: 0, toLayer: decoderMaskedStages[i + 1], toNeuron: 0, opacity: 0.2 });
    }
  }

  ctx.connect({
    fromLayer: decoderFeedForwardStages[decoderFeedForwardStages.length - 1],
    fromNeuron: 0,
    toLayer: linearStage,
    toNeuron: 0,
    opacity: 0.2
  });
  ctx.connect({ fromLayer: linearStage, fromNeuron: 0, toLayer: softmaxStage, toNeuron: 0, opacity: 0.2 });

  const crossCount = Math.min(encoderFeedForwardStages.length, decoderCrossStages.length);
  for (let i = 0; i < crossCount; i += 1) {
    ctx.connect({
      fromLayer: encoderFeedForwardStages[i],
      fromNeuron: 0,
      toLayer: decoderCrossStages[i],
      toNeuron: 0,
      opacity: 0.26
    });
  }

  return ctx.finalize();
};

const buildLegacyMoeNetwork = ({ scene, layers, connectionBudget }) => {
  const ctx = createLegacyContext({ scene, connectionBudget });
  const expertsLayer = (layers || []).find((layer) => (layer?.name || '').toLowerCase().includes('expert'));
  const expertCount = clamp(Math.round(expertsLayer?.neurons || 4), 2, 8);

  const gateStage = ctx.addStage([
    { name: 'Gating Network', type: 'gate', geometry: () => new THREE.SphereGeometry(1, 16, 12), color: 0xffd700, x: 0, y: 2, z: 0 }
  ]);

  const radius = 6;
  const expertSpecs = [];
  for (let i = 0; i < expertCount; i += 1) {
    const angle = (i / expertCount) * Math.PI * 2;
    expertSpecs.push({
      name: `Expert ${i + 1}`,
      type: 'expert',
      geometry: () => new THREE.CylinderGeometry(0.8, 0.8, 2, 8),
      color: 0x3b82f6,
      x: Math.cos(angle) * radius,
      y: 0,
      z: Math.sin(angle) * radius
    });
  }
  const expertStage = ctx.addStage(expertSpecs);

  ctx.connectAll({ fromLayer: gateStage, toLayer: expertStage, opacity: 0.22 });

  return ctx.finalize();
};

const resolveRecurrentSteps = (layers, fallback = 5) => {
  const layerMax = Math.max(...(layers || []).map((layer) => layer?.neurons || 0), fallback);
  return clamp(Math.round(layerMax / 2), 4, 8);
};

const buildLegacyRnnNetwork = ({ scene, layers, connectionBudget }) => {
  const ctx = createLegacyContext({ scene, connectionBudget });
  const steps = resolveRecurrentSteps(layers, 5);
  const stages = [];

  for (let t = 0; t < steps; t += 1) {
    const x = t * 4 - (steps - 1) * 2;
    stages.push(
      ctx.addStage([
        { name: `x${t}`, type: 'input', geometry: () => new THREE.SphereGeometry(0.5, 12, 8), color: 0x3b82f6, x, y: -4, z: 0 },
        { name: `h${t}`, type: 'hidden', geometry: () => new THREE.SphereGeometry(0.8, 12, 8), color: 0x22c55e, x, y: 0, z: 0 },
        { name: `y${t}`, type: 'output', geometry: () => new THREE.SphereGeometry(0.5, 12, 8), color: 0xef4444, x, y: 4, z: 0 }
      ])
    );
  }

  stages.forEach((stageIndex, t) => {
    ctx.connect({ fromLayer: stageIndex, fromNeuron: 0, toLayer: stageIndex, toNeuron: 1, opacity: 0.2 });
    ctx.connect({ fromLayer: stageIndex, fromNeuron: 1, toLayer: stageIndex, toNeuron: 2, opacity: 0.2 });
    if (t < stages.length - 1) {
      ctx.connect({ fromLayer: stageIndex, fromNeuron: 1, toLayer: stages[t + 1], toNeuron: 1, opacity: 0.26 });
    }
  });

  return ctx.finalize();
};

const buildLegacyLstmNetwork = ({ scene, layers, connectionBudget }) => {
  const ctx = createLegacyContext({ scene, connectionBudget });
  const steps = resolveRecurrentSteps(layers, 4);
  const stages = [];

  for (let t = 0; t < steps; t += 1) {
    const x = t * 4 - (steps - 1) * 2;
    stages.push(
      ctx.addStage([
        { name: `c${t}`, type: 'cell', geometry: () => new THREE.CylinderGeometry(0.35, 0.35, 1.8, 10), color: 0x2196f3, x, y: 0, z: 0 },
        { name: `h${t}`, type: 'hidden', geometry: () => new THREE.SphereGeometry(0.65, 12, 8), color: 0x4caf50, x, y: 2.5, z: 0 },
        { name: `f${t}`, type: 'forget', geometry: () => new THREE.BoxGeometry(0.6, 0.6, 0.3), color: 0xff5722, x: x - 0.8, y: 1.1, z: 0 },
        { name: `i${t}`, type: 'input_gate', geometry: () => new THREE.BoxGeometry(0.6, 0.6, 0.3), color: 0x9c27b0, x, y: 1.1, z: 0 },
        { name: `o${t}`, type: 'output_gate', geometry: () => new THREE.BoxGeometry(0.6, 0.6, 0.3), color: 0xff9800, x: x + 0.8, y: 1.1, z: 0 }
      ])
    );
  }

  stages.forEach((stageIndex, t) => {
    ctx.connect({ fromLayer: stageIndex, fromNeuron: 2, toLayer: stageIndex, toNeuron: 0, opacity: 0.2 });
    ctx.connect({ fromLayer: stageIndex, fromNeuron: 3, toLayer: stageIndex, toNeuron: 0, opacity: 0.2 });
    ctx.connect({ fromLayer: stageIndex, fromNeuron: 4, toLayer: stageIndex, toNeuron: 1, opacity: 0.2 });
    if (t < stages.length - 1) {
      ctx.connect({ fromLayer: stageIndex, fromNeuron: 0, toLayer: stages[t + 1], toNeuron: 0, opacity: 0.28 });
      ctx.connect({ fromLayer: stageIndex, fromNeuron: 1, toLayer: stages[t + 1], toNeuron: 1, opacity: 0.24 });
    }
  });

  return ctx.finalize();
};

const buildLegacyGruNetwork = ({ scene, layers, connectionBudget }) => {
  const ctx = createLegacyContext({ scene, connectionBudget });
  const steps = resolveRecurrentSteps(layers, 4);
  const stages = [];

  for (let t = 0; t < steps; t += 1) {
    const x = t * 4 - (steps - 1) * 2;
    stages.push(
      ctx.addStage([
        { name: `h${t}`, type: 'hidden', geometry: () => new THREE.SphereGeometry(0.8, 12, 8), color: 0x607d8b, x, y: 0, z: 0 },
        { name: `r${t}`, type: 'reset', geometry: () => new THREE.BoxGeometry(0.6, 0.6, 0.3), color: 0xf44336, x: x - 1.2, y: 0, z: 0 },
        { name: `z${t}`, type: 'update', geometry: () => new THREE.BoxGeometry(0.6, 0.6, 0.3), color: 0x2196f3, x, y: 1.2, z: 0 }
      ])
    );
  }

  stages.forEach((stageIndex, t) => {
    ctx.connect({ fromLayer: stageIndex, fromNeuron: 1, toLayer: stageIndex, toNeuron: 0, opacity: 0.2 });
    ctx.connect({ fromLayer: stageIndex, fromNeuron: 2, toLayer: stageIndex, toNeuron: 0, opacity: 0.2 });
    if (t < stages.length - 1) {
      ctx.connect({ fromLayer: stageIndex, fromNeuron: 0, toLayer: stages[t + 1], toNeuron: 0, opacity: 0.28 });
    }
  });

  return ctx.finalize();
};

const buildLegacyGanNetwork = ({ scene, connectionBudget }) => {
  const ctx = createLegacyContext({ scene, connectionBudget });

  const latentStage = ctx.addStage([
    { name: 'Latent Noise', type: 'latent', geometry: () => new THREE.SphereGeometry(0.7, 12, 8), color: 0x9c88ff, x: -12, y: 0, z: 0 }
  ]);
  const generatorStage = ctx.addStage([
    { name: 'Generator', type: 'generator', geometry: () => new THREE.ConeGeometry(2, 6, 8), color: 0x22c55e, x: -7, y: 0, z: 0 }
  ]);
  const sampleStage = ctx.addStage([
    { name: 'Fake Sample', type: 'fake', geometry: () => new THREE.BoxGeometry(2.5, 1.5, 1.5), color: 0x3b82f6, x: -1, y: -1.5, z: 0 },
    { name: 'Real Sample', type: 'real', geometry: () => new THREE.BoxGeometry(2.5, 1.5, 1.5), color: 0xf59e0b, x: -1, y: 1.5, z: 0 }
  ]);
  const discriminatorStage = ctx.addStage([
    { name: 'Discriminator', type: 'discriminator', geometry: () => new THREE.CylinderGeometry(2, 2, 4, 10), color: 0xef4444, x: 7, y: 0, z: 0 }
  ]);
  const decisionStage = ctx.addStage([
    { name: 'Real/Fake', type: 'decision', geometry: () => new THREE.SphereGeometry(0.8, 12, 8), color: 0xf8fafc, x: 12, y: 0, z: 0 }
  ]);

  ctx.connect({ fromLayer: latentStage, fromNeuron: 0, toLayer: generatorStage, toNeuron: 0, opacity: 0.24 });
  ctx.connect({ fromLayer: generatorStage, fromNeuron: 0, toLayer: sampleStage, toNeuron: 0, opacity: 0.24 });
  ctx.connectAll({ fromLayer: sampleStage, toLayer: discriminatorStage, opacity: 0.2 });
  ctx.connect({ fromLayer: discriminatorStage, fromNeuron: 0, toLayer: decisionStage, toNeuron: 0, opacity: 0.24 });

  return ctx.finalize();
};

const LEGACY_MODEL_BUILDERS = {
  CNN: buildLegacyCnnNetwork,
  Transformer: buildLegacyTransformerNetwork,
  MoE: buildLegacyMoeNetwork,
  RNN: buildLegacyRnnNetwork,
  LSTM: buildLegacyLstmNetwork,
  GRU: buildLegacyGruNetwork,
  GAN: buildLegacyGanNetwork
};

const buildDenseFallbackNetwork = ({ scene, layers, random, selectedModel, connectionBudget }) => {
  const rng = typeof random === 'function' ? random : Math.random;
  const totalPossibleConnections = getTotalPossibleConnections(layers);
  const budgetDensity = Math.min(1, connectionBudget / Math.max(1, totalPossibleConnections));
  const layerSpacing = MODEL_SPACING[selectedModel] || LAYER_SPACING;

  const neurons = [];
  const connections = [];
  const decorations = [];
  const neuronGeometry = new THREE.SphereGeometry(NEURON_SIZE, 8, 6);

  const addConnection = (fromNode, toNode, fromLayerIndex, toLayerIndex, fromIndex, toIndex, opacity = 0.15) => {
    if (connections.length >= connectionBudget) return;
    const geometry = new THREE.BufferGeometry().setFromPoints([
      fromNode.position.clone(),
      toNode.position.clone()
    ]);
    const material = new THREE.LineBasicMaterial({
      color: 0x666666,
      transparent: true,
      opacity
    });
    const line = new THREE.Line(geometry, material);
    line.userData = {
      fromLayer: fromLayerIndex,
      toLayer: toLayerIndex,
      fromNeuron: fromIndex,
      toNeuron: toIndex
    };
    scene.add(line);
    connections.push(line);
  };

  layers.forEach((layer, layerIndex) => {
    const layerNodes = [];
    const layerX = layerIndex * layerSpacing - (layers.length - 1) * layerSpacing / 2;

    for (let neuronIndex = 0; neuronIndex < layer.neurons; neuronIndex += 1) {
      const material = new THREE.MeshPhongMaterial({
        color: layer.color,
        transparent: true,
        opacity: 0.8,
        shininess: 100
      });
      const neuron = new THREE.Mesh(neuronGeometry, material);
      const offsets = resolveNeuronOffset({
        selectedModel,
        layer,
        layerIndex,
        layerCount: layers.length,
        neuronIndex
      });
      const safeY = Math.max(offsets.y, GRID_PLANE_Y + NEURON_SIZE + GRID_CLEARANCE);
      neuron.position.set(layerX, safeY, offsets.z);
      neuron.userData = {
        originalColor: layer.color,
        activation: 0,
        layerIndex,
        neuronIndex
      };
      scene.add(neuron);
      layerNodes.push(neuron);
    }

    neurons.push(layerNodes);
    if (layerNodes.length > 0 && layer.name) {
      const topY = Math.max(...layerNodes.map((node) => node.position.y));
      const label = createLabelSprite(layer.name, '#cbd5e1');
      if (label) {
        label.position.set(layerX, topY + 0.5, 0);
        scene.add(label);
        decorations.push(label);
      }
    }
  });

  for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex += 1) {
    if (connections.length >= connectionBudget) break;
    const currentLayer = neurons[layerIndex];
    const nextLayer = neurons[layerIndex + 1];
    const baseDensity = getConnectionDensity(currentLayer.length, nextLayer.length);

    currentLayer.forEach((fromNeuron, fromIndex) => {
      nextLayer.forEach((toNeuron, toIndex) => {
        if (shouldConnectDense({
          selectedModel,
          fromIndex,
          toIndex,
          fromCount: currentLayer.length,
          toCount: nextLayer.length,
          baseDensity,
          budgetDensity,
          random: rng
        })) {
          addConnection(fromNeuron, toNeuron, layerIndex, layerIndex + 1, fromIndex, toIndex);
        }
      });
    });
  }

  if (selectedModel === 'Autoencoder') {
    const center = Math.floor(layers.length / 2);
    for (let left = 0; left < center - 1; left += 1) {
      if (connections.length >= connectionBudget) break;
      const right = layers.length - 1 - left;
      if (right <= left + 1) continue;

      const leftLayer = neurons[left];
      const rightLayer = neurons[right];
      leftLayer.forEach((leftNeuron, idx) => {
        if (connections.length >= connectionBudget) return;
        const rightIndex = Math.floor((idx / Math.max(1, leftLayer.length - 1)) * Math.max(0, rightLayer.length - 1));
        const rightNeuron = rightLayer[rightIndex];
        if (rightNeuron && rng() < 0.35) {
          addConnection(leftNeuron, rightNeuron, left, right, idx, rightIndex, 0.12);
        }
      });
    }
  }

  return {
    neurons,
    connections,
    decorations,
    stats: {
      neuronCount: neurons.reduce((sum, layer) => sum + layer.length, 0),
      connectionCount: connections.length,
      connectionBudget
    }
  };
};

export const buildDenseNetwork = ({ scene, layers, random, selectedModel, maxConnections }) => {
  if (!scene) {
    return { neurons: [], connections: [], stats: { neuronCount: 0, connectionCount: 0 } };
  }

  const safeLayers = Array.isArray(layers) ? layers : [];
  const connectionBudget = resolveConnectionBudget({ selectedModel, maxConnections });

  if (LEGACY_MODELS.has(selectedModel) && LEGACY_MODEL_BUILDERS[selectedModel]) {
    return LEGACY_MODEL_BUILDERS[selectedModel]({
      scene,
      layers: safeLayers,
      random,
      connectionBudget
    });
  }

  return buildDenseFallbackNetwork({
    scene,
    layers: safeLayers,
    random,
    selectedModel,
    connectionBudget
  });
};
