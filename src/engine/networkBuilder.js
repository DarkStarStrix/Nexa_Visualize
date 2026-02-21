import * as THREE from 'three';

const LAYER_SPACING = 3.5;
const NEURON_SIZE = 0.15;
const BASE_MAX_CONNECTIONS = 1800;

const MODEL_CONNECTION_BUDGET = {
  CNN: 2400,
  Transformer: 2200,
  'Neural Operator': 2000,
  MoE: 1800,
  Autoencoder: 2000,
  MLP: 2200,
  Custom: BASE_MAX_CONNECTIONS
};

const MODEL_SPACING = {
  CNN: 4.6,
  Transformer: 5.2,
  'Neural Operator': 4.8,
  MoE: 4.5,
  Autoencoder: 4.4,
  MLP: 3.8,
  Custom: LAYER_SPACING
};

const MODEL_SCALE = {
  CNN: 1.9,
  Transformer: 2.2,
  'Neural Operator': 2.0,
  MoE: 1.9,
  Autoencoder: 1.95,
  MLP: 1.8,
  Custom: 1.8
};

export const calculateOptimalCameraDistance = (layers, selectedModel = 'Custom') => {
  if (!layers?.length) return 12;
  const spacing = MODEL_SPACING[selectedModel] || LAYER_SPACING;
  const networkWidth = layers.length * spacing;
  const networkHeight = Math.max(...layers.map(layer => layer.gridSize[1])) * 2;
  const networkDepth = Math.max(...layers.map(layer => layer.gridSize[2])) * 2;
  const maxDimension = Math.max(networkWidth, networkHeight, networkDepth);
  const scale = MODEL_SCALE[selectedModel] || 1.8;
  return Math.max(12, maxDimension * scale);
};

export const clearNetwork = ({ scene, neurons = [], connections = [] }) => {
  if (!scene) return;

  neurons.forEach(layer => {
    layer.forEach(neuron => {
      scene.remove(neuron);
      neuron.geometry?.dispose();
      neuron.material?.dispose();
    });
  });

  connections.forEach(conn => {
    scene.remove(conn);
    conn.geometry?.dispose();
    conn.material?.dispose();
  });
};

const getConnectionDensity = (fromCount, toCount) => {
  return Math.min(1.0, 50 / Math.max(1, fromCount * toCount));
};

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

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const resolveGridCoordinates = (index, gridX, gridY) => {
  const x = index % gridX;
  const y = Math.floor(index / gridX) % gridY;
  const z = Math.floor(index / (gridX * gridY));
  return { x, y, z };
};

const getLayerYOffset = (selectedModel, layerIndex, layerCount, layerName = '') => {
  const name = layerName.toLowerCase();
  if (selectedModel === 'Transformer') {
    if (name.includes('token')) return -1.8;
    if (name.includes('self-attention')) return layerIndex % 2 === 0 ? 1.3 : -0.6;
    if (name.includes('feed')) return 1.6;
    if (name.includes('output')) return 0.3;
  }

  if (selectedModel === 'Autoencoder') {
    const center = (layerCount - 1) / 2;
    const distance = Math.abs(layerIndex - center);
    return (layerIndex < center ? 1 : -1) * distance * 0.6;
  }

  if (selectedModel === 'Neural Operator') {
    return Math.sin(layerIndex * 0.8) * 0.9;
  }

  if (selectedModel === 'CNN') {
    if (name.includes('input')) return 1.2;
    if (name.includes('conv')) return 0.8 - layerIndex * 0.2;
    if (name.includes('dense')) return -0.3;
    if (name.includes('output')) return -0.8;
  }

  return 0;
};

const resolveNeuronOffset = ({
  selectedModel,
  layer,
  layerIndex,
  layerCount,
  neuronIndex
}) => {
  const layerName = layer?.name?.toLowerCase() || '';
  const layerYOffset = getLayerYOffset(selectedModel, layerIndex, layerCount, layer?.name);

  if (selectedModel === 'Transformer') {
    const radius = layerName.includes('attention') ? 1.45 : 1.05;
    const angle = (neuronIndex / Math.max(1, layer.neurons)) * Math.PI * 2;
    return {
      y: layerYOffset + Math.sin(angle) * radius,
      z: Math.cos(angle) * radius * 1.2
    };
  }

  if (selectedModel === 'Neural Operator') {
    const angle = (neuronIndex / Math.max(1, layer.neurons)) * Math.PI * 2;
    const spiral = (neuronIndex / Math.max(1, layer.neurons) - 0.5) * 1.4;
    return {
      y: layerYOffset + Math.sin(angle * 1.8) * 1.0,
      z: Math.cos(angle) * 1.5 + spiral * 0.45
    };
  }

  if (selectedModel === 'MoE') {
    if (layerName.includes('experts')) {
      const expertGroups = 4;
      const groupSize = Math.ceil(layer.neurons / expertGroups);
      const group = Math.floor(neuronIndex / groupSize);
      const localIndex = neuronIndex % groupSize;
      const localAngle = (localIndex / Math.max(1, groupSize)) * Math.PI * 2;
      const centerY = (group - (expertGroups - 1) / 2) * 1.15;
      const centerZ = group % 2 === 0 ? -0.65 : 0.65;
      return {
        y: layerYOffset + centerY + Math.sin(localAngle) * 0.3,
        z: centerZ + Math.cos(localAngle) * 0.3
      };
    }

    if (layerName.includes('gating')) {
      const angle = (neuronIndex / Math.max(1, layer.neurons)) * Math.PI * 2;
      return { y: layerYOffset + Math.sin(angle) * 0.5, z: Math.cos(angle) * 0.5 };
    }
  }

  if (selectedModel === 'Autoencoder') {
    const center = (layerCount - 1) / 2;
    const distance = Math.abs(layerIndex - center);
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

  if (selectedModel === 'CNN') {
    const mapLike = layerName.includes('input') || layerName.includes('conv');
    if (mapLike) {
      const cols = Math.ceil(Math.sqrt(layer.neurons));
      const rows = Math.ceil(layer.neurons / cols);
      const col = neuronIndex % cols;
      const row = Math.floor(neuronIndex / cols);
      const featureScale = clamp(2.8 - layerIndex * 0.35, 1.2, 3.0);
      return {
        y: layerYOffset + ((row - (rows - 1) / 2) / Math.max(1, rows - 1)) * featureScale,
        z: ((col - (cols - 1) / 2) / Math.max(1, cols - 1)) * featureScale
      };
    }

    const angle = (neuronIndex / Math.max(1, layer.neurons)) * Math.PI * 2;
    return {
      y: layerYOffset + Math.sin(angle) * 0.95,
      z: Math.cos(angle) * 0.95
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

const shouldConnect = ({
  selectedModel,
  fromLayer,
  toLayer,
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
  const fromName = fromLayer?.name?.toLowerCase() || '';
  const toName = toLayer?.name?.toLowerCase() || '';

  if (selectedModel === 'CNN') {
    const mappedCenter = (fromIndex / Math.max(1, fromCount - 1)) * (toCount - 1);
    const distance = Math.abs(toIndex - mappedCenter);
    const receptive = Math.max(1, Math.ceil(toCount / Math.max(2, fromCount)));
    const localProb = distance <= receptive ? 0.9 : distance <= receptive * 2 ? 0.25 : 0.03;
    return rng() < baseProbability * localProb;
  }

  if (selectedModel === 'Transformer') {
    const fromNorm = fromIndex / Math.max(1, fromCount - 1);
    const toNorm = toIndex / Math.max(1, toCount - 1);
    const bandProb = Math.abs(fromNorm - toNorm) < 0.2 ? 0.85 : 0.2;
    return rng() < baseProbability * bandProb;
  }

  if (selectedModel === 'MoE') {
    if (fromName.includes('gating') && toName.includes('experts')) {
      const expertGroups = 4;
      const groupSize = Math.ceil(toCount / expertGroups);
      const gateGroup = fromIndex % expertGroups;
      const expertGroup = Math.floor(toIndex / groupSize);
      return rng() < baseProbability * (gateGroup === expertGroup ? 0.95 : 0.04);
    }

    if (fromName.includes('experts') && toName.includes('router')) {
      return rng() < baseProbability * 0.7;
    }
  }

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

export const buildDenseNetwork = ({ scene, layers, random, selectedModel, maxConnections }) => {
  if (!scene) return { neurons: [], connections: [], stats: { neuronCount: 0, connectionCount: 0 } };
  const rng = typeof random === 'function' ? random : Math.random;
  const totalPossibleConnections = getTotalPossibleConnections(layers);
  const connectionBudget = resolveConnectionBudget({ selectedModel, maxConnections });
  const budgetDensity = Math.min(1, connectionBudget / Math.max(1, totalPossibleConnections));
  const layerSpacing = MODEL_SPACING[selectedModel] || LAYER_SPACING;

  const neurons = [];
  const connections = [];
  const neuronGeometry = new THREE.SphereGeometry(NEURON_SIZE, 8, 6);
  const addConnection = (fromNode, toNode, fromLayerIndex, toLayerIndex, fromIndex, toIndex, opacity = 0.15) => {
    if (connections.length >= connectionBudget) return;
    const points = [fromNode.position.clone(), toNode.position.clone()];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
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
    const layerNeurons = [];
    const layerX = layerIndex * layerSpacing - (layers.length - 1) * layerSpacing / 2;

    for (let neuronIndex = 0; neuronIndex < layer.neurons; neuronIndex++) {
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
      neuron.position.set(layerX, offsets.y, offsets.z);
      neuron.userData = {
        originalColor: layer.color,
        activation: 0,
        layerIndex,
        neuronIndex
      };
      scene.add(neuron);
      layerNeurons.push(neuron);
    }

    neurons.push(layerNeurons);
  });

  for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex++) {
    if (connections.length >= connectionBudget) break;
    const currentLayer = neurons[layerIndex];
    const nextLayer = neurons[layerIndex + 1];
    const baseDensity = getConnectionDensity(currentLayer.length, nextLayer.length);

    currentLayer.forEach((neuron1, idx1) => {
      nextLayer.forEach((neuron2, idx2) => {
        if (shouldConnect({
          selectedModel,
          fromLayer: layers[layerIndex],
          toLayer: layers[layerIndex + 1],
          fromIndex: idx1,
          toIndex: idx2,
          fromCount: currentLayer.length,
          toCount: nextLayer.length,
          baseDensity,
          budgetDensity,
          random: rng
        })) {
          addConnection(neuron1, neuron2, layerIndex, layerIndex + 1, idx1, idx2);
        }
      });
    });
  }

  if (selectedModel === 'Autoencoder') {
    const center = Math.floor(layers.length / 2);
    for (let left = 0; left < center - 1; left++) {
      if (connections.length >= connectionBudget) break;
      const right = layers.length - 1 - left;
      if (right <= left + 1) continue;
      const leftNeurons = neurons[left];
      const rightNeurons = neurons[right];
      leftNeurons.forEach((leftNeuron, idx) => {
        if (connections.length >= connectionBudget) return;
        const targetIdx = Math.floor((idx / Math.max(1, leftNeurons.length - 1)) * Math.max(0, rightNeurons.length - 1));
        const rightNeuron = rightNeurons[targetIdx];
        if (rightNeuron && rng() < 0.35) {
          addConnection(leftNeuron, rightNeuron, left, right, idx, targetIdx, 0.12);
        }
      });
    }
  }

  if (selectedModel === 'Transformer') {
    for (let i = 0; i < layers.length - 2; i++) {
      if (connections.length >= connectionBudget) break;
      const fromLayer = neurons[i];
      const toLayer = neurons[i + 2];
      fromLayer.forEach((fromNeuron, idx) => {
        if (connections.length >= connectionBudget) return;
        const targetIdx = Math.floor((idx / Math.max(1, fromLayer.length - 1)) * Math.max(0, toLayer.length - 1));
        const toNeuron = toLayer[targetIdx];
        if (toNeuron && rng() < 0.28) {
          addConnection(fromNeuron, toNeuron, i, i + 2, idx, targetIdx, 0.1);
        }
      });
    }
  }

  return {
    neurons,
    connections,
    stats: {
      neuronCount: neurons.reduce((sum, layer) => sum + layer.length, 0),
      connectionCount: connections.length,
      connectionBudget
    }
  };
};
