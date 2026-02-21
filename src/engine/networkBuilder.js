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

export const calculateOptimalCameraDistance = (layers) => {
  if (!layers?.length) return 12;
  const networkWidth = layers.length * 3;
  const networkHeight = Math.max(...layers.map(layer => layer.gridSize[1])) * 2;
  const networkDepth = Math.max(...layers.map(layer => layer.gridSize[2])) * 2;
  const maxDimension = Math.max(networkWidth, networkHeight, networkDepth);
  return Math.max(12, maxDimension * 1.8);
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

export const buildDenseNetwork = ({ scene, layers, random, selectedModel, maxConnections }) => {
  if (!scene) return { neurons: [], connections: [], stats: { neuronCount: 0, connectionCount: 0 } };
  const rng = typeof random === 'function' ? random : Math.random;
  const totalPossibleConnections = getTotalPossibleConnections(layers);
  const connectionBudget = resolveConnectionBudget({ selectedModel, maxConnections });
  const budgetDensity = Math.min(1, connectionBudget / Math.max(1, totalPossibleConnections));

  const neurons = [];
  const connections = [];
  const neuronGeometry = new THREE.SphereGeometry(NEURON_SIZE, 8, 6);

  layers.forEach((layer, layerIndex) => {
    const layerNeurons = [];
    const [gridX, gridY, gridZ] = layer.gridSize;
    const spacingX = gridX > 1 ? 2.5 / (gridX - 1) : 0;
    const spacingY = gridY > 1 ? 2.5 / (gridY - 1) : 0;
    const spacingZ = gridZ > 1 ? 1.5 / (gridZ - 1) : 0;

    let neuronIndex = 0;
    for (let z = 0; z < gridZ && neuronIndex < layer.neurons; z++) {
      for (let y = 0; y < gridY && neuronIndex < layer.neurons; y++) {
        for (let x = 0; x < gridX && neuronIndex < layer.neurons; x++) {
          const material = new THREE.MeshPhongMaterial({
            color: layer.color,
            transparent: true,
            opacity: 0.8,
            shininess: 100
          });
          const neuron = new THREE.Mesh(neuronGeometry, material);
          neuron.position.set(
            layerIndex * LAYER_SPACING - (layers.length - 1) * LAYER_SPACING / 2,
            (y * spacingY) - (gridY - 1) * spacingY / 2,
            (z * spacingZ) - (gridZ - 1) * spacingZ / 2 + (x * spacingX) - (gridX - 1) * spacingX / 2
          );
          neuron.userData = {
            originalColor: layer.color,
            activation: 0,
            layerIndex,
            neuronIndex
          };
          scene.add(neuron);
          layerNeurons.push(neuron);
          neuronIndex++;
        }
      }
    }

    neurons.push(layerNeurons);
  });

  for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex++) {
    const currentLayer = neurons[layerIndex];
    const nextLayer = neurons[layerIndex + 1];
    const baseDensity = getConnectionDensity(currentLayer.length, nextLayer.length);
    const connectionDensity = Math.min(1, Math.max(baseDensity * 0.6, budgetDensity));

    currentLayer.forEach((neuron1, idx1) => {
      nextLayer.forEach((neuron2, idx2) => {
        if (connections.length >= connectionBudget) return;
        if (rng() < connectionDensity) {
          const points = [neuron1.position.clone(), neuron2.position.clone()];
          const geometry = new THREE.BufferGeometry().setFromPoints(points);
          const material = new THREE.LineBasicMaterial({
            color: 0x666666,
            transparent: true,
            opacity: 0.15
          });
          const line = new THREE.Line(geometry, material);
          line.userData = {
            fromLayer: layerIndex,
            toLayer: layerIndex + 1,
            fromNeuron: idx1,
            toNeuron: idx2
          };
          scene.add(line);
          connections.push(line);
        }
      });
    });
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
