import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import * as THREE from 'three';
import {
  DEFAULT_ARCHITECTURE,
  DEFAULT_CAMERA_STATE,
  DEFAULT_SESSION_STATE,
  DEFAULT_TRAINING_PARAMS,
  generateSessionId,
  parseSessionPayload,
  serializeSession
} from './domain/sessionSchema';
import { buildDenseNetwork, calculateOptimalCameraDistance, clearNetwork } from './engine/networkBuilder';
import { createSeededRng, sanitizeSeed } from './engine/random';
import { computeTrainingTick } from './engine/trainingSimulation';
import { createSceneManager } from './engine/sceneManager';
import { ASSISTANT_GOALS, DATASET_PRESETS, MODEL_COMPARISON_BIAS, SCENARIO_PRESETS } from './content/enhancementContent';

const LOCAL_DRAFT_KEY = 'nexa.visualize.session.draft';
const SHARED_PREFIX = 'nexa.visualize.session.shared.';
const EXPERIMENT_RUNS_KEY = 'nexa.visualize.experiment.runs';
const STORY_KEY = 'nexa.visualize.story.';
const COLLAB_ANNOTATIONS_KEY = 'nexa.visualize.collab.annotations';

const MODEL_PRESETS = {
  Custom: DEFAULT_ARCHITECTURE,
  MLP: DEFAULT_ARCHITECTURE,
  CNN: [
    { neurons: 16, activation: 'Linear', name: 'Input Image' },
    { neurons: 24, activation: 'ReLU', name: 'Conv Block 1' },
    { neurons: 20, activation: 'ReLU', name: 'Conv Block 2' },
    { neurons: 12, activation: 'ReLU', name: 'Dense' },
    { neurons: 4, activation: 'Softmax', name: 'Output' }
  ],
  Transformer: [
    { neurons: 12, activation: 'Linear', name: 'Token Input' },
    { neurons: 18, activation: 'ReLU', name: 'Self-Attention 1' },
    { neurons: 18, activation: 'ReLU', name: 'Self-Attention 2' },
    { neurons: 14, activation: 'ReLU', name: 'Feed Forward' },
    { neurons: 6, activation: 'Softmax', name: 'Output Head' }
  ],
  'Neural Operator': [
    { neurons: 10, activation: 'Linear', name: 'Input Field' },
    { neurons: 16, activation: 'ReLU', name: 'Projection' },
    { neurons: 20, activation: 'ReLU', name: 'Spectral Block 1' },
    { neurons: 20, activation: 'ReLU', name: 'Spectral Block 2' },
    { neurons: 10, activation: 'Linear', name: 'Output Field' }
  ],
  MoE: [
    { neurons: 10, activation: 'Linear', name: 'Token Input' },
    { neurons: 8, activation: 'ReLU', name: 'Gating' },
    { neurons: 24, activation: 'ReLU', name: 'Experts' },
    { neurons: 12, activation: 'ReLU', name: 'Router Merge' },
    { neurons: 5, activation: 'Softmax', name: 'Output' }
  ],
  Autoencoder: [
    { neurons: 14, activation: 'Linear', name: 'Input' },
    { neurons: 10, activation: 'ReLU', name: 'Encoder 1' },
    { neurons: 6, activation: 'ReLU', name: 'Latent' },
    { neurons: 10, activation: 'ReLU', name: 'Decoder 1' },
    { neurons: 14, activation: 'Linear', name: 'Reconstruction' }
  ]
};


const NeuralNetwork3D = ({ sessionId, onSessionRouteChange }) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const animationRef = useRef(null);
  const neuronsRef = useRef([]);
  const connectionsRef = useRef([]);
  const animationProgressRef = useRef(0);
  const optimizationBallRef = useRef(null);
  const cameraDistanceRef = useRef(DEFAULT_CAMERA_STATE.distance);
  const lossRef = useRef(1.0);
  const accuracyRef = useRef(0.1);
  const layersLengthRef = useRef(DEFAULT_ARCHITECTURE.length);
  const createNetworkRef = useRef(null);
  const cameraModeRef = useRef(DEFAULT_SESSION_STATE.cameraMode);
  const isTrainingRef = useRef(false);
  const isTrainingCompleteRef = useRef(false);
  const animationSpeedRef = useRef(DEFAULT_SESSION_STATE.uiState.animationSpeed);
  const cameraStateRef = useRef(DEFAULT_CAMERA_STATE);
  const trainingRngRef = useRef(createSeededRng(DEFAULT_SESSION_STATE.simulation.seed));
  const batchRef = useRef(0);
  const epochRef = useRef(0);

  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [batch, setBatch] = useState(0);
  const [loss, setLoss] = useState(1.0);
  const [accuracy, setAccuracy] = useState(0.1);
  const [isTrainingComplete, setIsTrainingComplete] = useState(false);
  const [activePanel, setActivePanel] = useState(DEFAULT_SESSION_STATE.uiState.activePanel);
  const [cameraMode, setCameraMode] = useState(DEFAULT_SESSION_STATE.cameraMode);
  const [animationSpeed, setAnimationSpeed] = useState(DEFAULT_SESSION_STATE.uiState.animationSpeed);
  const [simulationSeed, setSimulationSeed] = useState(DEFAULT_SESSION_STATE.simulation.seed);
  const [selectedModel, setSelectedModel] = useState(DEFAULT_SESSION_STATE.selectedModel);
  const [datasetName, setDatasetName] = useState('Spiral');
  const [compareModel, setCompareModel] = useState('Transformer');
  const [compareSnapshot, setCompareSnapshot] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [timelineIndex, setTimelineIndex] = useState(-1);
  const [selectedScenario, setSelectedScenario] = useState('stable');
  const [probeLayerIndex, setProbeLayerIndex] = useState(0);
  const [predictionInput, setPredictionInput] = useState({ x1: 0.3, x2: 0.6, x3: 0.4 });
  const [assistantGoal, setAssistantGoal] = useState(ASSISTANT_GOALS[0]);
  const [assistantMessage, setAssistantMessage] = useState('');
  const [experimentRuns, setExperimentRuns] = useState([]);
  const [experimentNote, setExperimentNote] = useState('');
  const [storySteps, setStorySteps] = useState([]);
  const [storyLink, setStoryLink] = useState('');
  const [collabMode, setCollabMode] = useState(false);
  const [annotationDraft, setAnnotationDraft] = useState('');
  const [annotations, setAnnotations] = useState([]);
  const [presentationMode, setPresentationMode] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');

  // Architecture customization
  const [architecture, setArchitecture] = useState(DEFAULT_ARCHITECTURE);

  // Training parameters
  const [trainingParams, setTrainingParams] = useState(DEFAULT_TRAINING_PARAMS);

  // Manual camera state
  const [cameraState, setCameraState] = useState(DEFAULT_CAMERA_STATE);

  useEffect(() => {
    cameraDistanceRef.current = cameraState.distance;
  }, [cameraState.distance]);

  useEffect(() => {
    lossRef.current = loss;
  }, [loss]);

  useEffect(() => {
    accuracyRef.current = accuracy;
  }, [accuracy]);

  useEffect(() => {
    cameraModeRef.current = cameraMode;
  }, [cameraMode]);

  useEffect(() => {
    isTrainingRef.current = isTraining;
  }, [isTraining]);

  useEffect(() => {
    isTrainingCompleteRef.current = isTrainingComplete;
  }, [isTrainingComplete]);

  useEffect(() => {
    animationSpeedRef.current = animationSpeed;
  }, [animationSpeed]);

  useEffect(() => {
    cameraStateRef.current = cameraState;
  }, [cameraState]);

  useEffect(() => {
    trainingRngRef.current = createSeededRng(simulationSeed);
  }, [simulationSeed, datasetName]);

  useEffect(() => {
    batchRef.current = batch;
  }, [batch]);

  useEffect(() => {
    epochRef.current = epoch;
  }, [epoch]);

  useEffect(() => {
    try {
      const storedRuns = localStorage.getItem(EXPERIMENT_RUNS_KEY);
      if (storedRuns) setExperimentRuns(JSON.parse(storedRuns));
      const storedAnnotations = localStorage.getItem(COLLAB_ANNOTATIONS_KEY);
      if (storedAnnotations) setAnnotations(JSON.parse(storedAnnotations));
    } catch {
      setStatusMessage('Saved enhancement data could not be loaded.');
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(EXPERIMENT_RUNS_KEY, JSON.stringify(experimentRuns));
  }, [experimentRuns]);

  useEffect(() => {
    localStorage.setItem(COLLAB_ANNOTATIONS_KEY, JSON.stringify(annotations));
  }, [annotations]);

  useEffect(() => {
    setProbeLayerIndex((prev) => Math.max(0, Math.min(prev, architecture.length - 1)));
  }, [architecture.length]);

  // Generate optimized layer configuration
  const layers = useMemo(() => {
    const colors = [0x4CAF50, 0x2196F3, 0x9C27B0, 0xFF9800, 0xF44336, 0x607D8B, 0x795548, 0x009688, 0xFFEB3B];
    const maxNeurons = Math.max(...architecture.map(layer => layer.neurons));

    return architecture.map((layer, idx) => {
      const neurons = layer.neurons;

      // Better grid arrangement for larger networks
      let gridSize;
      if (neurons <= 4) {
        gridSize = [Math.min(neurons, 2), Math.ceil(neurons / 2), 1];
      } else if (neurons <= 9) {
        gridSize = [3, 3, Math.ceil(neurons / 9)];
      } else if (neurons <= 16) {
        gridSize = [4, 4, Math.ceil(neurons / 16)];
      } else if (neurons <= 36) {
        gridSize = [6, 6, Math.ceil(neurons / 36)];
      } else {
        gridSize = [8, 8, Math.ceil(neurons / 64)];
      }

      return {
        ...layer,
        color: colors[idx % colors.length],
        gridSize,
        maxNeurons
      };
    });
  }, [architecture]);

  const activeDataset = DATASET_PRESETS[datasetName] || DATASET_PRESETS.Spiral;

  useEffect(() => {
    layersLengthRef.current = layers.length;
  }, [layers.length]);

  // Create dense network connections like in the provided image
  const createNetwork = useCallback(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    clearNetwork({ scene, neurons: neuronsRef.current, connections: connectionsRef.current });
    const seededNetworkRng = createSeededRng(simulationSeed + layers.length * 131);
    const { neurons, connections } = buildDenseNetwork({
      scene,
      layers,
      random: seededNetworkRng,
      selectedModel
    });

    neuronsRef.current = neurons;
    connectionsRef.current = connections;

    const optimalDistance = calculateOptimalCameraDistance(layers, selectedModel);
    setCameraState(prev => ({
      ...prev,
      targetDistance: optimalDistance,
      distance: optimalDistance
    }));
  }, [layers, selectedModel, simulationSeed]);

  useEffect(() => {
    createNetworkRef.current = createNetwork;
  }, [createNetwork]);

  // Create loss landscape for optimization view
  const createLossLandscape = useCallback((showLandscape) => {
    const scene = sceneRef.current;
    if (!scene) return;

    const existingLandscape = scene.getObjectByName('lossLandscape');
    if (existingLandscape) scene.remove(existingLandscape);

    if (!showLandscape) return;

    // Create 3D loss surface
    const landscapeGroup = new THREE.Group();
    landscapeGroup.name = 'lossLandscape';

    const size = 20;
    const resolution = 30;
    const geometry = new THREE.PlaneGeometry(size, size, resolution, resolution);

    // Generate loss landscape height map
    const vertices = geometry.attributes.position.array;
    for (let i = 0; i < vertices.length; i += 3) {
      const x = vertices[i];
      const z = vertices[i + 2];
      // Create realistic loss landscape with local minima
      const height = 2 * Math.exp(-0.1 * (x*x + z*z)) +
                   0.5 * Math.sin(x * 0.5) * Math.cos(z * 0.5) +
                   0.2 * Math.sin((x + z) * 0.35);
      vertices[i + 1] = height;
    }
    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
      color: 0x6366f1,
      transparent: true,
      opacity: 0.7,
      wireframe: false
    });

    const landscape = new THREE.Mesh(geometry, material);
    landscape.rotation.x = -Math.PI / 2;
    landscape.position.y = -8;
    landscapeGroup.add(landscape);

    // Add optimization ball
    const ballGeometry = new THREE.SphereGeometry(0.2, 16, 12);
    const ballMaterial = new THREE.MeshPhongMaterial({
      color: 0xff4444,
      emissive: 0x440000
    });
    const ball = new THREE.Mesh(ballGeometry, ballMaterial);
    ball.position.set(5, -6, 5); // Start position
    optimizationBallRef.current = ball;
    landscapeGroup.add(ball);

    scene.add(landscapeGroup);
  }, []);

  useEffect(() => {
    if (!mountRef.current) return;
    const mountNode = mountRef.current;
    const sceneManager = createSceneManager(mountNode);
    const { scene, camera, renderer } = sceneManager;
    sceneRef.current = scene;
    cameraRef.current = camera;
    rendererRef.current = renderer;

    createNetworkRef.current?.();

    // Mouse controls
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };

    const onMouseDown = (event) => {
      if (cameraModeRef.current === 'manual') {
        isDragging = true;
        previousMousePosition = { x: event.clientX, y: event.clientY };
      }
    };

    const onMouseMove = (event) => {
      if (isDragging && cameraModeRef.current === 'manual') {
        const deltaX = (event.clientX - previousMousePosition.x) * 0.008;
        const deltaY = (event.clientY - previousMousePosition.y) * 0.008;

        setCameraState(prev => {
          const next = {
            ...prev,
            targetAngleX: prev.targetAngleX + deltaX,
            targetAngleY: Math.max(-Math.PI / 2.2, Math.min(Math.PI / 2.2, prev.targetAngleY + deltaY))
          };
          cameraStateRef.current = next;
          return next;
        });

        previousMousePosition = { x: event.clientX, y: event.clientY };
      }
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    const onWheel = (event) => {
      if (cameraModeRef.current === 'manual') {
        event.preventDefault();
        setCameraState(prev => {
          const next = {
            ...prev,
            targetDistance: Math.max(5, Math.min(80, prev.targetDistance + event.deltaY * 0.05))
          };
          cameraStateRef.current = next;
          return next;
        });
      }
    };

    mountNode.addEventListener('mousedown', onMouseDown);
    mountNode.addEventListener('mousemove', onMouseMove);
    mountNode.addEventListener('mouseup', onMouseUp);
    window.addEventListener('mouseup', onMouseUp);
    mountNode.addEventListener('wheel', onWheel, { passive: false });

    // Main animation loop - SIMPLIFIED AND RELIABLE
    const animate = (currentTime) => {
      const deltaTime = currentTime * 0.001;

      // Camera positioning
      if (cameraModeRef.current === 'auto') {
        const time = deltaTime * 0.3;
        const distance = cameraDistanceRef.current;
        camera.position.x = Math.cos(time) * distance;
        camera.position.z = Math.sin(time) * distance;
        camera.position.y = distance * 0.4;
        camera.lookAt(0, 0, 0);
      } else {
        // Smooth manual camera using refs to avoid re-rendering every frame.
        const prev = cameraStateRef.current;
        const lerpFactor = 0.08;
        const newDistance = THREE.MathUtils.lerp(prev.distance, prev.targetDistance, lerpFactor);
        const newAngleX = THREE.MathUtils.lerp(prev.angleX, prev.targetAngleX, lerpFactor);
        const newAngleY = THREE.MathUtils.lerp(prev.angleY, prev.targetAngleY, lerpFactor);

        camera.position.x = Math.cos(newAngleX) * Math.cos(newAngleY) * newDistance;
        camera.position.z = Math.sin(newAngleX) * Math.cos(newAngleY) * newDistance;
        camera.position.y = Math.sin(newAngleY) * newDistance;
        camera.lookAt(0, 0, 0);

        cameraStateRef.current = {
          ...prev,
          distance: newDistance,
          angleX: newAngleX,
          angleY: newAngleY
        };
        cameraDistanceRef.current = newDistance;
      }

      // FIXED TRAINING ANIMATION - RELIABLE AND CONSISTENT
      if (isTrainingRef.current && !isTrainingCompleteRef.current) {
        const neurons = neuronsRef.current;
        const connections = connectionsRef.current;

        // Continuous animation progress
        animationProgressRef.current = (deltaTime * animationSpeedRef.current * 0.5) % (layersLengthRef.current * 2 + 2);
        const progress = animationProgressRef.current;

        // Reset all visuals
        neurons.forEach(layer => {
          layer.forEach(neuron => {
            neuron.material.emissive.setHex(0x000000);
            neuron.material.opacity = 0.8;
            neuron.scale.setScalar(1);
          });
        });

        connections.forEach(connection => {
          connection.material.color.setHex(0x666666);
          connection.material.opacity = 0.15;
        });

        // FORWARD PASS (Green) - First half of cycle
        if (progress < layersLengthRef.current) {
          for (let i = 0; i < layersLengthRef.current; i++) {
            const layerStart = i;
            const layerProgress = Math.max(0, Math.min(1, progress - layerStart));

            if (layerProgress > 0) {
              const intensity = Math.sin(Math.PI * layerProgress) * 0.8 + 0.2;

              // Light up neurons
              neurons[i]?.forEach(neuron => {
                neuron.material.emissive.setHex(0x00cc00);
                neuron.material.emissive.multiplyScalar(intensity);
                neuron.scale.setScalar(1 + intensity * 0.5);
              });

              // Light up outgoing connections
              connections.forEach(conn => {
                if (conn.userData.fromLayer === i) {
                  conn.material.color.setHex(0x00ff00);
                  conn.material.opacity = 0.2 + intensity * 0.6;
                }
              });
            }
          }
        }
        // BACKWARD PASS (Red) - Second half of cycle
        else if (progress < layersLengthRef.current * 2) {
          const backProgress = progress - layersLengthRef.current;
          for (let i = layersLengthRef.current - 1; i >= 0; i--) {
            const reverseIdx = layersLengthRef.current - 1 - i;
            const layerProgress = Math.max(0, Math.min(1, backProgress - reverseIdx));

            if (layerProgress > 0) {
              const intensity = Math.sin(Math.PI * layerProgress) * 0.8 + 0.2;

              // Light up neurons
              neurons[i]?.forEach(neuron => {
                neuron.material.emissive.setHex(0xcc0000);
                neuron.material.emissive.multiplyScalar(intensity);
                neuron.scale.setScalar(1 + intensity * 0.5);
              });

              // Light up incoming connections
              connections.forEach(conn => {
                if (conn.userData.toLayer === i) {
                  conn.material.color.setHex(0xff0000);
                  conn.material.opacity = 0.2 + intensity * 0.6;
                }
              });
            }
          }
        }
        // WEIGHT UPDATE (Blue) - Brief flash
        else {
          const updateIntensity = Math.sin((progress - layersLengthRef.current * 2) * Math.PI * 4) * 0.5 + 0.5;
          neurons.forEach(layer => {
            layer.forEach(neuron => {
              neuron.material.emissive.setHex(0x0099ff);
              neuron.material.emissive.multiplyScalar(updateIntensity * 0.8);
            });
          });
          connections.forEach(conn => {
            conn.material.color.setHex(0x0099ff);
            conn.material.opacity = 0.3 + updateIntensity * 0.5;
          });
        }
      }
      // Update optimization ball
      if (optimizationBallRef.current && isTrainingRef.current) {
        const ball = optimizationBallRef.current;
        const time = deltaTime * 2;
        ball.position.x = Math.cos(time) * (3 - lossRef.current * 2.5);
        ball.position.z = Math.sin(time) * (3 - lossRef.current * 2.5);
        ball.position.y = -6 + lossRef.current * 2;
      }

      renderer.render(scene, camera);
      animationRef.current = requestAnimationFrame(animate);
    };

    animate(0);

    // Handle resize
    const handleResize = () => sceneManager.resize();

    window.addEventListener('resize', handleResize);

  return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (mountNode && renderer.domElement) {
        mountNode.removeEventListener('mousedown', onMouseDown);
        mountNode.removeEventListener('mousemove', onMouseMove);
        mountNode.removeEventListener('mouseup', onMouseUp);
        window.removeEventListener('mouseup', onMouseUp);
        mountNode.removeEventListener('wheel', onWheel);
      }
      window.removeEventListener('resize', handleResize);

      // Cleanup
      clearNetwork({ scene, neurons: neuronsRef.current, connections: connectionsRef.current });

      sceneManager.dispose();
    };
  }, []);

  // Recreate network when architecture changes
  useEffect(() => {
    if (sceneRef.current) {
      createNetworkRef.current?.();
    }
  }, [createNetwork]);

  // Create loss landscape when optimization panel is active
  useEffect(() => {
    createLossLandscape(activePanel === 'optimization');
  }, [createLossLandscape, activePanel]);

  // Training simulation - STOPS AT 90%
  useEffect(() => {
    if (!isTraining || isTrainingComplete) return;

    const interval = setInterval(() => {
      const nextBatch = batchRef.current + 1;
      batchRef.current = nextBatch;
      setBatch(nextBatch);
      if (nextBatch % 10 === 0) {
        epochRef.current += 1;
        setEpoch(epochRef.current);
      }

      const datasetDifficulty = activeDataset.difficulty;

      const tick = computeTrainingTick({
        previousLoss: lossRef.current,
        previousAccuracy: accuracyRef.current,
        learningRate: trainingParams.learningRate / datasetDifficulty,
        random: trainingRngRef.current
      });

      lossRef.current = tick.loss;
      accuracyRef.current = tick.accuracy;
      setLoss(tick.loss);
      setAccuracy(tick.accuracy);
      setTrainingHistory((prev) => {
        const next = [
          ...prev,
          {
            epoch: epochRef.current,
            batch: nextBatch,
            loss: tick.loss,
            accuracy: tick.accuracy
          }
        ];
        return next.slice(-300);
      });

      // STOP AT 90% and flash blue
      if (tick.complete && !isTrainingCompleteRef.current) {
        setIsTrainingComplete(true);
        setIsTraining(false);

        setTimeout(() => {
          neuronsRef.current.forEach(layer => {
            layer.forEach(neuron => {
              neuron.material.emissive.setHex(0x0099ff);
              neuron.material.emissive.multiplyScalar(1);
            });
          });
          connectionsRef.current.forEach(conn => {
            conn.material.color.setHex(0x0099ff);
            conn.material.opacity = 0.8;
          });
        }, 100);
      }
    }, 200 / animationSpeed);

  return () => clearInterval(interval);
  }, [isTraining, isTrainingComplete, trainingParams.learningRate, animationSpeed, activeDataset.difficulty]);

  const timelinePoint = timelineIndex >= 0 ? trainingHistory[timelineIndex] : trainingHistory[trainingHistory.length - 1];

  const explainerCards = useMemo(() => {
    if (trainingHistory.length < 4) {
      return [
        'Start training to generate trend-aware explainers.',
        'Use scenario presets to observe underfitting vs overfitting.',
        'Open comparison mode to understand architecture tradeoffs.'
      ];
    }

    const recent = trainingHistory.slice(-4);
    const lossDelta = recent[recent.length - 1].loss - recent[0].loss;
    const accDelta = recent[recent.length - 1].accuracy - recent[0].accuracy;
    const cards = [];

    if (lossDelta > 0.02) {
      cards.push('Loss is rising. This usually means learning rate is too high for the current dataset.');
    } else if (lossDelta < -0.02) {
      cards.push('Loss is steadily dropping. The optimizer is finding a good descent path.');
    } else {
      cards.push('Loss is plateauing. Consider changing architecture depth or lowering batch size.');
    }

    if (accDelta < 0.01) {
      cards.push('Accuracy is barely moving. Add capacity or switch to a richer preset.');
    } else {
      cards.push('Accuracy is improving. Keep current setup and monitor overfitting signals.');
    }

    if (trainingParams.learningRate > 0.04) {
      cards.push('Learning rate is aggressive; oscillations are expected at this scale.');
    } else if (trainingParams.learningRate < 0.005) {
      cards.push('Learning rate is conservative; training will be stable but slower.');
    } else {
      cards.push('Learning rate is in a balanced range for demonstration runs.');
    }
    return cards;
  }, [trainingHistory, trainingParams.learningRate]);

  const predictionProbs = useMemo(() => {
    const modelBias = MODEL_COMPARISON_BIAS[selectedModel] || 0;
    const datasetBoost = 1 / activeDataset.difficulty;
    const scoreA = (predictionInput.x1 * 1.8 + predictionInput.x2 * 0.7 + accuracy * 0.5) * datasetBoost + modelBias;
    const scoreB = (predictionInput.x2 * 1.6 + predictionInput.x3 * 0.9 + loss * 0.15) * datasetBoost - modelBias;
    const scoreC = (predictionInput.x3 * 1.4 + predictionInput.x1 * 0.5 + (1 - accuracy) * 0.4) * datasetBoost;
    const maxScore = Math.max(scoreA, scoreB, scoreC);
    const exps = [scoreA, scoreB, scoreC].map((score) => Math.exp(score - maxScore));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((value) => value / sum);
  }, [selectedModel, activeDataset.difficulty, predictionInput, accuracy, loss]);

  const confusionMatrix = useMemo(() => {
    const correct = Math.max(0.35, Math.min(0.97, accuracy));
    const offDiag = (1 - correct) / 2;
    return [
      [correct, offDiag * 0.7, offDiag * 1.3],
      [offDiag * 1.2, correct * 0.95, offDiag * 0.8],
      [offDiag * 0.9, offDiag * 1.1, correct * 1.05]
    ].map((row) => row.map((value) => Math.max(0, Math.min(1, value))));
  }, [accuracy]);

  const classMetrics = useMemo(() => {
    const names = activeDataset.classes;
    return names.map((name, index) => {
      const precision = Math.max(0.4, Math.min(0.99, accuracy - index * 0.03 + 0.1));
      const recall = Math.max(0.4, Math.min(0.99, accuracy - (2 - index) * 0.025 + 0.08));
      const f1 = (2 * precision * recall) / Math.max(0.0001, precision + recall);
      return {
        name,
        precision,
        recall,
        f1
      };
    });
  }, [activeDataset.classes, accuracy]);

  const missionStatus = useMemo(() => {
    const quickWin = accuracy >= 0.75;
    const expert = accuracy >= 0.9 && loss <= 0.08;
    const leanNet = architecture.length <= 5 && accuracy >= 0.8;
    const compareReady = Boolean(compareSnapshot);
    return [
      { name: 'Mission 1: Reach 75% accuracy', complete: quickWin },
      { name: 'Mission 2: Reach 90% with loss under 0.08', complete: expert },
      { name: 'Mission 3: Keep <=5 layers and hit 80%', complete: leanNet },
      { name: 'Mission 4: Complete side-by-side comparison', complete: compareReady }
    ];
  }, [accuracy, loss, architecture.length, compareSnapshot]);

  const layerProbe = useMemo(() => {
    const safeIndex = Math.max(0, Math.min(probeLayerIndex, architecture.length - 1));
    const layer = architecture[safeIndex];
    if (!layer) {
      return { index: 0, name: 'N/A', neurons: 0, saturation: 0, utilization: 0 };
    }
    const utilization = Math.max(0.15, Math.min(0.99, accuracy + (safeIndex / Math.max(1, architecture.length)) * 0.1));
    const saturation = Math.max(0.01, Math.min(0.95, 1 - loss + safeIndex * 0.04));
    return {
      index: safeIndex,
      name: layer.name,
      neurons: layer.neurons,
      saturation,
      utilization
    };
  }, [architecture, probeLayerIndex, accuracy, loss]);

  // UI Helper functions
  const addLayer = () => {
    setArchitecture(prev => [
      ...prev.slice(0, -1),
      { neurons: 8, activation: 'ReLU', name: `Hidden ${prev.length - 1}` },
      prev[prev.length - 1]
    ]);
  };

  const removeLayer = (index) => {
    if (architecture.length > 3 && index > 0 && index < architecture.length - 1) {
      setArchitecture(prev => prev.filter((_, i) => i !== index));
    }
  };

  const updateLayer = (index, field, value) => {
    setArchitecture(prev => prev.map((layer, i) =>
      i === index ? { ...layer, [field]: value } : layer
    ));
  };


  const applyModelPreset = useCallback((modelName) => {
    const preset = MODEL_PRESETS[modelName] || MODEL_PRESETS.Custom;
    setArchitecture(preset.map((layer) => ({ ...layer })));
    setSelectedModel(modelName);
    setIsTraining(false);
    setIsTrainingComplete(false);
    setEpoch(0);
    setBatch(0);
    setLoss(1.0);
    setAccuracy(0.1);
    setTrainingHistory([]);
    setTimelineIndex(-1);
    epochRef.current = 0;
    batchRef.current = 0;
    lossRef.current = 1.0;
    accuracyRef.current = 0.1;
    trainingRngRef.current = createSeededRng(simulationSeed);
    setStatusMessage(`${modelName} architecture loaded for ${datasetName}.`);
    setStorySteps((prev) => [...prev.slice(-7), `Loaded ${modelName} preset`]);
    setProbeLayerIndex(0);
  }, [simulationSeed, datasetName]);

  const startTraining = () => {
    trainingRngRef.current = createSeededRng(simulationSeed);
    setIsTraining(true);
    setIsTrainingComplete(false);
    setEpoch(0);
    setBatch(0);
    setLoss(1.0);
    setAccuracy(0.1);
    setTrainingHistory([]);
    setTimelineIndex(-1);
    epochRef.current = 0;
    batchRef.current = 0;
    lossRef.current = 1.0;
    accuracyRef.current = 0.1;
    animationProgressRef.current = 0;
    setStorySteps((prev) => [...prev.slice(-7), `Training started on ${datasetName}`]);
  };

  const stopTraining = () => {
    setIsTraining(false);
    setStorySteps((prev) => [...prev.slice(-7), `Training stopped at ${(accuracy * 100).toFixed(1)}%`]);
  };

  const resetNetwork = () => {
    setIsTraining(false);
    setIsTrainingComplete(false);
    setEpoch(0);
    setBatch(0);
    setLoss(1.0);
    setAccuracy(0.1);
    setTrainingHistory([]);
    setTimelineIndex(-1);
    epochRef.current = 0;
    batchRef.current = 0;
    lossRef.current = 1.0;
    accuracyRef.current = 0.1;
    trainingRngRef.current = createSeededRng(simulationSeed);
    animationProgressRef.current = 0;
    setStorySteps((prev) => [...prev.slice(-7), 'Network reset']);
  };

  const togglePanel = (panelName) => {
    setActivePanel((prev) => (prev === panelName ? null : panelName));
  };

  const applyScenarioPreset = useCallback((scenarioKey) => {
    const scenario = SCENARIO_PRESETS[scenarioKey];
    if (!scenario) return;
    setSelectedScenario(scenarioKey);
    setTrainingParams((prev) => ({
      ...prev,
      learningRate: scenario.learningRate,
      batchSize: scenario.batchSize
    }));
    applyModelPreset(scenario.model);
    setStatusMessage(`${scenario.label} scenario loaded.`);
    setStorySteps((prev) => [...prev.slice(-7), `Scenario: ${scenario.label}`]);
  }, [applyModelPreset]);

  const runModelComparison = useCallback(() => {
    const base = accuracy;
    const primaryBias = MODEL_COMPARISON_BIAS[selectedModel] || 0;
    const compareBias = MODEL_COMPARISON_BIAS[compareModel] || 0;
    const datasetPenalty = (activeDataset.difficulty - 1) * 0.06;
    const projectedCompare = Math.max(0.05, Math.min(0.98, base - primaryBias + compareBias - datasetPenalty));
    setCompareSnapshot({
      baselineModel: selectedModel,
      compareModel,
      baselineAccuracy: base,
      compareAccuracy: projectedCompare,
      delta: projectedCompare - base,
      baselineLoss: loss,
      compareLoss: Math.max(0.001, loss - (projectedCompare - base) * 0.4)
    });
    setStorySteps((prev) => [...prev.slice(-7), `Compared ${selectedModel} vs ${compareModel}`]);
  }, [accuracy, activeDataset.difficulty, compareModel, selectedModel, loss]);

  const runArchitectureAssistant = useCallback(() => {
    if (assistantGoal === 'Improve Accuracy') {
      setTrainingParams((prev) => ({ ...prev, learningRate: Math.min(0.03, prev.learningRate + 0.003) }));
      setAssistantMessage('Suggested: increase learning rate slightly and compare CNN/Transformer presets.');
    } else if (assistantGoal === 'Reduce Complexity') {
      if (architecture.length > 3) {
        setArchitecture((prev) => prev.filter((_, idx) => idx !== prev.length - 2));
      }
      setAssistantMessage('Suggested: remove one hidden layer and keep neurons under 16.');
    } else if (assistantGoal === 'Stabilize Training') {
      setTrainingParams((prev) => ({ ...prev, learningRate: Math.max(0.004, prev.learningRate * 0.75), batchSize: 64 }));
      setAssistantMessage('Suggested: lower learning rate and increase batch size for smoother gradients.');
    } else {
      setTrainingParams((prev) => ({ ...prev, learningRate: Math.min(0.02, prev.learningRate * 1.15), batchSize: 32 }));
      setAssistantMessage('Suggested: moderate LR increase with balanced batch size to speed convergence.');
    }
    setStorySteps((prev) => [...prev.slice(-7), `Assistant goal: ${assistantGoal}`]);
  }, [assistantGoal, architecture.length]);

  const saveExperiment = useCallback(() => {
    const run = {
      id: generateSessionId(),
      model: selectedModel,
      dataset: datasetName,
      accuracy,
      loss,
      epoch,
      batch,
      note: experimentNote || 'No note'
    };
    setExperimentRuns((prev) => [run, ...prev].slice(0, 20));
    setExperimentNote('');
    setStatusMessage('Experiment snapshot saved.');
    setStorySteps((prev) => [...prev.slice(-7), `Saved run ${run.id}`]);
  }, [selectedModel, datasetName, accuracy, loss, epoch, batch, experimentNote]);

  const addStoryStep = useCallback(() => {
    const step = `Step ${storySteps.length + 1}: ${selectedModel} on ${datasetName}, acc ${(accuracy * 100).toFixed(1)}%`;
    setStorySteps((prev) => [...prev.slice(-11), step]);
  }, [storySteps.length, selectedModel, datasetName, accuracy]);

  const createStoryShareLink = useCallback(() => {
    const storyId = generateSessionId();
    localStorage.setItem(`${STORY_KEY}${storyId}`, JSON.stringify({
      steps: storySteps,
      snapshot: {
        architecture,
        trainingParams,
        selectedModel,
        datasetName,
        accuracy,
        loss,
        epoch,
        batch
      }
    }));
    const link = `${window.location.origin}${window.location.pathname}#story=${storyId}`;
    setStoryLink(link);
    setStatusMessage('Story link generated.');
  }, [storySteps, architecture, trainingParams, selectedModel, datasetName, accuracy, loss, epoch, batch]);

  const addAnnotation = useCallback(() => {
    const text = annotationDraft.trim();
    if (!text) return;
    const annotation = {
      id: generateSessionId(),
      text,
      model: selectedModel,
      ts: new Date().toISOString()
    };
    setAnnotations((prev) => [annotation, ...prev].slice(0, 50));
    setAnnotationDraft('');
  }, [annotationDraft, selectedModel]);

  const buildSession = useCallback(() => ({
    architecture,
    trainingParams,
    cameraMode,
    cameraState,
    selectedModel,
    uiState: {
      activePanel,
      animationSpeed,
      datasetName,
      selectedScenario,
      presentationMode
    },
    simulation: {
      seed: simulationSeed
    }
  }), [architecture, trainingParams, cameraMode, cameraState, selectedModel, activePanel, animationSpeed, datasetName, selectedScenario, presentationMode, simulationSeed]);

  const applySession = useCallback((rawSession) => {
    const { session, error } = parseSessionPayload(rawSession);
    if (error || !session) {
      setStatusMessage(error || 'Session payload is invalid.');
      return false;
    }

    setArchitecture(session.architecture);
    setTrainingParams(session.trainingParams);
    setCameraMode(session.cameraMode);
    setCameraState(session.cameraState);
    cameraStateRef.current = session.cameraState;
    setSelectedModel(session.selectedModel);
    setActivePanel(session.uiState.activePanel);
    setAnimationSpeed(session.uiState.animationSpeed);
    setDatasetName(session.uiState.datasetName);
    setSelectedScenario(session.uiState.selectedScenario);
    setPresentationMode(session.uiState.presentationMode);
    setSimulationSeed(sanitizeSeed(session.simulation.seed));
    trainingRngRef.current = createSeededRng(sanitizeSeed(session.simulation.seed));
    setTrainingHistory([]);
    setTimelineIndex(-1);
    setStorySteps((prev) => [...prev.slice(-7), 'Session imported']);
    setStatusMessage('Session loaded.');
    return true;
  }, []);

  const saveDraftSession = useCallback(() => {
    const payload = serializeSession(buildSession());
    localStorage.setItem(LOCAL_DRAFT_KEY, payload);
    setStatusMessage('Session saved to local storage.');
  }, [buildSession]);

  const loadDraftSession = useCallback(() => {
    const payload = localStorage.getItem(LOCAL_DRAFT_KEY);
    if (!payload) {
      setStatusMessage('No saved local session found.');
      return;
    }

    applySession(payload);
  }, [applySession]);

  const exportSession = useCallback(() => {
    const blob = new Blob([serializeSession(buildSession())], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'nexa-visualize-session.json';
    anchor.click();
    URL.revokeObjectURL(url);
    setStatusMessage('Session exported as JSON.');
  }, [buildSession]);

  const importSession = useCallback((event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (loadEvent) => {
      applySession(loadEvent.target?.result);
    };
    reader.readAsText(file);
    event.target.value = '';
  }, [applySession]);

  const createShareLink = useCallback(() => {
    const newSessionId = generateSessionId();
    const payload = serializeSession(buildSession());
    localStorage.setItem(`${SHARED_PREFIX}${newSessionId}`, payload);
    if (onSessionRouteChange) {
      onSessionRouteChange(newSessionId);
    }
    setStatusMessage(`Share link ready: /session/${newSessionId}`);
  }, [buildSession, onSessionRouteChange]);


  useEffect(() => {
    if (!sessionId) return;

    const sharedPayload = localStorage.getItem(`${SHARED_PREFIX}${sessionId}`);
    if (!sharedPayload) {
      setStatusMessage(`Session ${sessionId} not found in local storage.`);
      return;
    }

    applySession(sharedPayload);
  }, [sessionId, applySession]);

  return (
    <div className="w-full h-screen bg-gray-900 relative overflow-hidden">
      <div ref={mountRef} data-testid="three-canvas-mount" className="w-full h-full" />

      {/* Header */}
      <div className="absolute top-4 left-4 z-20 bg-black bg-opacity-90 text-white p-3 rounded-lg backdrop-blur-sm">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-bold text-cyan-400">Nexa Visualize</h2>
          <div className={`text-xs px-2 py-1 rounded font-bold ${
            isTrainingComplete ? 'bg-blue-600 animate-pulse' :
            isTraining ? 'bg-green-600' : 'bg-gray-600'
          }`}>
            {isTrainingComplete ? 'TRAINED' : isTraining ? 'TRAINING' : 'IDLE'}
          </div>
        </div>
        <div className="text-xs text-gray-300">
          {architecture.length} layers • {architecture.reduce((sum, layer) => sum + layer.neurons, 0)} neurons
        </div>
        <div className="text-xs text-cyan-300 mt-1">Model: {selectedModel}</div>
        <div className="text-xs text-gray-300 mt-1">Dataset: {datasetName}</div>
        {statusMessage && <div className="text-xs text-yellow-300 mt-1">{statusMessage}</div>}
      </div>

      {/* Controls */}
      {!presentationMode && (
      <div className="absolute top-4 right-4 z-20 bg-black bg-opacity-90 text-white rounded-lg backdrop-blur-sm">
        <div className="flex">
          <button
            onClick={() => togglePanel('architecture')}
            className={`px-3 py-2 text-xs font-medium transition-colors ${
              activePanel === 'architecture' ? 'bg-purple-600' : 'bg-gray-700 hover:bg-gray-600'
            }`}
          >
            Architecture
          </button>
          <button
            onClick={() => togglePanel('parameters')}
            className={`px-3 py-2 text-xs font-medium transition-colors ${
              activePanel === 'parameters' ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
            }`}
          >
            Parameters
          </button>
          <button
            onClick={() => togglePanel('optimization')}
            className={`px-3 py-2 text-xs font-medium transition-colors ${
              activePanel === 'optimization' ? 'bg-green-600' : 'bg-gray-700 hover:bg-gray-600'
            }`}
          >
            Loss Landscape
          </button>
          <button
            onClick={() => togglePanel('enhancements')}
            className={`px-3 py-2 rounded-r text-xs font-medium transition-colors ${
              activePanel === 'enhancements' ? 'bg-indigo-600' : 'bg-gray-700 hover:bg-gray-600'
            }`}
          >
            Enhancements
          </button>
        </div>

        <div className="p-3 border-t border-gray-600">
          <div className="flex items-center gap-3 mb-2">
            <div className="flex gap-1">
              <button
                onClick={() => setCameraMode('auto')}
                className={`px-2 py-1 rounded text-xs ${
                  cameraMode === 'auto' ? 'bg-purple-500' : 'bg-gray-600'
                }`}
              >
                Auto
              </button>
              <button
                onClick={() => setCameraMode('manual')}
                className={`px-2 py-1 rounded text-xs ${
                  cameraMode === 'manual' ? 'bg-purple-500' : 'bg-gray-600'
                }`}
              >
                Manual
              </button>
            </div>

            <div className="flex items-center gap-1 text-xs">
              <span>Speed:</span>
              <input
                type="range"
                min="0.5"
                max="3.0"
                step="0.1"
                value={animationSpeed}
                onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
                className="w-12 h-1"
              />
              <span className="w-8">{animationSpeed.toFixed(1)}x</span>
            </div>
            <button
              onClick={() => setPresentationMode((prev) => !prev)}
              className="bg-gray-600 hover:bg-gray-700 text-white px-2 py-1 rounded text-xs font-medium"
            >
              {presentationMode ? 'Exit Present' : 'Present'}
            </button>
          </div>

          <div className="grid grid-cols-3 gap-1 mb-2">
            <button
              onClick={saveDraftSession}
              className="bg-blue-700 hover:bg-blue-800 text-white px-2 py-1 rounded text-xs font-medium"
            >
              Save Local
            </button>
            <button
              onClick={loadDraftSession}
              className="bg-blue-700 hover:bg-blue-800 text-white px-2 py-1 rounded text-xs font-medium"
            >
              Load Local
            </button>
            <button
              onClick={createShareLink}
              className="bg-indigo-700 hover:bg-indigo-800 text-white px-2 py-1 rounded text-xs font-medium"
            >
              Share Link
            </button>
            <button
              onClick={exportSession}
              className="bg-gray-600 hover:bg-gray-700 text-white px-2 py-1 rounded text-xs font-medium"
            >
              Export JSON
            </button>
            <label className="bg-gray-600 hover:bg-gray-700 text-white px-2 py-1 rounded text-xs font-medium text-center cursor-pointer">
              Import JSON
              <input type="file" accept="application/json" onChange={importSession} className="hidden" />
            </label>
            <select
              value={selectedModel}
              onChange={(e) => applyModelPreset(e.target.value)}
              className="bg-gray-700 text-white px-2 py-1 rounded text-xs"
            >
              <option value="Custom">Custom</option>
              <option value="CNN">CNN</option>
              <option value="Transformer">Transformer</option>
              <option value="Neural Operator">Neural Operator</option>
              <option value="MoE">MoE</option>
              <option value="Autoencoder">Autoencoder</option>
              <option value="MLP">MLP</option>
            </select>
          </div>

          <div className="flex gap-2">
            {!isTraining ? (
              <button
                onClick={startTraining}
                className="bg-green-600 hover:bg-green-700 text-white px-4 py-1 rounded text-xs font-medium"
                disabled={isTrainingComplete}
              >
                Start Training
              </button>
            ) : (
              <button
                onClick={stopTraining}
                className="bg-red-600 hover:bg-red-700 text-white px-4 py-1 rounded text-xs font-medium"
              >
                Stop
              </button>
            )}
            <button
              onClick={resetNetwork}
              className="bg-gray-600 hover:bg-gray-700 text-white px-3 py-1 rounded text-xs font-medium"
            >
              Reset
            </button>
          </div>
        </div>
      </div>
      )}

      {presentationMode && (
        <button
          onClick={() => setPresentationMode(false)}
          className="absolute top-4 right-4 z-20 bg-gray-700 hover:bg-gray-600 text-white px-3 py-2 rounded text-xs font-medium"
        >
          Exit Presentation
        </button>
      )}

      {/* Training Status */}
      <div className="absolute bottom-4 left-4 right-4 z-20 bg-black bg-opacity-90 text-white p-3 rounded-lg backdrop-blur-sm">
        <div className="grid grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-xs text-gray-400">Epoch</div>
            <div className="text-lg font-bold text-cyan-400">{epoch}</div>
          </div>
          <div>
            <div className="text-xs text-gray-400">Batch</div>
            <div aria-label="Batch value" className="text-lg font-bold text-blue-400">{batch}</div>
          </div>
          <div>
            <div className="text-xs text-gray-400">Loss</div>
            <div className="text-lg font-bold text-red-400">{loss.toFixed(4)}</div>
          </div>
          <div>
            <div className={`text-xs ${accuracy >= 0.9 ? 'text-green-300' : 'text-gray-400'}`}>
              Accuracy {accuracy >= 0.9 && '🎉'}
            </div>
            <div className={`text-lg font-bold ${accuracy >= 0.9 ? 'text-green-300 animate-pulse' : 'text-green-400'}`}>
              {(accuracy * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>

      {/* Active Panel */}
      {activePanel && !presentationMode && (
        <div
          data-testid="config-panel"
          className="absolute top-24 right-4 z-20 bg-black bg-opacity-95 text-white rounded-lg backdrop-blur-sm max-w-sm max-h-96 overflow-y-auto"
        >
          {activePanel === 'architecture' && (
            <div className="p-4">
              <h3 className="text-sm font-bold mb-3 text-purple-400">Network Architecture</h3>
              <div className="space-y-2">
                {architecture.map((layer, index) => (
                  <div key={index} className="bg-gray-800 p-2 rounded">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold">{layer.name}</span>
                      {index > 0 && index < architecture.length - 1 && (
                        <button
                          onClick={() => removeLayer(index)}
                          className="text-red-400 hover:text-red-300 text-xs w-4 h-4 flex items-center justify-center"
                        >
                          ×
                        </button>
                      )}
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="block text-xs">Neurons:</label>
                        <input
                          type="number"
                          min="1"
                          max="64"
                          value={layer.neurons}
                          onChange={(e) => updateLayer(index, 'neurons', parseInt(e.target.value) || 1)}
                          className="w-full bg-gray-700 text-white text-xs p-1 rounded"
                        />
                      </div>
                      <div>
                        <label className="block text-xs">Activation:</label>
                        <select
                          value={layer.activation}
                          onChange={(e) => updateLayer(index, 'activation', e.target.value)}
                          className="w-full bg-gray-700 text-white text-xs p-1 rounded"
                          disabled={index === 0 || index === architecture.length - 1}
                        >
                          <option value="ReLU">ReLU</option>
                          <option value="Sigmoid">Sigmoid</option>
                          <option value="Tanh">Tanh</option>
                          <option value="Linear">Linear</option>
                          <option value="Softmax">Softmax</option>
                        </select>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <button
                onClick={addLayer}
                className="w-full mt-3 bg-purple-600 hover:bg-purple-700 text-white py-2 rounded text-xs font-medium"
              >
                Add Hidden Layer
              </button>
            </div>
          )}

          {activePanel === 'parameters' && (
            <div className="p-4">
              <h3 className="text-sm font-bold mb-3 text-blue-400">Training Parameters</h3>
              <div className="space-y-3">
                <div>
                  <label className="block text-xs mb-1">Learning Rate:</label>
                  <input
                    type="number"
                    step="0.001"
                    min="0.001"
                    max="1"
                    value={trainingParams.learningRate}
                    onChange={(e) => setTrainingParams(prev => ({ ...prev, learningRate: parseFloat(e.target.value) || 0.01 }))}
                    className="w-full bg-gray-700 text-white text-xs p-2 rounded"
                  />
                </div>
                <div>
                  <label className="block text-xs mb-1">Batch Size:</label>
                  <select
                    value={trainingParams.batchSize}
                    onChange={(e) => setTrainingParams(prev => ({ ...prev, batchSize: parseInt(e.target.value) }))}
                    className="w-full bg-gray-700 text-white text-xs p-2 rounded"
                  >
                    <option value={16}>16</option>
                    <option value={32}>32</option>
                    <option value={64}>64</option>
                    <option value={128}>128</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs mb-1">Optimizer:</label>
                  <select
                    value={trainingParams.optimizer}
                    onChange={(e) => setTrainingParams(prev => ({ ...prev, optimizer: e.target.value }))}
                    className="w-full bg-gray-700 text-white text-xs p-2 rounded"
                  >
                    <option value="Adam">Adam</option>
                    <option value="SGD">SGD</option>
                    <option value="RMSprop">RMSprop</option>
                    <option value="AdaGrad">AdaGrad</option>
                  </select>
                </div>
              </div>
            </div>
          )}

          {activePanel === 'optimization' && (
            <div className="p-4">
              <h3 className="text-sm font-bold mb-3 text-green-400">Loss Landscape</h3>
              <div className="text-xs space-y-2">
                <p>🔴 Red ball shows optimizer descent</p>
                <p>🟦 Blue surface is loss landscape</p>
                <p>Ball moves toward global minimum as loss decreases</p>
                <div className="bg-gray-800 p-2 rounded mt-3">
                  <div>Optimizer: <span className="text-blue-400">{trainingParams.optimizer}</span></div>
                  <div>Current Loss: <span className="text-red-400">{loss.toFixed(4)}</span></div>
                  <div>Learning Rate: <span className="text-yellow-400">{trainingParams.learningRate}</span></div>
                </div>
              </div>
            </div>
          )}

          {activePanel === 'enhancements' && (
            <div className="p-4">
              <h3 className="text-sm font-bold mb-3 text-indigo-300">Enhancement Lab</h3>
              <div className="space-y-3 text-xs">
                <div className="bg-gray-800 p-2 rounded">
                  <div className="font-bold mb-1">1. Dataset Playground</div>
                  <select
                    value={datasetName}
                    onChange={(e) => setDatasetName(e.target.value)}
                    className="w-full bg-gray-700 text-white text-xs p-1 rounded mb-1"
                  >
                    {Object.keys(DATASET_PRESETS).map((name) => (
                      <option key={name} value={name}>{name}</option>
                    ))}
                  </select>
                  <div className="text-gray-300">{activeDataset.summary}</div>
                </div>

                <div className="bg-gray-800 p-2 rounded">
                  <div className="font-bold mb-1">2. Scenario Presets</div>
                  <div className="grid grid-cols-2 gap-1">
                    {Object.entries(SCENARIO_PRESETS).map(([key, scenario]) => (
                      <button
                        key={key}
                        onClick={() => applyScenarioPreset(key)}
                        className={`px-2 py-1 rounded text-xs ${selectedScenario === key ? 'bg-indigo-600' : 'bg-gray-600 hover:bg-gray-700'}`}
                      >
                        {scenario.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="bg-gray-800 p-2 rounded">
                  <div className="font-bold mb-1">3. Side-by-Side Comparison</div>
                  <div className="flex gap-1 items-center mb-1">
                    <select
                      value={compareModel}
                      onChange={(e) => setCompareModel(e.target.value)}
                      className="w-full bg-gray-700 text-white text-xs p-1 rounded"
                    >
                      {Object.keys(MODEL_PRESETS).map((name) => (
                        <option key={name} value={name}>{name}</option>
                      ))}
                    </select>
                    <button onClick={runModelComparison} className="bg-indigo-700 hover:bg-indigo-800 px-2 py-1 rounded">Run</button>
                  </div>
                  {compareSnapshot && (
                    <div className="text-gray-300">
                      <div>{compareSnapshot.baselineModel}: {(compareSnapshot.baselineAccuracy * 100).toFixed(1)}%</div>
                      <div>{compareSnapshot.compareModel}: {(compareSnapshot.compareAccuracy * 100).toFixed(1)}%</div>
                      <div className={compareSnapshot.delta >= 0 ? 'text-green-300' : 'text-red-300'}>
                        Δ {(compareSnapshot.delta * 100).toFixed(1)}%
                      </div>
                    </div>
                  )}
                </div>

                <div className="bg-gray-800 p-2 rounded">
                  <div className="font-bold mb-1">4. Why This Happened</div>
                  <div className="space-y-1">
                    {explainerCards.map((card, idx) => (
                      <div key={`explain-${idx}`} className="text-gray-300">{card}</div>
                    ))}
                  </div>
                </div>

                <div className="bg-gray-800 p-2 rounded">
                  <div className="font-bold mb-1">5. Interactive Prediction</div>
                  <div className="grid grid-cols-3 gap-1 mb-1">
                    {['x1', 'x2', 'x3'].map((key) => (
                      <input
                        key={key}
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={predictionInput[key]}
                        onChange={(e) => setPredictionInput((prev) => ({ ...prev, [key]: parseFloat(e.target.value) }))}
                      />
                    ))}
                  </div>
                  {activeDataset.classes.map((name, idx) => (
                    <div key={name} className="mb-1">
                      <div className="flex justify-between"><span>{name}</span><span>{(predictionProbs[idx] * 100).toFixed(1)}%</span></div>
                      <div className="bg-gray-700 rounded">
                        <div className="bg-blue-600 h-1 rounded" style={{ width: `${predictionProbs[idx] * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="bg-gray-800 p-2 rounded">
                  <div className="font-bold mb-1">6. Layer Probe Mode</div>
                  <div className="flex gap-1 mb-1 overflow-y-auto">
                    {architecture.map((layer, idx) => (
                      <button
                        key={`probe-${idx}`}
                        onClick={() => setProbeLayerIndex(idx)}
                        className={`px-2 py-1 rounded text-xs ${layerProbe.index === idx ? 'bg-blue-600' : 'bg-gray-600 hover:bg-gray-700'}`}
                      >
                        L{idx + 1}
                      </button>
                    ))}
                  </div>
                  <div>{layerProbe.name} • {layerProbe.neurons} neurons</div>
                  <div>Utilization {(layerProbe.utilization * 100).toFixed(1)}% • Saturation {(layerProbe.saturation * 100).toFixed(1)}%</div>
                </div>

                <div className="bg-gray-800 p-2 rounded">
                  <div className="font-bold mb-1">7. Training Timeline Replay</div>
                  {trainingHistory.length > 0 ? (
                    <>
                      <input
                        aria-label="Timeline replay slider"
                        type="range"
                        min="0"
                        max={trainingHistory.length - 1}
                        step="1"
                        value={timelineIndex >= 0 ? timelineIndex : trainingHistory.length - 1}
                        onChange={(e) => setTimelineIndex(parseInt(e.target.value, 10))}
                        className="w-full"
                      />
                      {timelinePoint && (
                        <div>
                          <div>Epoch {timelinePoint.epoch} • Batch {timelinePoint.batch}</div>
                          <div>Loss {timelinePoint.loss.toFixed(4)} • Acc {(timelinePoint.accuracy * 100).toFixed(1)}%</div>
                        </div>
                      )}
                    </>
                  ) : <div className="text-gray-300">No training timeline yet.</div>}
                </div>

                <div className="bg-gray-800 p-2 rounded">
                  <div className="font-bold mb-1">8. Missions</div>
                  {missionStatus.map((mission) => (
                    <div key={mission.name} className={mission.complete ? 'text-green-300' : 'text-gray-300'}>
                      {mission.complete ? '✓' : '•'} {mission.name}
                    </div>
                  ))}
                </div>

                <div className="bg-gray-800 p-2 rounded">
                  <div className="font-bold mb-1">9. Architecture Assistant</div>
                  <div className="flex gap-1 mb-1">
                    <select
                      value={assistantGoal}
                      onChange={(e) => setAssistantGoal(e.target.value)}
                      className="w-full bg-gray-700 text-white text-xs p-1 rounded"
                    >
                      {ASSISTANT_GOALS.map((goal) => <option key={goal} value={goal}>{goal}</option>)}
                    </select>
                    <button onClick={runArchitectureAssistant} className="bg-indigo-700 hover:bg-indigo-800 px-2 py-1 rounded">Apply</button>
                  </div>
                  {assistantMessage && <div className="text-gray-300">{assistantMessage}</div>}
                </div>

                <div className="bg-gray-800 p-2 rounded">
                  <div className="font-bold mb-1">10. Experiment Tracker</div>
                  <div className="flex gap-1 mb-1">
                    <input
                      placeholder="Run note"
                      value={experimentNote}
                      onChange={(e) => setExperimentNote(e.target.value)}
                      className="w-full bg-gray-700 text-white text-xs p-1 rounded"
                    />
                    <button onClick={saveExperiment} className="bg-blue-700 hover:bg-blue-800 px-2 py-1 rounded">Save</button>
                  </div>
                  <div className="max-h-24 overflow-y-auto">
                    {experimentRuns.slice(0, 6).map((run) => (
                      <div key={run.id} className="text-gray-300 mb-1">
                        {run.model} • {run.dataset} • {(run.accuracy * 100).toFixed(1)}% • {run.note}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-gray-800 p-2 rounded">
                  <div className="font-bold mb-1">11. Shareable Story Links</div>
                  <div className="flex gap-1 mb-1">
                    <button onClick={addStoryStep} className="bg-gray-600 hover:bg-gray-700 px-2 py-1 rounded">Add Step</button>
                    <button onClick={createStoryShareLink} className="bg-indigo-700 hover:bg-indigo-800 px-2 py-1 rounded">Create Link</button>
                  </div>
                  {storySteps.map((step, idx) => <div key={`story-step-${idx}`} className="text-gray-300">{step}</div>)}
                  {storyLink && <div className="text-cyan-300 mt-1 break-all">{storyLink}</div>}
                </div>

                <div className="bg-gray-800 p-2 rounded">
                  <div className="font-bold mb-1">12. Confusion Matrix + Class Metrics</div>
                  <div className="grid grid-cols-3 gap-1 mb-1">
                    {confusionMatrix.flatMap((row, rIdx) =>
                      row.map((value, cIdx) => (
                        <div key={`cm-${rIdx}-${cIdx}`} className="bg-gray-700 p-1 text-center rounded">
                          {(value * 100).toFixed(0)}
                        </div>
                      ))
                    )}
                  </div>
                  {classMetrics.map((metric) => (
                    <div key={metric.name} className="text-gray-300">
                      {metric.name}: P {(metric.precision * 100).toFixed(0)} / R {(metric.recall * 100).toFixed(0)} / F1 {(metric.f1 * 100).toFixed(0)}
                    </div>
                  ))}
                </div>

                <div className="bg-gray-800 p-2 rounded">
                  <div className="font-bold mb-1">13. Failure Case Gallery</div>
                  {activeDataset.failureCases.map((failure) => (
                    <div key={failure} className="text-red-300">• {failure}</div>
                  ))}
                </div>

                <div className="bg-gray-800 p-2 rounded">
                  <div className="font-bold mb-1">14. Presentation Mode</div>
                  <button
                    onClick={() => setPresentationMode((prev) => !prev)}
                    className="bg-blue-700 hover:bg-blue-800 px-2 py-1 rounded"
                  >
                    {presentationMode ? 'Disable Presentation Mode' : 'Enable Presentation Mode'}
                  </button>
                </div>

                <div className="bg-gray-800 p-2 rounded">
                  <div className="font-bold mb-1">15. Collaboration Mode</div>
                  <div className="flex gap-1 mb-1">
                    <button
                      onClick={() => setCollabMode((prev) => !prev)}
                      className={`px-2 py-1 rounded ${collabMode ? 'bg-green-700' : 'bg-gray-600 hover:bg-gray-700'}`}
                    >
                      {collabMode ? 'Collab On' : 'Collab Off'}
                    </button>
                    <input
                      placeholder="Add annotation"
                      value={annotationDraft}
                      onChange={(e) => setAnnotationDraft(e.target.value)}
                      className="w-full bg-gray-700 text-white text-xs p-1 rounded"
                    />
                    <button onClick={addAnnotation} className="bg-indigo-700 hover:bg-indigo-800 px-2 py-1 rounded">Post</button>
                  </div>
                  <div className="max-h-24 overflow-y-auto">
                    {annotations.slice(0, 8).map((note) => (
                      <div key={note.id} className="text-gray-300 mb-1">
                        {note.model}: {note.text}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default NeuralNetwork3D;
