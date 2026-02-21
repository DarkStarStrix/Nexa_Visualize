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
import { computeTrainingTick, GUIDED_PHASE_COPY, getTrainingPhase, TRAINING_PHASE } from './engine/trainingSimulation';
import { createSceneManager } from './engine/sceneManager';
import { getProgressPercent, getTelemetryLabel, getTrainingBadge } from './selectors/trainingSelectors';

const LOCAL_DRAFT_KEY = 'nexa.visualize.session.draft';
const SHARED_PREFIX = 'nexa.visualize.session.shared.';

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
  const telemetryLevelRef = useRef(DEFAULT_SESSION_STATE.uiState.telemetryLevel);
  const reducedMotionRef = useRef(false);
  const cameraStateRef = useRef(DEFAULT_CAMERA_STATE);
  const trainingRngRef = useRef(createSeededRng(DEFAULT_SESSION_STATE.simulation.seed));
  const lastPerfSampleRef = useRef({ time: 0, frames: 0 });
  const lastTimelineSampleRef = useRef(0);
  const phaseUiRef = useRef({ phase: TRAINING_PHASE.IDLE, time: 0 });
  const timelinePreviewRef = useRef(false);
  const timelinePreviewTimeoutRef = useRef(null);

  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [batch, setBatch] = useState(0);
  const [loss, setLoss] = useState(1.0);
  const [accuracy, setAccuracy] = useState(0.1);
  const [isTrainingComplete, setIsTrainingComplete] = useState(false);
  const [activePanel, setActivePanel] = useState(DEFAULT_SESSION_STATE.uiState.activePanel);
  const [cameraMode, setCameraMode] = useState(DEFAULT_SESSION_STATE.cameraMode);
  const [animationSpeed, setAnimationSpeed] = useState(DEFAULT_SESSION_STATE.uiState.animationSpeed);
  const [guidedMode, setGuidedMode] = useState(DEFAULT_SESSION_STATE.uiState.guidedMode);
  const [telemetryLevel, setTelemetryLevel] = useState(DEFAULT_SESSION_STATE.uiState.telemetryLevel);
  const [simulationSeed, setSimulationSeed] = useState(DEFAULT_SESSION_STATE.simulation.seed);
  const [selectedModel, setSelectedModel] = useState(DEFAULT_SESSION_STATE.selectedModel);
  const [trainingPhase, setTrainingPhase] = useState(TRAINING_PHASE.IDLE);
  const [telemetry, setTelemetry] = useState({ fps: 0, neuronCount: 0, connectionCount: 0 });
  const [timelineProgress, setTimelineProgress] = useState(0);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);
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
    telemetryLevelRef.current = telemetryLevel;
  }, [telemetryLevel]);

  useEffect(() => {
    reducedMotionRef.current = prefersReducedMotion;
  }, [prefersReducedMotion]);

  useEffect(() => {
    cameraStateRef.current = cameraState;
  }, [cameraState]);

  useEffect(() => {
    trainingRngRef.current = createSeededRng(simulationSeed);
  }, [simulationSeed]);

  useEffect(() => {
    if (!window.matchMedia) return;
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setPrefersReducedMotion(mediaQuery.matches);
    update();

    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', update);
      return () => mediaQuery.removeEventListener('change', update);
    }

    mediaQuery.addListener(update);
    return () => mediaQuery.removeListener(update);
  }, []);

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

  useEffect(() => {
    layersLengthRef.current = layers.length;
  }, [layers.length]);

  // Create dense network connections like in the provided image
  const createNetwork = useCallback(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    clearNetwork({ scene, neurons: neuronsRef.current, connections: connectionsRef.current });
    const seededNetworkRng = createSeededRng(simulationSeed + layers.length * 131);
    const { neurons, connections, stats } = buildDenseNetwork({
      scene,
      layers,
      random: seededNetworkRng,
      selectedModel
    });

    neuronsRef.current = neurons;
    connectionsRef.current = connections;
    setTelemetry(prev => ({
      ...prev,
      neuronCount: stats.neuronCount,
      connectionCount: stats.connectionCount
    }));

    const optimalDistance = calculateOptimalCameraDistance(layers);
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
    mountNode.addEventListener('wheel', onWheel, { passive: false });

    // Main animation loop - SIMPLIFIED AND RELIABLE
    const animate = (currentTime) => {
      const deltaTime = currentTime * 0.001;

      // Camera positioning
      if (cameraModeRef.current === 'auto') {
        const orbitSpeed = reducedMotionRef.current ? 0.15 : 0.3;
        const time = deltaTime * orbitSpeed;
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
      if ((isTrainingRef.current && !isTrainingCompleteRef.current) || timelinePreviewRef.current) {
        const neurons = neuronsRef.current;
        const connections = connectionsRef.current;

        // Continuous animation progress
        animationProgressRef.current = (deltaTime * animationSpeedRef.current * 0.5) % (layersLengthRef.current * 2 + 2);
        const progress = animationProgressRef.current;
        if (currentTime - lastTimelineSampleRef.current > 200) {
          const cycleLength = layersLengthRef.current * 2 + 2;
          setTimelineProgress((progress / cycleLength) * 100);
          lastTimelineSampleRef.current = currentTime;
        }
        const phase = getTrainingPhase(progress, layersLengthRef.current);
        if (phaseUiRef.current.phase !== phase) {
          phaseUiRef.current = { phase, time: currentTime };
          setTrainingPhase(phase);
        }

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
      else if (!timelinePreviewRef.current && phaseUiRef.current.phase !== TRAINING_PHASE.IDLE) {
        phaseUiRef.current = { phase: TRAINING_PHASE.IDLE, time: currentTime };
        setTrainingPhase(TRAINING_PHASE.IDLE);
      }

      // Update optimization ball
      if (optimizationBallRef.current && isTrainingRef.current) {
        const ball = optimizationBallRef.current;
        const time = deltaTime * 2;
        ball.position.x = Math.cos(time) * (3 - lossRef.current * 2.5);
        ball.position.z = Math.sin(time) * (3 - lossRef.current * 2.5);
        ball.position.y = -6 + lossRef.current * 2;
      }

      if (telemetryLevelRef.current === 'basic') {
        lastPerfSampleRef.current.frames += 1;
        const elapsed = currentTime - lastPerfSampleRef.current.time;
        if (elapsed >= 1000) {
          const fps = Math.round((lastPerfSampleRef.current.frames * 1000) / elapsed);
          setTelemetry(prev => ({ ...prev, fps }));
          lastPerfSampleRef.current = { time: currentTime, frames: 0 };
        }
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
        if (timelinePreviewTimeoutRef.current) {
          clearTimeout(timelinePreviewTimeoutRef.current);
          timelinePreviewTimeoutRef.current = null;
        }
        mountNode.removeEventListener('mousedown', onMouseDown);
        mountNode.removeEventListener('mousemove', onMouseMove);
        mountNode.removeEventListener('mouseup', onMouseUp);
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
      setBatch(prev => {
        const nextBatch = prev + 1;
        if (nextBatch % 10 === 0) {
          setEpoch(current => current + 1);
        }
        return nextBatch;
      });

      const tick = computeTrainingTick({
        previousLoss: lossRef.current,
        previousAccuracy: accuracyRef.current,
        learningRate: trainingParams.learningRate,
        random: trainingRngRef.current
      });

      lossRef.current = tick.loss;
      accuracyRef.current = tick.accuracy;
      setLoss(tick.loss);
      setAccuracy(tick.accuracy);

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
  }, [isTraining, isTrainingComplete, trainingParams.learningRate, animationSpeed]);

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
    setTrainingPhase(TRAINING_PHASE.IDLE);
    lossRef.current = 1.0;
    accuracyRef.current = 0.1;
    trainingRngRef.current = createSeededRng(simulationSeed);
    setStatusMessage(`${modelName} architecture loaded.`);
  }, [simulationSeed]);

  const startTraining = () => {
    timelinePreviewRef.current = false;
    if (timelinePreviewTimeoutRef.current) {
      clearTimeout(timelinePreviewTimeoutRef.current);
      timelinePreviewTimeoutRef.current = null;
    }
    trainingRngRef.current = createSeededRng(simulationSeed);
    setIsTraining(true);
    setIsTrainingComplete(false);
    setEpoch(0);
    setBatch(0);
    setLoss(1.0);
    setAccuracy(0.1);
    setTrainingPhase(TRAINING_PHASE.FORWARD);
    phaseUiRef.current = { phase: TRAINING_PHASE.FORWARD, time: 0 };
    lossRef.current = 1.0;
    accuracyRef.current = 0.1;
    animationProgressRef.current = 0;
  };

  const stopTraining = () => {
    setIsTraining(false);
    setTrainingPhase(TRAINING_PHASE.IDLE);
    timelinePreviewRef.current = false;
  };

  const resetNetwork = () => {
    setIsTraining(false);
    setIsTrainingComplete(false);
    setEpoch(0);
    setBatch(0);
    setLoss(1.0);
    setAccuracy(0.1);
    setTrainingPhase(TRAINING_PHASE.IDLE);
    lossRef.current = 1.0;
    accuracyRef.current = 0.1;
    trainingRngRef.current = createSeededRng(simulationSeed);
    animationProgressRef.current = 0;
    setTimelineProgress(0);
    timelinePreviewRef.current = false;
  };

  const togglePanel = (panelName) => {
    setActivePanel((prev) => (prev === panelName ? null : panelName));
  };

  const scrubTimeline = useCallback((nextPercent) => {
    const safePercent = Math.max(0, Math.min(100, nextPercent));
    setTimelineProgress(safePercent);
    const cycleLength = layersLengthRef.current * 2 + 2;
    const scrubbedProgress = (safePercent / 100) * cycleLength;
    animationProgressRef.current = scrubbedProgress;
    const phase = getTrainingPhase(scrubbedProgress, layersLengthRef.current);
    setTrainingPhase(phase);
    phaseUiRef.current = { phase, time: performance.now() };
    timelinePreviewRef.current = true;
    if (timelinePreviewTimeoutRef.current) {
      clearTimeout(timelinePreviewTimeoutRef.current);
    }
    timelinePreviewTimeoutRef.current = setTimeout(() => {
      timelinePreviewRef.current = false;
      if (!isTrainingRef.current) {
        setTrainingPhase(TRAINING_PHASE.IDLE);
      }
    }, 1500);
  }, []);

  const buildSession = useCallback(() => ({
    architecture,
    trainingParams,
    cameraMode,
    cameraState,
    selectedModel,
    uiState: {
      activePanel,
      animationSpeed,
      guidedMode,
      telemetryLevel
    },
    simulation: {
      seed: simulationSeed
    }
  }), [architecture, trainingParams, cameraMode, cameraState, selectedModel, activePanel, animationSpeed, guidedMode, telemetryLevel, simulationSeed]);

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
    setGuidedMode(session.uiState.guidedMode);
    setTelemetryLevel(session.uiState.telemetryLevel);
    setSimulationSeed(sanitizeSeed(session.simulation.seed));
    trainingRngRef.current = createSeededRng(sanitizeSeed(session.simulation.seed));
    setTimelineProgress(0);
    timelinePreviewRef.current = false;
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

  const guidedPhase = GUIDED_PHASE_COPY[trainingPhase] || GUIDED_PHASE_COPY[TRAINING_PHASE.IDLE];
  const trainingBadge = getTrainingBadge({ isTraining, isTrainingComplete });
  const telemetryLabel = getTelemetryLabel(telemetry);
  const accuracyPercent = getProgressPercent(accuracy);

  return (
    <div className="w-full h-screen bg-gray-900 relative overflow-hidden">
      <div ref={mountRef} data-testid="three-canvas-mount" className="w-full h-full" />

      {/* Header */}
      <div className="absolute top-4 left-4 bg-black bg-opacity-90 text-white p-3 rounded-lg backdrop-blur-sm">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-bold text-cyan-400">Nexa Visualize</h2>
          <div className={`text-xs px-2 py-1 rounded font-bold ${trainingBadge.className} ${isTrainingComplete && !prefersReducedMotion ? 'animate-pulse' : ''}`}>
            {trainingBadge.label}
          </div>
        </div>
        <div className="text-xs text-gray-300">
          {architecture.length} layers • {architecture.reduce((sum, layer) => sum + layer.neurons, 0)} neurons
        </div>
        <div className="text-xs text-cyan-300 mt-1">Model: {selectedModel}</div>
        {guidedMode && (
          <div className="text-xs text-blue-400 mt-1">Phase: {guidedPhase.title}</div>
        )}
        {telemetryLevel === 'basic' && (
          <div className="text-xs text-gray-400 mt-1">{telemetryLabel}</div>
        )}
        {statusMessage && <div className="text-xs text-yellow-300 mt-1">{statusMessage}</div>}
      </div>

      {/* Controls */}
      <div className="absolute top-4 right-4 bg-black bg-opacity-90 text-white rounded-lg backdrop-blur-sm">
        <div className="flex">
          <button
            onClick={() => togglePanel('architecture')}
            aria-pressed={activePanel === 'architecture'}
            className={`px-3 py-2 rounded-l text-xs font-medium transition-colors ${
              activePanel === 'architecture' ? 'bg-purple-600' : 'bg-gray-700 hover:bg-gray-600'
            }`}
          >
            Architecture
          </button>
          <button
            onClick={() => togglePanel('parameters')}
            aria-pressed={activePanel === 'parameters'}
            className={`px-3 py-2 text-xs font-medium transition-colors ${
              activePanel === 'parameters' ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
            }`}
          >
            Parameters
          </button>
          <button
            onClick={() => togglePanel('optimization')}
            aria-pressed={activePanel === 'optimization'}
            className={`px-3 py-2 rounded-r text-xs font-medium transition-colors ${
              activePanel === 'optimization' ? 'bg-green-600' : 'bg-gray-700 hover:bg-gray-600'
            }`}
          >
            Loss Landscape
          </button>
        </div>

        <div className="p-3 border-t border-gray-600">
          <div className="flex items-center gap-3 mb-2">
            <div className="flex gap-1">
              <button
                onClick={() => setCameraMode('auto')}
                aria-pressed={cameraMode === 'auto'}
                className={`px-2 py-1 rounded text-xs ${
                  cameraMode === 'auto' ? 'bg-purple-500' : 'bg-gray-600'
                }`}
              >
                Auto
              </button>
              <button
                onClick={() => setCameraMode('manual')}
                aria-pressed={cameraMode === 'manual'}
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
              onClick={() => setGuidedMode(prev => !prev)}
              aria-pressed={guidedMode}
              className={`px-2 py-1 rounded text-xs ${guidedMode ? 'bg-blue-600' : 'bg-gray-600'}`}
            >
              Guided
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

      {/* Training Status */}
      <div className="absolute bottom-4 left-4 right-4 bg-black bg-opacity-90 text-white p-3 rounded-lg backdrop-blur-sm">
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
              Accuracy {accuracyPercent >= 90 ? '🎉' : ''}
            </div>
            <div className={`text-lg font-bold ${accuracyPercent >= 90 ? `text-green-300 ${prefersReducedMotion ? '' : 'animate-pulse'}` : 'text-green-400'}`}>
              {accuracyPercent.toFixed(1)}%
            </div>
          </div>
        </div>
        {guidedMode && (
          <div className="text-xs text-gray-300 mt-1">
            <div>{guidedPhase.description}</div>
            <div className="mt-1 flex items-center gap-2">
              <label htmlFor="timeline-scrubber">Timeline</label>
              <input
                id="timeline-scrubber"
                aria-label="Training timeline scrubber"
                type="range"
                min="0"
                max="100"
                step="1"
                value={timelineProgress}
                onChange={(e) => scrubTimeline(parseInt(e.target.value, 10))}
                className="w-12 h-1"
              />
              <span>{Math.round(timelineProgress)}%</span>
            </div>
          </div>
        )}
      </div>

      {/* Active Panel */}
      {activePanel && (
        <div
          data-testid="config-panel"
          className="absolute top-24 right-4 bg-black bg-opacity-95 text-white rounded-lg backdrop-blur-sm max-w-sm max-h-96 overflow-y-auto"
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
                <div>
                  <label className="block text-xs mb-1">Telemetry:</label>
                  <select
                    value={telemetryLevel}
                    onChange={(e) => setTelemetryLevel(e.target.value)}
                    className="w-full bg-gray-700 text-white text-xs p-2 rounded"
                  >
                    <option value="off">Off</option>
                    <option value="basic">Basic</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs mb-1">Simulation Seed:</label>
                  <input
                    type="number"
                    min="1"
                    value={simulationSeed}
                    onChange={(e) => setSimulationSeed(sanitizeSeed(parseInt(e.target.value, 10)))}
                    className="w-full bg-gray-700 text-white text-xs p-2 rounded"
                  />
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
        </div>
      )}
    </div>
  );
};

export default NeuralNetwork3D;
