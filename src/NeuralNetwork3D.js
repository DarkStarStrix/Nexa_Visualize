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

const LOCAL_DRAFT_KEY = 'nexa.visualize.session.draft';
const SHARED_PREFIX = 'nexa.visualize.session.shared.';

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
  const optimizationHistoryRef = useRef([]);

  const [isTraining, setIsTraining] = useState(false);
  const [currentPhase, setCurrentPhase] = useState('idle');
  const [epoch, setEpoch] = useState(0);
  const [batch, setBatch] = useState(0);
  const [loss, setLoss] = useState(1.0);
  const [accuracy, setAccuracy] = useState(0.1);
  const [isTrainingComplete, setIsTrainingComplete] = useState(false);
  const [activePanel, setActivePanel] = useState(DEFAULT_SESSION_STATE.uiState.activePanel);
  const [cameraMode, setCameraMode] = useState(DEFAULT_SESSION_STATE.cameraMode);
  const [animationSpeed, setAnimationSpeed] = useState(DEFAULT_SESSION_STATE.uiState.animationSpeed);
  const [selectedModel, setSelectedModel] = useState(DEFAULT_SESSION_STATE.selectedModel);
  const [statusMessage, setStatusMessage] = useState('');

  // Architecture customization
  const [architecture, setArchitecture] = useState(DEFAULT_ARCHITECTURE);

  // Training parameters
  const [trainingParams, setTrainingParams] = useState(DEFAULT_TRAINING_PARAMS);

  // Manual camera state
  const [cameraState, setCameraState] = useState(DEFAULT_CAMERA_STATE);

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

  // Calculate optimal camera distance based on network size
  const calculateOptimalDistance = useCallback(() => {
    const networkWidth = layers.length * 3;
    const networkHeight = Math.max(...layers.map(layer => layer.gridSize[1])) * 2;
    const networkDepth = Math.max(...layers.map(layer => layer.gridSize[2])) * 2;
    const maxDimension = Math.max(networkWidth, networkHeight, networkDepth);
    return Math.max(12, maxDimension * 1.8);
  }, [layers]);

  // Create dense network connections like in the provided image
  const createNetwork = useCallback(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    // Clear existing network
    neuronsRef.current.forEach(layer => {
      layer.forEach(neuron => {
        scene.remove(neuron);
        neuron.geometry?.dispose();
        neuron.material?.dispose();
      });
    });
    connectionsRef.current.forEach(conn => {
      scene.remove(conn);
      conn.geometry?.dispose();
      conn.material?.dispose();
    });

    const neurons = [];
    const connections = [];
    const layerSpacing = 3.5;
    const neuronSize = 0.15;

    // Shared geometry for performance
    const neuronGeometry = new THREE.SphereGeometry(neuronSize, 8, 6);

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
              layerIndex * layerSpacing - (layers.length - 1) * layerSpacing / 2,
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

    // Create DENSE connections like in the reference image
    for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex++) {
      const currentLayer = neurons[layerIndex];
      const nextLayer = neurons[layerIndex + 1];

      // Connect every neuron to every neuron in next layer for dense connectivity
      currentLayer.forEach((neuron1, idx1) => {
        nextLayer.forEach((neuron2, idx2) => {
          // Skip some connections for very large networks to maintain performance
          const connectionDensity = Math.min(1.0, 50 / (currentLayer.length * nextLayer.length));
          if (Math.random() < connectionDensity) {
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

    neuronsRef.current = neurons;
    connectionsRef.current = connections;

    // Update camera distance for new network size
    const optimalDistance = calculateOptimalDistance();
    setCameraState(prev => ({
      ...prev,
      targetDistance: optimalDistance,
      distance: optimalDistance
    }));
  }, [layers, calculateOptimalDistance]);

  // Create loss landscape for optimization view
  const createLossLandscape = useCallback(() => {
    const scene = sceneRef.current;
    if (!scene || activePanel !== 'optimization') return;

    // Remove existing landscape
    const existingLandscape = scene.getObjectByName('lossLandscape');
    if (existingLandscape) scene.remove(existingLandscape);

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
                   0.3 * Math.random();
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
  }, [activePanel]);

  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "high-performance"
    });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Enhanced lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(15, 15, 10);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    scene.add(directionalLight);

    const fillLight = new THREE.DirectionalLight(0x4444ff, 0.3);
    fillLight.position.set(-10, 10, -10);
    scene.add(fillLight);

    // Grid floor
    const gridHelper = new THREE.GridHelper(40, 40, 0x444444, 0x222222);
    gridHelper.position.y = -8;
    scene.add(gridHelper);

    createNetwork();

    // Mouse controls
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };

    const onMouseDown = (event) => {
      if (cameraMode === 'manual') {
        isDragging = true;
        previousMousePosition = { x: event.clientX, y: event.clientY };
      }
    };

    const onMouseMove = (event) => {
      if (isDragging && cameraMode === 'manual') {
        const deltaX = (event.clientX - previousMousePosition.x) * 0.008;
        const deltaY = (event.clientY - previousMousePosition.y) * 0.008;

        setCameraState(prev => ({
          ...prev,
          targetAngleX: prev.targetAngleX + deltaX,
          targetAngleY: Math.max(-Math.PI/2.2, Math.min(Math.PI/2.2, prev.targetAngleY + deltaY))
        }));

        previousMousePosition = { x: event.clientX, y: event.clientY };
      }
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    const onWheel = (event) => {
      if (cameraMode === 'manual') {
        event.preventDefault();
        setCameraState(prev => ({
          ...prev,
          targetDistance: Math.max(5, Math.min(80, prev.targetDistance + event.deltaY * 0.05))
        }));
      }
    };

    mountRef.current.addEventListener('mousedown', onMouseDown);
    mountRef.current.addEventListener('mousemove', onMouseMove);
    mountRef.current.addEventListener('mouseup', onMouseUp);
    mountRef.current.addEventListener('wheel', onWheel, { passive: false });

    // Main animation loop - SIMPLIFIED AND RELIABLE
    const animate = (currentTime) => {
      const deltaTime = currentTime * 0.001;

      // Camera positioning
      if (cameraMode === 'auto') {
        const time = deltaTime * 0.3;
        const distance = cameraState.distance;
        camera.position.x = Math.cos(time) * distance;
        camera.position.z = Math.sin(time) * distance;
        camera.position.y = distance * 0.4;
        camera.lookAt(0, 0, 0);
      } else {
        // Smooth manual camera
        setCameraState(prev => {
          const lerpFactor = 0.08;
          const newDistance = THREE.MathUtils.lerp(prev.distance, prev.targetDistance, lerpFactor);
          const newAngleX = THREE.MathUtils.lerp(prev.angleX, prev.targetAngleX, lerpFactor);
          const newAngleY = THREE.MathUtils.lerp(prev.angleY, prev.targetAngleY, lerpFactor);

          camera.position.x = Math.cos(newAngleX) * Math.cos(newAngleY) * newDistance;
          camera.position.z = Math.sin(newAngleX) * Math.cos(newAngleY) * newDistance;
          camera.position.y = Math.sin(newAngleY) * newDistance;
          camera.lookAt(0, 0, 0);

          return { ...prev, distance: newDistance, angleX: newAngleX, angleY: newAngleY };
        });
      }

      // FIXED TRAINING ANIMATION - RELIABLE AND CONSISTENT
      if (isTraining && !isTrainingComplete) {
        const neurons = neuronsRef.current;
        const connections = connectionsRef.current;

        // Continuous animation progress
        animationProgressRef.current = (deltaTime * animationSpeed * 0.5) % (layers.length * 2 + 2);
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
        if (progress < layers.length) {
          for (let i = 0; i < layers.length; i++) {
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
        else if (progress < layers.length * 2) {
          const backProgress = progress - layers.length;
          for (let i = layers.length - 1; i >= 0; i--) {
            const reverseIdx = layers.length - 1 - i;
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
          const updateIntensity = Math.sin((progress - layers.length * 2) * Math.PI * 4) * 0.5 + 0.5;
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
      if (optimizationBallRef.current && isTraining) {
        const ball = optimizationBallRef.current;
        const time = deltaTime * 2;
        ball.position.x = Math.cos(time) * (3 - loss * 2.5);
        ball.position.z = Math.sin(time) * (3 - loss * 2.5);
        ball.position.y = -6 + loss * 2;
      }

      renderer.render(scene, camera);
      animationRef.current = requestAnimationFrame(animate);
    };

    animate(0);

    // Handle resize
    const handleResize = () => {
      if (!mountRef.current) return;
      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    };

    window.addEventListener('resize', handleResize);

  return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
        mountRef.current.removeEventListener('mousedown', onMouseDown);
        mountRef.current.removeEventListener('mousemove', onMouseMove);
        mountRef.current.removeEventListener('mouseup', onMouseUp);
        mountRef.current.removeEventListener('wheel', onWheel);
      }
      window.removeEventListener('resize', handleResize);

      // Cleanup
      neuronsRef.current.forEach(layer => {
        layer.forEach(neuron => {
          neuron.geometry?.dispose();
          neuron.material?.dispose();
        });
      });
      connectionsRef.current.forEach(conn => {
        conn.geometry?.dispose();
        conn.material?.dispose();
      });

      renderer.dispose();
    };
  }, [layers, createNetwork, cameraMode, animationSpeed, isTraining, isTrainingComplete, activePanel]);

  // Recreate network when architecture changes
  useEffect(() => {
    if (sceneRef.current) {
      createNetwork();
    }
  }, [createNetwork]);

  // Create loss landscape when optimization panel is active
  useEffect(() => {
    createLossLandscape();
  }, [createLossLandscape]);

  // Training simulation - STOPS AT 90%
  useEffect(() => {
    if (!isTraining || isTrainingComplete) return;

    const interval = setInterval(() => {
      setBatch(prev => prev + 1);

      setLoss(prev => {
        const decay = 1 - trainingParams.learningRate * 20;
        const noise = (Math.random() - 0.5) * 0.01;
        return Math.max(0.001, prev * decay + noise);
      });

      setAccuracy(prev => {
        const improvement = (0.9 - prev) * trainingParams.learningRate * 15;
        const noise = (Math.random() - 0.5) * 0.005;
        const newAcc = Math.min(0.92, prev + improvement + noise);

        // STOP AT 90% and flash blue
        if (newAcc >= 0.9 && !isTrainingComplete) {
          setIsTrainingComplete(true);
          setIsTraining(false);

          // Blue completion flash
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

        return newAcc;
      });

      if (batch % 10 === 0) {
        setEpoch(prev => prev + 1);
      }
    }, 200 / animationSpeed);

  return () => clearInterval(interval);
  }, [isTraining, isTrainingComplete, trainingParams, batch, animationSpeed]);

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

  const startTraining = () => {
    setIsTraining(true);
    setIsTrainingComplete(false);
    setEpoch(0);
    setBatch(0);
    setLoss(1.0);
    setAccuracy(0.1);
    animationProgressRef.current = 0;
  };

  const stopTraining = () => {
    setIsTraining(false);
  };

  const resetNetwork = () => {
    setIsTraining(false);
    setIsTrainingComplete(false);
    setEpoch(0);
    setBatch(0);
    setLoss(1.0);
    setAccuracy(0.1);
    animationProgressRef.current = 0;
  };

  const togglePanel = (panelName) => {
    setActivePanel(activePanel === panelName ? null : panelName);
  };

  const buildSession = useCallback(() => ({
    architecture,
    trainingParams,
    cameraMode,
    cameraState,
    selectedModel,
    uiState: {
      activePanel,
      animationSpeed
    }
  }), [architecture, trainingParams, cameraMode, cameraState, selectedModel, activePanel, animationSpeed]);

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
    setSelectedModel(session.selectedModel);
    setActivePanel(session.uiState.activePanel);
    setAnimationSpeed(session.uiState.animationSpeed);
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
      <div ref={mountRef} className="w-full h-full" />

      {/* Header */}
      <div className="absolute top-4 left-4 bg-black bg-opacity-90 text-white p-3 rounded-lg backdrop-blur-sm">
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
        {statusMessage && <div className="text-xs text-yellow-300 mt-1">{statusMessage}</div>}
      </div>

      {/* Controls */}
      <div className="absolute top-4 right-4 bg-black bg-opacity-90 text-white rounded-lg backdrop-blur-sm">
        <div className="flex">
          <button
            onClick={() => togglePanel('architecture')}
            className={`px-3 py-2 rounded-l text-xs font-medium transition-colors ${
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
              onChange={(e) => setSelectedModel(e.target.value)}
              className="bg-gray-700 text-white px-2 py-1 rounded text-xs"
            >
              <option value="Custom">Custom</option>
              <option value="CNN">CNN</option>
              <option value="Transformer">Transformer</option>
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
            <div className="text-lg font-bold text-blue-400">{batch}</div>
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
      {activePanel && (
        <div className="absolute top-24 right-4 bg-black bg-opacity-95 text-white rounded-lg backdrop-blur-sm max-w-sm max-h-96 overflow-y-auto">
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
        </div>
      )}
    </div>
  );
};

export default NeuralNetwork3D;

