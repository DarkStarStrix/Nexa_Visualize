import { useEffect, useRef } from 'react';
import * as THREE from 'three';

const ARCH_MESH_COLORS = {
  fnn: 0x22d3ee,
  transformer: 0xa855f7,
  cnn: 0x3b82f6,
  operator: 0xf59e0b,
  moe: 0x22c55e,
  autoencoder: 0xef4444
};

function VisualizerCanvas({ archType, animationSpeed, isTraining, cameraMode }) {
  const canvasRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const meshRef = useRef(null);
  const cameraRef = useRef(null);
  const frameRef = useRef(null);
  const animationSpeedRef = useRef(animationSpeed);
  const isTrainingRef = useRef(isTraining);
  const cameraModeRef = useRef(cameraMode);

  useEffect(() => {
    animationSpeedRef.current = animationSpeed;
  }, [animationSpeed]);

  useEffect(() => {
    isTrainingRef.current = isTraining;
  }, [isTraining]);

  useEffect(() => {
    cameraModeRef.current = cameraMode;
  }, [cameraMode]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);

    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 1200);
    camera.position.set(0, 0, 6);

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    } catch (error) {
      return undefined;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.1);
    directionalLight.position.set(4, 5, 6);
    scene.add(directionalLight);

    const geometry = new THREE.TorusKnotGeometry(1.2, 0.35, 128, 16);
    const material = new THREE.MeshStandardMaterial({
      color: ARCH_MESH_COLORS.fnn,
      roughness: 0.35,
      metalness: 0.4
    });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    const resize = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };

    const animate = () => {
      const speedFactor = isTrainingRef.current ? animationSpeedRef.current : 0.2;
      mesh.rotation.x += 0.005 * speedFactor;
      mesh.rotation.y += 0.0075 * speedFactor;
      if (cameraModeRef.current === 'manual') {
        camera.position.x = Math.sin(Date.now() * 0.0002) * 0.2;
      } else {
        camera.position.x = 0;
      }
      renderer.render(scene, camera);
      frameRef.current = requestAnimationFrame(animate);
    };

    window.addEventListener('resize', resize);
    animate();

    sceneRef.current = scene;
    rendererRef.current = renderer;
    meshRef.current = mesh;
    cameraRef.current = camera;

    return () => {
      window.removeEventListener('resize', resize);
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
      geometry.dispose();
      material.dispose();
      renderer.dispose();
    };
  }, []);

  useEffect(() => {
    if (meshRef.current) {
      meshRef.current.material.color.setHex(ARCH_MESH_COLORS[archType]);
    }
  }, [archType]);

  return <canvas ref={canvasRef} className="visualizer-canvas" aria-label="3D visualizer canvas" />;
}

export default VisualizerCanvas;
