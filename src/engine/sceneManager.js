import * as THREE from 'three';

export const createSceneManager = (mountNode) => {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0a);

  const camera = new THREE.PerspectiveCamera(
    75,
    mountNode.clientWidth / mountNode.clientHeight,
    0.1,
    1000
  );

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: 'high-performance'
  });
  renderer.setSize(mountNode.clientWidth, mountNode.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  mountNode.appendChild(renderer.domElement);

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

  const gridHelper = new THREE.GridHelper(40, 40, 0x444444, 0x222222);
  gridHelper.position.y = -8;
  scene.add(gridHelper);

  const resize = () => {
    camera.aspect = mountNode.clientWidth / mountNode.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(mountNode.clientWidth, mountNode.clientHeight);
  };

  const dispose = () => {
    if (mountNode && renderer.domElement && mountNode.contains(renderer.domElement)) {
      mountNode.removeChild(renderer.domElement);
    }
    renderer.dispose();
  };

  return { scene, camera, renderer, resize, dispose };
};
