import * as THREE from 'three';

export const createSceneManager = (mountNode) => {
  const THEMES = {
    dark: {
      background: 0x0a0a0a,
      gridMain: 0x444444,
      gridSub: 0x222222,
      ambient: 0.4,
      directional: 0.8,
      fill: 0x4444ff
    },
    light: {
      background: 0xf4f7fb,
      gridMain: 0x9ca3af,
      gridSub: 0xd1d5db,
      ambient: 0.55,
      directional: 0.9,
      fill: 0x93c5fd
    }
  };

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(THEMES.dark.background);

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

  const ambientLight = new THREE.AmbientLight(0x404040, THEMES.dark.ambient);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(15, 15, 10);
  directionalLight.intensity = THEMES.dark.directional;
  directionalLight.castShadow = true;
  directionalLight.shadow.mapSize.width = 2048;
  directionalLight.shadow.mapSize.height = 2048;
  scene.add(directionalLight);

  const fillLight = new THREE.DirectionalLight(THEMES.dark.fill, 0.3);
  fillLight.position.set(-10, 10, -10);
  scene.add(fillLight);

  const gridHelper = new THREE.GridHelper(40, 40, THEMES.dark.gridMain, THEMES.dark.gridSub);
  gridHelper.position.y = -8;
  scene.add(gridHelper);

  const setTheme = (themeName = 'dark') => {
    const palette = THEMES[themeName] || THEMES.dark;
    scene.background = new THREE.Color(palette.background);
    ambientLight.intensity = palette.ambient;
    directionalLight.intensity = palette.directional;
    if (fillLight.color?.setHex) fillLight.color.setHex(palette.fill);

    const gridMaterial = gridHelper.material;
    if (Array.isArray(gridMaterial)) {
      if (gridMaterial[0]?.color?.setHex) gridMaterial[0].color.setHex(palette.gridMain);
      if (gridMaterial[1]?.color?.setHex) gridMaterial[1].color.setHex(palette.gridSub);
    } else if (gridMaterial?.color?.setHex) {
      gridMaterial.color.setHex(palette.gridMain);
    }
  };

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

  return { scene, camera, renderer, resize, dispose, setTheme };
};
