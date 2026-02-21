const rendererInstances = [];

class MockObject3D {
  constructor() {
    this.children = [];
    this.position = {
      x: 0,
      y: 0,
      z: 0,
      set: (x, y, z) => {
        this.position.x = x;
        this.position.y = y;
        this.position.z = z;
      },
      clone: () => ({ ...this.position, clone: this.position.clone })
    };
    this.rotation = { x: 0, y: 0, z: 0 };
    this.scale = { setScalar: jest.fn(), set: jest.fn() };
    this.userData = {};
    this.name = '';
  }
}

class Scene extends MockObject3D {
  add(obj) {
    this.children.push(obj);
  }
  remove(obj) {
    this.children = this.children.filter((child) => child !== obj);
  }
  getObjectByName(name) {
    return this.children.find((child) => child.name === name);
  }
}

class PerspectiveCamera extends MockObject3D {
  constructor() {
    super();
    this.aspect = 1;
  }
  updateProjectionMatrix() {}
  lookAt() {}
}

class WebGLRenderer {
  constructor() {
    this.domElement = document.createElement('canvas');
    this.domElement.setAttribute('data-testid', 'three-renderer-canvas');
    this.setSize = jest.fn();
    this.setPixelRatio = jest.fn();
    this.render = jest.fn();
    this.dispose = jest.fn();
    this.shadowMap = { enabled: false, type: null };
    rendererInstances.push(this);
  }
}

class AmbientLight extends MockObject3D {}
class DirectionalLight extends MockObject3D {
  constructor() {
    super();
    this.color = { setHex: jest.fn() };
    this.intensity = 1;
    this.shadow = { mapSize: { width: 0, height: 0 } };
  }
}
class Group extends MockObject3D {
  add(obj) {
    this.children.push(obj);
  }
}
class Geometry {
  dispose = jest.fn();
  setFromPoints() {
    return this;
  }
}
class PlaneGeometry extends Geometry {
  constructor() {
    super();
    this.attributes = { position: { array: new Array(300).fill(0), needsUpdate: false } };
  }
  computeVertexNormals() {}
}
class SphereGeometry extends Geometry {}
class BoxGeometry extends Geometry {}
class CylinderGeometry extends Geometry {}
class ConeGeometry extends Geometry {}
class BufferGeometry extends Geometry {}

class Material {
  constructor() {
    this.dispose = jest.fn();
    this.opacity = 1;
    this.color = { setHex: jest.fn() };
    this.emissive = { setHex: jest.fn(), multiplyScalar: jest.fn() };
  }
}
class MeshPhongMaterial extends Material {}
class LineBasicMaterial extends Material {}
class SpriteMaterial extends Material {}
class CanvasTexture {
  constructor() {
    this.dispose = jest.fn();
    this.needsUpdate = false;
  }
}

class Mesh extends MockObject3D {
  constructor(geometry, material) {
    super();
    this.geometry = geometry;
    this.material = material;
  }
}
class Line extends Mesh {}
class Sprite extends Mesh {
  constructor(material) {
    super(null, material);
  }
}

class GridHelper extends MockObject3D {
  constructor() {
    super();
    this.material = { color: { setHex: jest.fn() } };
  }
}
class Color {
  constructor(hex) { this.hex = hex; }
  setHex(hex) { this.hex = hex; }
}
const MathUtils = { lerp: (a, b, t) => a + (b - a) * t };
const PCFSoftShadowMap = 'PCFSoftShadowMap';

module.exports = {
  Scene,
  PerspectiveCamera,
  WebGLRenderer,
  AmbientLight,
  DirectionalLight,
  SphereGeometry,
  BoxGeometry,
  CylinderGeometry,
  ConeGeometry,
  MeshPhongMaterial,
  SpriteMaterial,
  Mesh,
  Sprite,
  CanvasTexture,
  BufferGeometry,
  LineBasicMaterial,
  Line,
  PlaneGeometry,
  Group,
  GridHelper,
  Color,
  MathUtils,
  PCFSoftShadowMap,
  __mockRendererInstances: rendererInstances
};
