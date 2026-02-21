import { buildDenseNetwork, calculateOptimalCameraDistance } from './networkBuilder';
import * as THREE from 'three';

jest.mock('three');

describe('network builder', () => {
  test('respects connection budget caps', () => {
    const scene = new THREE.Scene();
    const layers = [
      { neurons: 8, color: 0xffffff, gridSize: [4, 2, 1] },
      { neurons: 8, color: 0xffffff, gridSize: [4, 2, 1] },
      { neurons: 8, color: 0xffffff, gridSize: [4, 2, 1] }
    ];

    const { stats } = buildDenseNetwork({
      scene,
      layers,
      selectedModel: 'Custom',
      maxConnections: 10,
      random: () => 0
    });

    expect(stats.connectionCount).toBeLessThanOrEqual(10);
  });

  test('uses model-aware camera distance scaling', () => {
    const layers = [
      { neurons: 16, color: 0xffffff, gridSize: [4, 4, 1], name: 'Input' },
      { neurons: 16, color: 0xffffff, gridSize: [4, 4, 1], name: 'Hidden' },
      { neurons: 16, color: 0xffffff, gridSize: [4, 4, 1], name: 'Output' }
    ];

    const mlpDistance = calculateOptimalCameraDistance(layers, 'MLP');
    const transformerDistance = calculateOptimalCameraDistance(layers, 'Transformer');

    expect(transformerDistance).toBeGreaterThan(mlpDistance);
  });

  test('adds transformer skip connections between non-adjacent layers', () => {
    const scene = new THREE.Scene();
    const layers = [
      { neurons: 6, color: 0xffffff, gridSize: [3, 2, 1], name: 'Token Input' },
      { neurons: 6, color: 0xffffff, gridSize: [3, 2, 1], name: 'Self-Attention 1' },
      { neurons: 6, color: 0xffffff, gridSize: [3, 2, 1], name: 'Feed Forward' },
      { neurons: 6, color: 0xffffff, gridSize: [3, 2, 1], name: 'Output Head' }
    ];

    const { connections } = buildDenseNetwork({
      scene,
      layers,
      selectedModel: 'Transformer',
      maxConnections: 400,
      random: () => 0
    });

    const hasSkipConnection = connections.some((conn) => conn.userData.toLayer - conn.userData.fromLayer === 2);
    expect(hasSkipConnection).toBe(true);
  });
});
