import { buildDenseNetwork } from './networkBuilder';
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
});
