import { createSceneManager } from './sceneManager';

jest.mock('three');

describe('scene manager', () => {
  test('initializes and disposes renderer cleanly', () => {
    const mountNode = document.createElement('div');
    Object.defineProperty(mountNode, 'clientWidth', { value: 800 });
    Object.defineProperty(mountNode, 'clientHeight', { value: 600 });
    document.body.appendChild(mountNode);

    const manager = createSceneManager(mountNode);
    expect(mountNode.querySelector('[data-testid="three-renderer-canvas"]')).toBeTruthy();

    manager.resize();
    manager.dispose();

    expect(manager.renderer.dispose).toHaveBeenCalled();
    document.body.removeChild(mountNode);
  });
});
