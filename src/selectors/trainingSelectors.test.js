import { getProgressPercent, getTelemetryLabel, getTrainingBadge } from './trainingSelectors';

describe('training selectors', () => {
  test('getTrainingBadge prioritizes complete state', () => {
    expect(getTrainingBadge({ isTraining: true, isTrainingComplete: true }).label).toBe('TRAINED');
    expect(getTrainingBadge({ isTraining: true, isTrainingComplete: false }).label).toBe('TRAINING');
    expect(getTrainingBadge({ isTraining: false, isTrainingComplete: false }).label).toBe('IDLE');
  });

  test('getProgressPercent clamps values', () => {
    expect(getProgressPercent(0.5)).toBe(50);
    expect(getProgressPercent(2)).toBe(100);
    expect(getProgressPercent(-1)).toBe(0);
  });

  test('getTelemetryLabel formats compact summary', () => {
    expect(getTelemetryLabel({ fps: 59, neuronCount: 24, connectionCount: 88 })).toBe('Perf 59 FPS • Graph 24N/88C');
  });
});
