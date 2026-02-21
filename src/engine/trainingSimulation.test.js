import { computeTrainingTick, getTrainingPhase, TRAINING_PHASE } from './trainingSimulation';
import { createSeededRng } from './random';

describe('training simulation engine', () => {
  test('phase mapping is stable around boundaries', () => {
    expect(getTrainingPhase(0.2, 4)).toBe(TRAINING_PHASE.FORWARD);
    expect(getTrainingPhase(4.1, 4)).toBe(TRAINING_PHASE.BACKWARD);
    expect(getTrainingPhase(8.2, 4)).toBe(TRAINING_PHASE.UPDATE);
    expect(getTrainingPhase(Number.NaN, 4)).toBe(TRAINING_PHASE.IDLE);
  });

  test('computeTrainingTick is deterministic with seeded RNG', () => {
    const rngA = createSeededRng(42);
    const rngB = createSeededRng(42);

    const tickA = computeTrainingTick({
      previousLoss: 1,
      previousAccuracy: 0.1,
      learningRate: 0.01,
      random: rngA
    });
    const tickB = computeTrainingTick({
      previousLoss: 1,
      previousAccuracy: 0.1,
      learningRate: 0.01,
      random: rngB
    });

    expect(tickA).toEqual(tickB);
    expect(tickA.loss).toBeGreaterThanOrEqual(0.001);
    expect(tickA.accuracy).toBeLessThanOrEqual(0.92);
  });
});
