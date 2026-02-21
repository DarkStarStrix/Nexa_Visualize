import { createSeededRng, DEFAULT_SIMULATION_SEED, sanitizeSeed } from './random';

describe('engine random', () => {
  test('sanitizeSeed falls back to default for invalid values', () => {
    expect(sanitizeSeed(undefined)).toBe(DEFAULT_SIMULATION_SEED);
    expect(sanitizeSeed(NaN)).toBe(DEFAULT_SIMULATION_SEED);
  });

  test('createSeededRng is deterministic for a given seed', () => {
    const rngA = createSeededRng(1337);
    const rngB = createSeededRng(1337);
    const seqA = [rngA(), rngA(), rngA(), rngA()];
    const seqB = [rngB(), rngB(), rngB(), rngB()];
    expect(seqA).toEqual(seqB);
  });
});
