export const DEFAULT_SIMULATION_SEED = 2026;

const MODULUS = 2147483647;
const MULTIPLIER = 16807;

export const sanitizeSeed = (seed) => {
  if (!Number.isFinite(seed)) return DEFAULT_SIMULATION_SEED;
  const normalized = Math.floor(Math.abs(seed));
  return normalized % MODULUS || DEFAULT_SIMULATION_SEED;
};

export const createSeededRng = (seed) => {
  let state = sanitizeSeed(seed);
  return () => {
    state = (state * MULTIPLIER) % MODULUS;
    return (state - 1) / (MODULUS - 1);
  };
};
