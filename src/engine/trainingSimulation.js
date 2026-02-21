export const TRAINING_PHASE = {
  IDLE: 'idle',
  FORWARD: 'forward',
  BACKWARD: 'backward',
  UPDATE: 'update'
};

export const getTrainingPhase = (progress, layersLength) => {
  if (!layersLength || !Number.isFinite(progress)) return TRAINING_PHASE.IDLE;
  if (progress < layersLength) return TRAINING_PHASE.FORWARD;
  if (progress < layersLength * 2) return TRAINING_PHASE.BACKWARD;
  return TRAINING_PHASE.UPDATE;
};

export const computeTrainingTick = ({
  previousLoss,
  previousAccuracy,
  learningRate,
  random
}) => {
  const rng = typeof random === 'function' ? random : Math.random;
  const decay = 1 - learningRate * 20;
  const lossNoise = (rng() - 0.5) * 0.01;
  const nextLoss = Math.max(0.001, previousLoss * decay + lossNoise);

  const improvement = (0.9 - previousAccuracy) * learningRate * 15;
  const accuracyNoise = (rng() - 0.5) * 0.005;
  const nextAccuracy = Math.min(0.92, previousAccuracy + improvement + accuracyNoise);

  return {
    loss: nextLoss,
    accuracy: nextAccuracy,
    complete: nextAccuracy >= 0.9
  };
};

export const GUIDED_PHASE_COPY = {
  [TRAINING_PHASE.IDLE]: {
    title: 'Ready',
    description: 'Choose a preset and start training to see learning dynamics in motion.'
  },
  [TRAINING_PHASE.FORWARD]: {
    title: 'Forward Pass',
    description: 'Signals move from input to output to produce predictions.'
  },
  [TRAINING_PHASE.BACKWARD]: {
    title: 'Backward Pass',
    description: 'Error gradients flow backward to identify which weights need correction.'
  },
  [TRAINING_PHASE.UPDATE]: {
    title: 'Weight Update',
    description: 'The optimizer adjusts weights to reduce future loss.'
  }
};
