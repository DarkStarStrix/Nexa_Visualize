export const DATASET_PRESETS = {
  Spiral: {
    difficulty: 1.2,
    classes: ['Inner Spiral', 'Outer Spiral', 'Boundary'],
    summary: 'Non-linear toy dataset that stresses representation learning.',
    failureCases: ['Boundary curl at theta=2.2', 'Outer arm overlap', 'Sparse center region']
  },
  Moons: {
    difficulty: 1.0,
    classes: ['Moon A', 'Moon B', 'Overlap'],
    summary: 'Good for quick binary-separation demos with some overlap noise.',
    failureCases: ['Mid-overlap ambiguity', 'Low-density tail', 'Noisy cusp']
  },
  Blobs: {
    difficulty: 0.8,
    classes: ['Cluster A', 'Cluster B', 'Cluster C'],
    summary: 'Easier clustered classes, useful for teaching baseline behavior.',
    failureCases: ['Edge outlier sample', 'Tiny minority pocket', 'Cluster drift near axis']
  },
  ImagePatches: {
    difficulty: 1.35,
    classes: ['Texture', 'Shape', 'Background'],
    summary: 'Small synthetic image patches to emulate visual pattern learning.',
    failureCases: ['Low-contrast patch', 'Texture aliasing', 'Shadowed patch']
  }
};

export const SCENARIO_PRESETS = {
  underfit: {
    label: 'Underfitting',
    description: 'Small network + high regularization behavior.',
    model: 'MLP',
    learningRate: 0.003,
    batchSize: 128
  },
  overfit: {
    label: 'Overfitting',
    description: 'Higher capacity and aggressive updates.',
    model: 'Transformer',
    learningRate: 0.03,
    batchSize: 16
  },
  stable: {
    label: 'Good Generalization',
    description: 'Balanced settings for smooth convergence.',
    model: 'CNN',
    learningRate: 0.01,
    batchSize: 32
  },
  high_lr: {
    label: 'Too High LR',
    description: 'Intentionally unstable to show oscillations.',
    model: 'MoE',
    learningRate: 0.08,
    batchSize: 32
  }
};

export const ASSISTANT_GOALS = [
  'Improve Accuracy',
  'Reduce Complexity',
  'Stabilize Training',
  'Converge Faster'
];

export const MODEL_COMPARISON_BIAS = {
  Custom: 0,
  MLP: -0.015,
  CNN: 0.02,
  Transformer: 0.03,
  'Neural Operator': 0.018,
  MoE: 0.01,
  RNN: 0.008,
  LSTM: 0.015,
  GRU: 0.012,
  GAN: -0.004,
  Autoencoder: 0.005
};
