export const ARCHITECTURE_OPTIONS = [
  { value: 'fnn', label: 'Feedforward Network' },
  { value: 'transformer', label: 'Transformer' },
  { value: 'cnn', label: 'Convolutional Network' },
  { value: 'operator', label: 'Neural Operator' },
  { value: 'moe', label: 'Mixture of Experts' },
  { value: 'autoencoder', label: 'Autoencoder' }
];

export const INITIAL_VISUALIZER_STATE = {
  isTraining: false,
  cameraMode: 'auto',
  animationSpeed: 1,
  epoch: 0,
  batch: 0,
  loss: 1,
  accuracy: 0.1,
  archType: 'fnn',
  architecture: {
    fnn: [
      { neurons: 4, activation: 'Linear', name: 'Input' },
      { neurons: 8, activation: 'ReLU', name: 'Hidden 1' },
      { neurons: 6, activation: 'ReLU', name: 'Hidden 2' },
      { neurons: 3, activation: 'Softmax', name: 'Output' }
    ]
  },
  trainingParams: {
    learningRate: 0.01,
    batchSize: 32,
    epochs: 100,
    optimizer: 'Adam',
    lossFunction: 'CrossEntropy'
  }
};
