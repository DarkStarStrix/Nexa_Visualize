import { useReducer, useCallback } from 'react';

export const initialTrainingState = {
  isTraining: false,
  isTrainingComplete: false,
  epoch: 0,
  batch: 0,
  loss: 1,
  accuracy: 0.1
};

export function trainingStateReducer(state, action) {
  switch (action.type) {
    case 'START':
      return { ...initialTrainingState, isTraining: true };
    case 'STOP':
      return { ...state, isTraining: false };
    case 'RESET':
      return { ...initialTrainingState };
    case 'TICK':
      return {
        ...state,
        epoch: action.payload.epoch,
        batch: action.payload.batch,
        loss: action.payload.loss,
        accuracy: action.payload.accuracy
      };
    case 'COMPLETE':
      return { ...state, isTraining: false, isTrainingComplete: true };
    default:
      return state;
  }
}

export const initialArchitecture = [
  { neurons: 4, activation: 'Linear', name: 'Input' },
  { neurons: 8, activation: 'ReLU', name: 'Hidden 1' },
  { neurons: 6, activation: 'ReLU', name: 'Hidden 2' },
  { neurons: 3, activation: 'Softmax', name: 'Output' }
];

export function architectureReducer(state, action) {
  switch (action.type) {
    case 'ADD_LAYER': {
      const layerCount = state.length - 1;
      return [
        ...state.slice(0, -1),
        { neurons: 8, activation: 'ReLU', name: `Hidden ${layerCount}` },
        state[state.length - 1]
      ];
    }
    case 'REMOVE_LAYER':
      return state.filter((_, index) => index !== action.payload.index);
    case 'UPDATE_LAYER':
      return state.map((layer, index) =>
        index === action.payload.index
          ? { ...layer, [action.payload.field]: action.payload.value }
          : layer
      );
    default:
      return state;
  }
}

export const initialTrainingParams = {
  learningRate: 0.01,
  batchSize: 32,
  epochs: 100,
  optimizer: 'Adam',
  lossFunction: 'CrossEntropy'
};

export function trainingParamsReducer(state, action) {
  if (action.type === 'UPDATE_PARAM') {
    return { ...state, [action.payload.field]: action.payload.value };
  }

  return state;
}

export function useTrainingState() {
  const [state, dispatch] = useReducer(trainingStateReducer, initialTrainingState);

  const startTraining = useCallback(() => dispatch({ type: 'START' }), []);
  const stopTraining = useCallback(() => dispatch({ type: 'STOP' }), []);
  const resetTraining = useCallback(() => dispatch({ type: 'RESET' }), []);

  return {
    state,
    dispatch,
    startTraining,
    stopTraining,
    resetTraining
  };
}
