import { act, renderHook } from '@testing-library/react';
import {
  trainingStateReducer,
  initialTrainingState,
  architectureReducer,
  initialArchitecture,
  trainingParamsReducer,
  initialTrainingParams,
  useTrainingState
} from './state';

describe('app state reducers and hooks', () => {
  test('trainingStateReducer handles start/stop/reset and tick updates', () => {
    const started = trainingStateReducer(initialTrainingState, { type: 'START' });
    expect(started.isTraining).toBe(true);
    expect(started.loss).toBe(1);

    const ticked = trainingStateReducer(started, {
      type: 'TICK',
      payload: { epoch: 3, batch: 9, loss: 0.42, accuracy: 0.71 }
    });
    expect(ticked).toMatchObject({ epoch: 3, batch: 9, loss: 0.42, accuracy: 0.71 });

    const stopped = trainingStateReducer(ticked, { type: 'STOP' });
    expect(stopped.isTraining).toBe(false);

    const completed = trainingStateReducer(ticked, { type: 'COMPLETE' });
    expect(completed.isTrainingComplete).toBe(true);

    const reset = trainingStateReducer(completed, { type: 'RESET' });
    expect(reset).toEqual(initialTrainingState);
  });

  test('architectureReducer handles add, remove, and layer updates', () => {
    const added = architectureReducer(initialArchitecture, { type: 'ADD_LAYER' });
    expect(added).toHaveLength(initialArchitecture.length + 1);
    expect(added.at(-2).name).toMatch(/Hidden/);

    const updated = architectureReducer(added, {
      type: 'UPDATE_LAYER',
      payload: { index: 1, field: 'activation', value: 'Tanh' }
    });
    expect(updated[1].activation).toBe('Tanh');

    const removed = architectureReducer(updated, { type: 'REMOVE_LAYER', payload: { index: 1 } });
    expect(removed).toHaveLength(updated.length - 1);
  });

  test('trainingParamsReducer updates individual training params', () => {
    const next = trainingParamsReducer(initialTrainingParams, {
      type: 'UPDATE_PARAM',
      payload: { field: 'optimizer', value: 'SGD' }
    });

    expect(next.optimizer).toBe('SGD');
    expect(next.learningRate).toBe(initialTrainingParams.learningRate);
  });

  test('useTrainingState hook exposes training controls', () => {
    const { result } = renderHook(() => useTrainingState());

    act(() => {
      result.current.startTraining();
    });
    expect(result.current.state.isTraining).toBe(true);

    act(() => {
      result.current.stopTraining();
    });
    expect(result.current.state.isTraining).toBe(false);

    act(() => {
      result.current.resetTraining();
    });
    expect(result.current.state).toEqual(initialTrainingState);
  });
});
