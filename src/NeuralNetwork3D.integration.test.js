import { render, screen, within, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import NeuralNetwork3D from './NeuralNetwork3D';
import * as THREE from 'three';

jest.mock('three');

describe('NeuralNetwork3D integration tests', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  test('start/stop/reset training behavior updates UI state', async () => {
        render(<NeuralNetwork3D />);

    await userEvent.click(screen.getByRole('button', { name: /start training/i }));
    expect(screen.getByText('TRAINING')).toBeInTheDocument();

    await act(async () => {
      jest.advanceTimersByTime(1000);
    });

    const batchLabel = screen.getByText(/^Batch$/);
    expect(batchLabel.nextElementSibling.textContent).not.toBe('0');

    await userEvent.click(screen.getByRole('button', { name: /^stop$/i }));
    expect(screen.getByText('IDLE')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /reset/i }));
    expect(screen.getByText('1.0000')).toBeInTheDocument();
  });

  test('architecture selector changes layer activation', async () => {
        render(<NeuralNetwork3D />);

    await userEvent.click(screen.getByRole('button', { name: /architecture/i }));
    const panel = screen.getByTestId('config-panel');
    const selects = within(panel).getAllByRole('combobox');

    await userEvent.selectOptions(selects[1], 'Tanh');
    expect(selects[1]).toHaveValue('Tanh');
  });

  test('panel expansion and collapse toggles detail panel', async () => {
        render(<NeuralNetwork3D />);

    const parametersButton = screen.getByRole('button', { name: /parameters/i });
    await userEvent.click(parametersButton);
    expect(screen.getByTestId('config-panel')).toBeInTheDocument();

    await userEvent.click(parametersButton);
    expect(screen.queryByTestId('config-panel')).not.toBeInTheDocument();
  });

  test('three canvas mounts and unmounts without renderer leaks', () => {
    const startingCount = THREE.__mockRendererInstances.length;
    const { unmount } = render(<NeuralNetwork3D />);
    expect(screen.getByTestId('three-canvas-mount').querySelector('canvas')).toBeInTheDocument();

    unmount();

    const newInstances = THREE.__mockRendererInstances.slice(startingCount);
    expect(newInstances.length).toBeGreaterThan(0);
    newInstances.forEach((instance) => {
      expect(instance.dispose).toHaveBeenCalled();
    });
  });
});
