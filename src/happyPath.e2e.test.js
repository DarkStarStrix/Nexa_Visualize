import { render, screen, within, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';

jest.mock('three');

test('happy path: open app -> configure -> start training -> observe metric updates', async () => {
    jest.useFakeTimers();

  render(<App />);
  expect(screen.getByText('Nexa Visualize')).toBeInTheDocument();

  await userEvent.click(screen.getByRole('button', { name: /parameters/i }));
  const paramsPanel = screen.getByTestId('config-panel');
  const optimizerSelect = within(paramsPanel).getByDisplayValue('Adam');
  await userEvent.selectOptions(optimizerSelect, 'SGD');
  expect(optimizerSelect).toHaveValue('SGD');

  await userEvent.click(screen.getByRole('button', { name: /start training/i }));
  expect(screen.getByText('TRAINING')).toBeInTheDocument();

  const batchValue = screen.getByText(/^Batch$/).nextElementSibling;
  const beforeBatch = batchValue.textContent;

  await act(async () => {
    jest.advanceTimersByTime(1200);
  });

  expect(batchValue.textContent).not.toBe(beforeBatch);

  jest.clearAllTimers();
  jest.useRealTimers();
});
