import { render, screen } from '@testing-library/react';
import App from './App';

jest.mock('./features/visualizer/VisualizerFeature', () => () => (
  <div data-testid="visualizer-feature" />
));

test('renders the visualizer feature entrypoint', () => {
  render(<App />);
  expect(screen.getByTestId('visualizer-feature')).toBeInTheDocument();
});
