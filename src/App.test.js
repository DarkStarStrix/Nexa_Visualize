import { render, screen } from '@testing-library/react';
import App from './App';

jest.mock('./NeuralNetwork3D', () => () => <div>Nexa Visualize</div>);

test('renders the visualization shell', () => {
  render(<App />);
  expect(screen.getByText('Nexa Visualize')).toBeInTheDocument();
});
