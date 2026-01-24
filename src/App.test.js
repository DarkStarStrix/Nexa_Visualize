import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the application shell without crashing', () => {
  render(<App />);
  expect(screen.getByTestId('app-shell')).toBeInTheDocument();
});
