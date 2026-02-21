import { migrateSession, parseSessionPayload, SESSION_SCHEMA_VERSION } from './sessionSchema';
import { DEFAULT_SIMULATION_SEED } from '../engine/random';

describe('session schema migrations', () => {
  test('migrates a v2-like session into v3 defaults', () => {
    const migrated = migrateSession({
      version: 2,
      architecture: [
        { neurons: 4, activation: 'Linear', name: 'Input' },
        { neurons: 2, activation: 'Softmax', name: 'Output' }
      ],
      trainingParams: { learningRate: 0.02, batchSize: 16, epochs: 20, optimizer: 'SGD' },
      cameraMode: 'manual',
      cameraState: { distance: 10, angleX: 0, angleY: 0, targetDistance: 10, targetAngleX: 0, targetAngleY: 0 },
      selectedModel: 'Custom',
      uiState: {
        activePanel: 'parameters',
        animationSpeed: 1.5
      }
    });

    expect(migrated.version).toBe(SESSION_SCHEMA_VERSION);
    expect(migrated.uiState.guidedMode).toBe(false);
    expect(migrated.uiState.telemetryLevel).toBe('off');
    expect(migrated.simulation.seed).toBe(DEFAULT_SIMULATION_SEED);
  });

  test('parseSessionPayload sanitizes bad simulation seed and telemetry values', () => {
    const { session, error } = parseSessionPayload({
      version: 3,
      uiState: {
        activePanel: null,
        animationSpeed: 1,
        guidedMode: true,
        telemetryLevel: 'verbose'
      },
      simulation: {
        seed: -100
      }
    });

    expect(error).toBeNull();
    expect(session.uiState.telemetryLevel).toBe('off');
    expect(session.simulation.seed).toBe(100);
  });
});
