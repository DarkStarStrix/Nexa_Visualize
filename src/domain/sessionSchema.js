import { DEFAULT_SIMULATION_SEED, sanitizeSeed } from '../engine/random';

export const SESSION_SCHEMA_VERSION = 3;

export const DEFAULT_ARCHITECTURE = [
  { neurons: 4, activation: 'Linear', name: 'Input' },
  { neurons: 8, activation: 'ReLU', name: 'Hidden 1' },
  { neurons: 6, activation: 'ReLU', name: 'Hidden 2' },
  { neurons: 3, activation: 'Softmax', name: 'Output' }
];

export const DEFAULT_TRAINING_PARAMS = {
  learningRate: 0.01,
  batchSize: 32,
  epochs: 100,
  optimizer: 'Adam',
  lossFunction: 'CrossEntropy'
};

export const DEFAULT_CAMERA_STATE = {
  distance: 15,
  angleX: 0,
  angleY: 0.3,
  targetDistance: 15,
  targetAngleX: 0,
  targetAngleY: 0.3
};

export const DEFAULT_SESSION_STATE = {
  version: SESSION_SCHEMA_VERSION,
  architecture: DEFAULT_ARCHITECTURE,
  trainingParams: DEFAULT_TRAINING_PARAMS,
  cameraMode: 'auto',
  cameraState: DEFAULT_CAMERA_STATE,
  selectedModel: 'Custom',
  uiState: {
    activePanel: null,
    animationSpeed: 1,
    datasetName: 'Spiral',
    selectedScenario: 'stable',
    presentationMode: false
  },
  simulation: {
    seed: DEFAULT_SIMULATION_SEED
  }
};

const isLayerValid = (layer) => {
  return layer
    && typeof layer.name === 'string'
    && typeof layer.activation === 'string'
    && Number.isInteger(layer.neurons)
    && layer.neurons > 0
    && layer.neurons <= 64;
};

const sanitizeArchitecture = (architecture) => {
  if (!Array.isArray(architecture)) return DEFAULT_ARCHITECTURE;
  const sanitized = architecture.filter(isLayerValid);
  return sanitized.length >= 2 ? sanitized : DEFAULT_ARCHITECTURE;
};

const sanitizeTrainingParams = (params) => {
  const safe = { ...DEFAULT_TRAINING_PARAMS, ...(params || {}) };
  safe.learningRate = Number.isFinite(safe.learningRate) ? safe.learningRate : DEFAULT_TRAINING_PARAMS.learningRate;
  safe.batchSize = Number.isFinite(safe.batchSize) ? safe.batchSize : DEFAULT_TRAINING_PARAMS.batchSize;
  safe.epochs = Number.isFinite(safe.epochs) ? safe.epochs : DEFAULT_TRAINING_PARAMS.epochs;
  return safe;
};

const sanitizeDatasetName = (datasetName) => {
  const allowed = ['Spiral', 'Moons', 'Blobs', 'ImagePatches'];
  return allowed.includes(datasetName) ? datasetName : 'Spiral';
};

const sanitizeScenario = (scenario) => {
  const allowed = ['underfit', 'overfit', 'stable', 'high_lr'];
  return allowed.includes(scenario) ? scenario : 'stable';
};

const sanitizeCameraState = (cameraState) => {
  const safe = { ...DEFAULT_CAMERA_STATE, ...(cameraState || {}) };
  Object.keys(DEFAULT_CAMERA_STATE).forEach((key) => {
    if (!Number.isFinite(safe[key])) {
      safe[key] = DEFAULT_CAMERA_STATE[key];
    }
  });
  return safe;
};

const migrateV1ToV2 = (session) => ({
  ...session,
  version: 2,
  selectedModel: session.selectedModel || 'Custom',
  uiState: {
    activePanel: session.activePanel ?? null,
    animationSpeed: session.animationSpeed ?? 1
  }
});

const migrateV2ToV3 = (session) => ({
  ...session,
  version: 3,
  uiState: {
    ...session.uiState,
    datasetName: sanitizeDatasetName(session.uiState?.datasetName),
    selectedScenario: sanitizeScenario(session.uiState?.selectedScenario),
    presentationMode: Boolean(session.uiState?.presentationMode)
  },
  simulation: {
    seed: sanitizeSeed(session.simulation?.seed)
  }
});

export const migrateSession = (session) => {
  if (!session || typeof session !== 'object') {
    return DEFAULT_SESSION_STATE;
  }

  const version = Number(session.version) || 1;
  let migrated = { ...session };

  if (version < 2) {
    migrated = migrateV1ToV2(migrated);
  }
  if (version < 3) {
    migrated = migrateV2ToV3(migrated);
  }

  return {
    version: SESSION_SCHEMA_VERSION,
    architecture: sanitizeArchitecture(migrated.architecture),
    trainingParams: sanitizeTrainingParams(migrated.trainingParams),
    cameraMode: migrated.cameraMode === 'manual' ? 'manual' : 'auto',
    cameraState: sanitizeCameraState(migrated.cameraState),
    selectedModel: typeof migrated.selectedModel === 'string' ? migrated.selectedModel : 'Custom',
    uiState: {
      activePanel: typeof migrated.uiState?.activePanel === 'string' ? migrated.uiState.activePanel : null,
      animationSpeed: Number.isFinite(migrated.uiState?.animationSpeed)
        ? migrated.uiState.animationSpeed
        : 1,
      datasetName: sanitizeDatasetName(migrated.uiState?.datasetName),
      selectedScenario: sanitizeScenario(migrated.uiState?.selectedScenario),
      presentationMode: Boolean(migrated.uiState?.presentationMode)
    },
    simulation: {
      seed: sanitizeSeed(migrated.simulation?.seed)
    }
  };
};

export const parseSessionPayload = (rawPayload) => {
  try {
    const parsed = typeof rawPayload === 'string' ? JSON.parse(rawPayload) : rawPayload;
    return { session: migrateSession(parsed), error: null };
  } catch (error) {
    return { session: null, error: 'Unable to parse session JSON.' };
  }
};

export const serializeSession = (sessionState) => {
  return JSON.stringify(migrateSession(sessionState), null, 2);
};

export const generateSessionId = () => Math.random().toString(36).slice(2, 10);
