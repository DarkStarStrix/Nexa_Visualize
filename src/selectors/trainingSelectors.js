export const getTrainingBadge = ({ isTraining, isTrainingComplete }) => {
  if (isTrainingComplete) {
    return { label: 'TRAINED', className: 'bg-blue-600' };
  }
  if (isTraining) {
    return { label: 'TRAINING', className: 'bg-green-600' };
  }
  return { label: 'IDLE', className: 'bg-gray-600' };
};

export const getProgressPercent = (accuracy) => {
  if (!Number.isFinite(accuracy)) return 0;
  return Math.max(0, Math.min(100, accuracy * 100));
};

export const getTelemetryLabel = (telemetry) => {
  if (!telemetry) return 'Perf unavailable';
  return `Perf ${telemetry.fps} FPS • Graph ${telemetry.neuronCount}N/${telemetry.connectionCount}C`;
};
