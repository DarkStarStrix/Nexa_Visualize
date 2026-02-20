function TrainingPanel({
  isTraining,
  animationSpeed,
  trainingParams,
  onSpeedChange,
  onTrainingParamChange,
  onStart,
  onStop,
  onReset
}) {
  return (
    <section className="panel-card" aria-label="Training controls">
      <h2>Training</h2>
      <label>
        Animation speed ({animationSpeed.toFixed(1)}x)
        <input
          type="range"
          min="0.5"
          max="3"
          step="0.1"
          value={animationSpeed}
          onChange={(event) => onSpeedChange(Number(event.target.value))}
        />
      </label>
      <label>
        Learning rate
        <input
          type="number"
          step="0.001"
          min="0.001"
          value={trainingParams.learningRate}
          onChange={(event) => onTrainingParamChange('learningRate', Number(event.target.value))}
        />
      </label>
      <label>
        Batch size
        <input
          type="number"
          min="1"
          value={trainingParams.batchSize}
          onChange={(event) => onTrainingParamChange('batchSize', Number(event.target.value))}
        />
      </label>
      <div className="button-row">
        <button type="button" onClick={onStart} disabled={isTraining}>Start</button>
        <button type="button" onClick={onStop} disabled={!isTraining}>Stop</button>
        <button type="button" onClick={onReset}>Reset</button>
      </div>
    </section>
  );
}

export default TrainingPanel;
