import { useEffect, useMemo, useState } from 'react';
import './App.css';
import { INITIAL_VISUALIZER_STATE } from './features/visualizer/constants';
import VisualizerCanvas from './features/visualizer/components/VisualizerCanvas';
import ArchitecturePanel from './features/visualizer/components/ArchitecturePanel';
import TrainingPanel from './features/visualizer/components/TrainingPanel';
import MetricsPanel from './features/visualizer/components/MetricsPanel';

function App() {
  const [state, setState] = useState(INITIAL_VISUALIZER_STATE);

  useEffect(() => {
    if (!state.isTraining) return undefined;

    const interval = setInterval(() => {
      setState((prev) => ({
        ...prev,
        batch: (prev.batch + 1) % 50,
        epoch: prev.batch >= 49 ? prev.epoch + 1 : prev.epoch,
        loss: Math.max(0.04, prev.loss * (0.985 - prev.animationSpeed * 0.001)),
        accuracy: Math.min(0.995, prev.accuracy + 0.003 * prev.animationSpeed)
      }));
    }, 120);

    return () => clearInterval(interval);
  }, [state.isTraining, state.animationSpeed]);

  const networkSummary = useMemo(() => {
    const fnn = state.architecture.fnn;
    return `${fnn.length} layers • ${fnn.reduce((sum, layer) => sum + layer.neurons, 0)} neurons`;
  }, [state.architecture]);

  const setTrainingParam = (key, value) => {
    setState((prev) => ({
      ...prev,
      trainingParams: {
        ...prev.trainingParams,
        [key]: value
      }
    }));
  };

  return (
    <main className="app-shell" data-testid="app-shell">
      <VisualizerCanvas
        archType={state.archType}
        animationSpeed={state.animationSpeed}
        isTraining={state.isTraining}
        cameraMode={state.cameraMode}
      />

      <header className="floating-panel header-panel">
        <h1>Nexa Visualize</h1>
        <p>{state.isTraining ? 'TRAINING' : 'IDLE'} • {networkSummary}</p>
      </header>

      <aside className="floating-panel controls-panel">
        <ArchitecturePanel
          archType={state.archType}
          cameraMode={state.cameraMode}
          onArchitectureChange={(archType) => setState((prev) => ({ ...prev, archType }))}
          onCameraModeChange={() => setState((prev) => ({
            ...prev,
            cameraMode: prev.cameraMode === 'auto' ? 'manual' : 'auto'
          }))}
        />
        <TrainingPanel
          isTraining={state.isTraining}
          animationSpeed={state.animationSpeed}
          trainingParams={state.trainingParams}
          onSpeedChange={(animationSpeed) => setState((prev) => ({ ...prev, animationSpeed }))}
          onTrainingParamChange={setTrainingParam}
          onStart={() => setState((prev) => ({ ...prev, isTraining: true }))}
          onStop={() => setState((prev) => ({ ...prev, isTraining: false }))}
          onReset={() => setState((prev) => ({
            ...prev,
            isTraining: false,
            epoch: 0,
            batch: 0,
            loss: 1,
            accuracy: 0.1
          }))}
        />
      </aside>

      <footer className="floating-panel metrics-panel-wrap">
        <MetricsPanel
          epoch={state.epoch}
          batch={state.batch}
          loss={state.loss}
          accuracy={state.accuracy}
        />
      </footer>
    </main>
  );
}

export default App;
