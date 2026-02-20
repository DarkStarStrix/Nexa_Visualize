import { ARCHITECTURE_OPTIONS } from '../constants';

function ArchitecturePanel({ archType, onArchitectureChange, cameraMode, onCameraModeChange }) {
  return (
    <section className="panel-card" aria-label="Architecture controls">
      <h2>Architecture</h2>
      <label>
        Model type
        <select value={archType} onChange={(event) => onArchitectureChange(event.target.value)}>
          {ARCHITECTURE_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
      <div className="toggle-row">
        <span>Camera</span>
        <button type="button" onClick={onCameraModeChange}>
          {cameraMode === 'auto' ? 'Auto' : 'Manual'}
        </button>
      </div>
    </section>
  );
}

export default ArchitecturePanel;
