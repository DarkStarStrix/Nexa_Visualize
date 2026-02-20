function MetricsPanel({ epoch, batch, loss, accuracy }) {
  return (
    <section className="panel-card" aria-label="Training metrics">
      <h2>Metrics</h2>
      <ul className="metrics-list">
        <li><span>Epoch</span><strong>{epoch}</strong></li>
        <li><span>Batch</span><strong>{batch}</strong></li>
        <li><span>Loss</span><strong>{loss.toFixed(4)}</strong></li>
        <li><span>Accuracy</span><strong>{(accuracy * 100).toFixed(1)}%</strong></li>
      </ul>
    </section>
  );
}

export default MetricsPanel;
