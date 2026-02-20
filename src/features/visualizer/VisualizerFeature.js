import { useEffect, useRef } from 'react';
import { initVisualizer } from './visualizerRuntime';
import { visualizerTemplate } from './template';

const VisualizerFeature = () => {
  const rootRef = useRef(null);

  useEffect(() => {
    if (!rootRef.current) return;
    rootRef.current.innerHTML = visualizerTemplate;
    initVisualizer();
  }, []);

  return <div ref={rootRef} />;
};

export default VisualizerFeature;
