import React, { useEffect, useMemo, useState } from 'react';
import NeuralNetwork3D from './NeuralNetwork3D';

const SESSION_ROUTE_PREFIX = '/session/';

const getRouteState = () => {
  const path = window.location.pathname;
  if (path.startsWith(SESSION_ROUTE_PREFIX)) {
    return {
      sessionId: path.replace(SESSION_ROUTE_PREFIX, '') || null
    };
  }

  return { sessionId: null };
};

function App() {
  const [routeState, setRouteState] = useState(getRouteState());

  useEffect(() => {
    const onPopState = () => setRouteState(getRouteState());
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  const onSessionRouteChange = useMemo(() => (sessionId) => {
    const path = sessionId ? `${SESSION_ROUTE_PREFIX}${sessionId}` : '/';
    window.history.pushState({}, '', path);
    setRouteState(getRouteState());
  }, []);

  return (
    <NeuralNetwork3D
      sessionId={routeState.sessionId}
      onSessionRouteChange={onSessionRouteChange}
    />
  );
}

export default App;
