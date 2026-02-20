import { useEffect, useMemo, useState } from 'react';
import './App.css';
import NeuralNetwork3D from './NeuralNetwork3D';
import { MODELS_MARKDOWN, SOURCES_MARKDOWN } from './content/docsContent';

const NAV_ROUTES = [
  { path: '/', label: 'Home' },
  { path: '/visualizer', label: 'Visualizer' },
  { path: '/docs-models', label: 'Docs/Models' },
  { path: '/about', label: 'About' }
];

const LEGAL_ROUTES = [
  { path: '/license', label: 'License', href: '/LICENSE' },
  { path: '/code-of-conduct', label: 'Code of Conduct', href: '/CODE_OF_CONDUCT.md' },
  { path: '/contributing', label: 'Contributing', href: '/CONTRIBUTING.md' },
  { path: '/privacy', label: 'Privacy Notice' }
];

const ROUTE_CONTENT = {
  '/license': {
    title: 'License',
    body: 'This project is distributed under the MIT License. See the linked LICENSE file for full terms and attribution requirements.'
  },
  '/code-of-conduct': {
    title: 'Code of Conduct',
    body: 'We follow the Contributor Covenant. Please use the linked Code of Conduct for expected behavior, reporting channels, and enforcement details.'
  },
  '/contributing': {
    title: 'Contributing',
    body: 'Contributions are welcome through issues and pull requests. Read the linked contributing guide for setup steps, branch strategy, and review expectations.'
  },
  '/privacy': {
    title: 'Privacy Notice',
    body: 'Nexa Visualize stores onboarding completion in localStorage (`nexa-onboarding-complete`) to improve user experience. No analytics or third-party trackers are configured in this baseline app.'
  }
};

function MarkdownRenderer({ markdown }) {
  const blocks = useMemo(() => {
    const lines = markdown.split('\n');
    const parsed = [];
    let code = null;
    let listItems = [];

    const flushList = () => {
      if (listItems.length) {
        parsed.push({ type: 'list', items: listItems });
        listItems = [];
      }
    };

    lines.forEach((line) => {
      if (line.startsWith('```')) {
        flushList();
        if (code !== null) {
          parsed.push({ type: 'code', value: code.join('\n') });
          code = null;
        } else {
          code = [];
        }
        return;
      }

      if (code !== null) {
        code.push(line);
        return;
      }

      if (/^\s*[-*]\s+/.test(line)) {
        listItems.push(line.replace(/^\s*[-*]\s+/, ''));
        return;
      }

      flushList();

      if (!line.trim()) {
        return;
      }

      if (line.startsWith('#### ')) parsed.push({ type: 'h4', value: line.slice(5) });
      else if (line.startsWith('### ')) parsed.push({ type: 'h3', value: line.slice(4) });
      else if (line.startsWith('## ')) parsed.push({ type: 'h2', value: line.slice(3) });
      else if (line.startsWith('# ')) parsed.push({ type: 'h1', value: line.slice(2) });
      else parsed.push({ type: 'p', value: line });
    });

    if (code !== null) parsed.push({ type: 'code', value: code.join('\n') });
    flushList();

    return parsed;
  }, [markdown]);

  return (
    <article className="markdown" aria-live="polite">
      {blocks.map((block, index) => {
        if (block.type === 'h1') return <h1 key={index}>{block.value}</h1>;
        if (block.type === 'h2') return <h2 key={index}>{block.value}</h2>;
        if (block.type === 'h3') return <h3 key={index}>{block.value}</h3>;
        if (block.type === 'h4') return <h4 key={index}>{block.value}</h4>;
        if (block.type === 'code') return <pre key={index}><code>{block.value}</code></pre>;
        if (block.type === 'list') {
          return (
            <ul key={index}>
              {block.items.map((item) => <li key={item}>{item}</li>)}
            </ul>
          );
        }
        return <p key={index}>{block.value}</p>;
      })}
    </article>
  );
}

function App() {
  const [route, setRoute] = useState(window.location.hash?.replace('#', '') || '/');
  const [showOnboarding, setShowOnboarding] = useState(false);

  useEffect(() => {
    const updateRoute = () => setRoute(window.location.hash?.replace('#', '') || '/');
    window.addEventListener('hashchange', updateRoute);
    return () => window.removeEventListener('hashchange', updateRoute);
  }, []);

  useEffect(() => {
    if (route === '/visualizer' && !localStorage.getItem('nexa-onboarding-complete')) {
      setShowOnboarding(true);
    }
  }, [route]);

  const dismissOnboarding = () => {
    localStorage.setItem('nexa-onboarding-complete', 'true');
    setShowOnboarding(false);
  };

  return (
    <div className="app-shell" data-testid="app-shell">
      <a href="#main-content" className="skip-link">Skip to main content</a>
      <header className="top-nav" role="banner">
        <h1>Nexa Visualize</h1>
        <nav aria-label="Primary">
          {NAV_ROUTES.map((item) => (
            <a
              key={item.path}
              href={`#${item.path}`}
              className={route === item.path ? 'active' : ''}
              aria-current={route === item.path ? 'page' : undefined}
            >
              {item.label}
            </a>
          ))}
        </nav>
      </header>

      <main id="main-content" tabIndex={-1}>
        {route === '/' && (
          <section>
            <h2>Home</h2>
            <p>Explore neural network architecture interactively with accessibility-first navigation and built-in learning material.</p>
          </section>
        )}

        {route === '/visualizer' && (
          <section>
            <h2>Visualizer</h2>
            <p className="helper-text">Tip: use <kbd>Arrow Left</kbd> / <kbd>Arrow Right</kbd> inside controls to switch configuration panels.</p>
            <div className="visualizer-wrap" role="region" aria-label="3D neural network visualizer">
              <NeuralNetwork3D />
            </div>
          </section>
        )}

        {route === '/docs-models' && (
          <section>
            <h2>Docs/Models</h2>
            <p>Built-in documentation from Models.md and Sources.md is rendered below.</p>
            <MarkdownRenderer markdown={MODELS_MARKDOWN} />
            <MarkdownRenderer markdown={SOURCES_MARKDOWN} />
          </section>
        )}

        {route === '/about' && (
          <section>
            <h2>About</h2>
            <p>Nexa Visualize is an educational playground for understanding architecture flow in FNNs, CNNs, Transformers, and MoE systems.</p>
          </section>
        )}

        {ROUTE_CONTENT[route] && (
          <section>
            <h2>{ROUTE_CONTENT[route].title}</h2>
            <p>{ROUTE_CONTENT[route].body}</p>
            {LEGAL_ROUTES.find((item) => item.path === route)?.href && (
              <p><a href={LEGAL_ROUTES.find((item) => item.path === route).href}>Open full document</a></p>
            )}
          </section>
        )}
      </main>

      <footer>
        <nav aria-label="Footer">
          {LEGAL_ROUTES.map((item) => (
            <a key={item.path} href={`#${item.path}`}>{item.label}</a>
          ))}
        </nav>
      </footer>

      {showOnboarding && (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="onboarding-title">
          <div className="modal-card">
            <h3 id="onboarding-title">Welcome to Nexa Visualize</h3>
            <ul>
              <li><strong>Controls:</strong> Start/stop/reset training and adjust camera + animation speed.</li>
              <li><strong>Panels:</strong> Architecture, Parameters, and Loss Landscape can be toggled from the top-right controls.</li>
              <li><strong>Model types:</strong> Visit Docs/Models for CNN, MoE, YOLO, and animation references.</li>
            </ul>
            <button type="button" onClick={dismissOnboarding} aria-label="Close onboarding modal">Got it</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
