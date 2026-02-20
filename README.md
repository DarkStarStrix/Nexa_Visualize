# Nexa Visualize

[![CI](https://github.com/OWNER/Nexa_Visualize/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/Nexa_Visualize/actions/workflows/ci.yml)
[![Deploy to GitHub Pages](https://github.com/OWNER/Nexa_Visualize/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/OWNER/Nexa_Visualize/actions/workflows/deploy-pages.yml)
[![Live Site](https://img.shields.io/badge/Live%20Site-GitHub%20Pages-blue?logo=github)](https://OWNER.github.io/Nexa_Visualize/)

Nexa Visualize is an interactive, educational tool for visualizing the architecture and training process of a neural network in real-time 3D. Built with React and Three.js, it provides an intuitive way to understand concepts like forward propagation, backpropagation, and hyperparameter tuning.

## Hosting Choice

This repository is configured for **static hosting on GitHub Pages**.

- CI workflow: `.github/workflows/ci.yml`
- Deployment workflow: `.github/workflows/deploy-pages.yml`
- Live URL format: `https://<github-username>.github.io/Nexa_Visualize/`

> Replace `OWNER` in the badge links above with your GitHub username or organization.

## Environment Configuration

This app is static and does not require runtime secrets for normal operation.

Create a `.env` file at the project root only if you need custom React environment variables:

```bash
# .env.example
REACT_APP_API_BASE_URL=https://api.example.com
REACT_APP_ENABLE_EXPERIMENTAL_MODE=false
```

Rules for frontend environment variables:

- Only variables prefixed with `REACT_APP_` are exposed to the browser.
- Do **not** put private credentials in frontend environment variables.
- Build-time variables are injected during `npm run build`.

## Required Repository Settings / Secrets

For GitHub Pages deployment in this repo, no additional custom secrets are required.

Required repository settings:

1. Go to **Settings → Pages**.
2. Under **Build and deployment**, select **Source: GitHub Actions**.
3. Ensure the default branch is `main` (or `master`, both are supported in workflows).

GitHub-provided token usage:

- `GITHUB_TOKEN` (automatically provided by GitHub Actions) is used by `actions/deploy-pages`.

## CI/CD Workflows

### Continuous Integration (`ci.yml`)

Runs on push and pull requests:

1. `npm ci`
2. `npm run lint`
3. `CI=true npm test -- --watch=false`
4. `npm run build`

### Deployment (`deploy-pages.yml`)

Runs on pushes to `main`/`master` and manual dispatch:

1. `npm ci`
2. Build with `PUBLIC_URL=/${{ github.event.repository.name }}`
3. Upload build output from `build/`
4. Deploy to GitHub Pages with `actions/deploy-pages`

## Features

- **Interactive 3D Visualization** of neural network architectures.
- **Multiple Model Architectures** including FNNs, CNN variants, Transformers, MoE, and Autoencoders/VAEs.
- **Dynamic Architecture Editing** for layers and neuron counts.
- **Real-time Training Animation** for forward pass, backward pass, and weight updates.
- **Hyperparameter Controls** for epochs, learning rate, optimizer, and loss.
- **Live Monitoring Graph** for loss and accuracy.
- **Camera Controls** for auto-rotate and manual navigation.

## Local Development

Install and run locally:

```bash
npm ci
npm start
```

Build for production:

```bash
npm run build
```

Run tests:

```bash
CI=true npm test -- --watch=false
```

Run lint:

```bash
npm run lint
```

## Technology Stack

- React
- Three.js
- JavaScript (ES6+)
- CSS3
