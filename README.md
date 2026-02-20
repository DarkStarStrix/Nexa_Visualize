# Nexa Visualize

Nexa Visualize is an interactive neural-network visualization app built with React + Three.js.
The project now uses a **single canonical implementation** in `src/features/visualizer/` and no longer side-loads runtime code from `public/main.js`.

## Canonical source decision

| Source | Capability level | Decision |
| --- | --- | --- |
| `public/main.js` | Full feature set (multi-architecture rendering, training animation phases, 2D/3D view toggle, operator inspection panels) | **Promoted into React feature module** |
| `src/NeuralNetwork3D.tsx` | Earlier React-only variant with narrower model/runtime controls | **Retired** |

Canonical runtime path:

`src/index.js` → `src/App.js` → `src/features/visualizer/VisualizerFeature.js` → `visualizerRuntime.js`

## Feature matrix (preserved behavior)

| Capability | Status | Notes |
| --- | --- | --- |
| Multiple architectures | ✅ Preserved | FNN, CNN variants, Transformer variants, MoE, Autoencoder/VAE, Neural Operator flows are still available via architecture selector/runtime logic. |
| Training animation | ✅ Preserved | Forward pass, backward pass, and weight update animation phases remain in the visual training loop. |
| 2D / 3D toggle | ✅ Preserved | Perspective + orthographic modes are retained with dedicated camera/view controls. |
| Operator details | ✅ Preserved | Operator legend and hover/click detail panels remain active for neural operator modules. |

## Development

```bash
npm install
npm start
```

## Build

```bash
npm run build
```
