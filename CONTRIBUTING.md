# Contributing to NexaVisualize

First of all, thank you for your interest in contributing!  

At its current state, **NexaVisualize works exactly as I intended it to**:  
- Feedforward networks, CNNs, Transformers, and Mixture of Experts are fully visualized.  
- Light and dark modes, training dynamics, and architecture customization are implemented.  
- The simulator has been stress-tested (100+ experts with Top-K routing). Beyond that, performance depends on your machine.  

For me personally, this project is **feature complete**.  
I may add a hosted version in the future, but for now I’m satisfied with the Docker-based setup.  

---

## How to Contribute
- **Pull Requests (PRs):** If you’d like to add features, fix bugs, or extend the project (e.g., ResNets, VAEs, better monitoring), feel free to open a PR. I’ll review when I can, but there are no guarantees on timeline.  
- **Issues:** If you find a bug or have a suggestion, open an issue. Clear repro steps or mockups will make it easier to evaluate.  
- **Major Contributors:** Significant contributions (like the Light/Dark mode feature from Narua) may lead to direct write access.  

---

## Scope and Boundaries
- I do not plan to add major new architectures myself beyond what’s already here.  
- The simulation is already near its tested limits (100 experts). If you want to push further, go for it — but know that performance beyond that may vary.  
- NexaVisualize is **open-source infrastructure first**. Use it, fork it, remix it — that’s the spirit.  

---

## Final Note
This repo will always remain free and open (Apache 2.0).  

For me, it’s “done.” For the community, it’s a playground.  

If you build something cool on top, let me know — I’d love to see it.  
