# Sources & Lessons – NexaVisualize

This document compiles all references, citations, and insights used to build **NexaVisualize**.  
It serves as both a bibliography and a reflection on the project.  

---

## Citations & References

The following resources were directly referenced while implementing different architectures in NexaVisualize:

- **Feedforward Neural Networks (FNNs)**  
  - Goodfellow, Ian, Bengio, Yoshua, Courville, Aaron. *Deep Learning.* MIT Press (2016).
  - Introduction to Feedforward Neural Networks (FNNs): [Intro to FNN's](https://novarkservices.com/feedforward-neural-networks-fnns-a-fundamental-deep-learning-model/)
  - Feedforward Neural Networks: The Backbone of Deep Learning: [The bacbone of DL](https://medium.com/@h6364749/feedforward-neural-networks-the-backbone-of-deep-learning-22bfa6635ab7)
  - Stanford CS231n: [Neural Networks Part 1](http://cs231n.github.io/neural-networks-1/).  

- **Convolutional Neural Networks (CNNs)**  
  - LeCun, Yann, et al. *Gradient-Based Learning Applied to Document Recognition.* Proceedings of the IEEE (1998).
  - An Introduction to Convolutional Neural Networks (CNNs): [An intro To CNN's](https://www.datacamp.com/tutorial/introduction-to-convolutional-neural-networks-cnns)
  - Demystifying CNNs: A Deep Dive into Convolutional Neural Network Fundamentals: [A Deep Dive into CNN's](https://medium.com/@karemsaeed2468/demystifying-cnns-a-deep-dive-into-convolutional-neural-network-fundamentals-0b9ed1d1d2fa)
  - Datacamp: [Convolutional Neural Networks Explained](https://www.datacamp.com/tutorial/convolutional-neural-networks).  

- **Transformers**  
  - Vaswani, A., et al. *Attention Is All You Need.* NeurIPS (2017).  
  - VitalFlux: [Transformer Neural Network Architecture](https://vitalflux.com/transformer-neural-network-architecture-explained/).
  - Attention is all you need: [Attention is all you need](https://huggingface.co/papers/1706.03762)
  - How do Transfomers work Huggingface: [How do transformers work](https://www.bing.com/ck/a?!&&p=1081c5106f8d5e08fe11b79afbc7a3cf679efaddff7ed983cb3bb8ff21c67520JmltdHM9MTc1ODU4NTYwMA&ptn=3&ver=2&hsh=4&fclid=11f78f6f-4226-66ca-395a-9afb432267e8&psq=how+do+transformers+work+ML&u=a1aHR0cHM6Ly9odWdnaW5nZmFjZS5jby9sZWFybi9sbG0tY291cnNlL2VuL2NoYXB0ZXIxLzQ) 
  - Datacamp: [Transformers Explained Visually](https://www.datacamp.com/tutorial/transformer-neural-network).  

- **Mixture of Experts (MoE)**  
  - Shazeer, N., et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* arXiv:1701.06538 (2017).
  - How do MOE's work: [How Do MOE's work](https://www.bing.com/ck/a?!&&p=bbe74d210d7863f6342b5f225a4c312ab7da83c61e256cc9dfc35e010f071c8eJmltdHM9MTc1ODU4NTYwMA&ptn=3&ver=2&hsh=4&fclid=11f78f6f-4226-66ca-395a-9afb432267e8&psq=How+do+MOE%27s+work+ML&u=a1aHR0cHM6Ly9tZWRpdW0uY29tL0BtbmUvZXhwbGFpbmluZy10aGUtbWl4dHVyZS1vZi1leHBlcnRzLW1vZS1hcmNoaXRlY3R1cmUtaW4tc2ltcGxlLXRlcm1zLTg1ZGU5ZDE5ZWE3Mw)
  - A Visual Guide to Mixture of Experts (MoE): [A Visual Guide to MOE's](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)
  - Hugging Face: [Visual Guide to Mixture of Experts (MoE)](https://huggingface.co/blog/moe).  

---

## Models Implemented in V1

- **Feedforward Neural Network (FNN)** – fully customizable  
- **Convolutional Neural Networks (CNNs)** – base + variants  
- **Transformers** – vanilla encoder-decoder, extendable for variants  
- **Mixture of Experts (MoE)** – router + expert visualization  
- *(Stretch goals, left for community)*: Autoencoder, Variational Autoencoder (VAE)  

---

## Lessons Learned

This project wasn’t about breaking new ground in ML theory — it was about **testing and solidifying my own understanding**.  

Key takeaways:  
- **Visualization matters.** Most ML work is hidden in math or code. Seeing the *flow of data* across blocks and layers helps build intuition and makes architectures less abstract.  
- **Refresher on fundamentals.** Re-implementing CNNs, Transformers, and MoEs from scratch was a great way to confirm I actually understood them at a structural level.  
- **Educational potential.** Visualizations combined with citations allow learners to both *see* the architecture and *read deeper* from the sources.  
- **Scope discipline.** By keeping V1 focused (FNN, CNN, Transformer, MoE + quality-of-life features like light/dark mode), the project reached a natural “feature complete” state instead of drifting endlessly.  

---

## Closing Note

NexaVisualize is **feature complete for me**.  
- Community contributions are welcome via PRs.  
- If you’d like to extend it (e.g., add ResNets, LSTMs, or VAEs), the modular base classes are designed to be extendable.  

For me, this project is done. For the community, it’s a playground.  
