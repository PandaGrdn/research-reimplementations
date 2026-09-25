## Spiking Hopfield Network in a basic transformer Model

I plan on replacing the attention blocks in this basic transformer implementation with spiking hopfield networks and then also a continuous hopfield network i am curious to see what differences a spiking hopfield network makes compared to a continuous one as implemented in "Hopfield Networks is All you need".


Plan for ml interpretability section

- Implement a basic sparse, overcomplete autoencoder to generate features because neural etworks try and encode more features than neurons by representing features as superpositions of multiple neuron activations.

-  This is in the mlp section of the transformer architecture, therefore, it will be interesting to see whether hopfield versus multiheaded attention block has any interesting differences when it comes to how features are encoded into superpositions.