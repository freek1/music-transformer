# music-transformer

Training a Transformer on music notes

Goal:

-   Pre-train a BERT(-like) model using MaskedLM on music data.
-         in parallel, to allow for larger batch sizes
-   listen to noisy (masked) data and reconstructed predictions

TODO:

-   adapt script to allow parallel gpus for pre-training,
-   change masks in every epoch (form roberta paper)
-   listen to data masked and reconstructed (test set)
