# xDeepFM

xDeepFM tries to model feature crossing more explicitly than a plain deep network.

DeepFM relies on an MLP to learn high order interactions. xDeepFM adds a compressed interaction network, often called CIN, to build bounded degree feature interactions in a more structured way.

On MovieLens, xDeepFM is a later experiment after FM and DeepFM are working. It uses the same basic fields, so the interesting question is whether the interaction module adds value beyond a normal MLP.

The first implementation should reuse the DeepFM data pipeline. Keep the model comparison fair: same split, same labels, same metrics, and similar embedding sizes.
