# Drug Repurposing through Meta-path Reconstruction using Relationship Embedding

<!-- <img src = "https://github.com/MVPpool/Multi-View-Node-Pruning-for-Accurate-Graph-Representation/blob/main/figs/concept_fig.png" width="80%"> -->

This repository is the official implementation of [Drug Repurposing through Meta-path Reconstruction using Relationship Embedding](https://openreview.net/forum?id=HhUm1cnsTb). 


## Requirements

The following package specification is required for executing the implementation.

```setup
Python==3.8.16
TensorFlow==2.4.0
Numpy==1.19.5
Pandas==1.5.3
Scipy==1.10.1
Scikit-learn==1.2.2
```

## Training & Evaluation

To train the models in the paper, run the given script:

```script
sh predr_classification.sh
```

This script will give a supervised learning experimental result for given dataset.

You can also specify some hyperparameter or select model with the argument specification.

```eval
python main.py --epochs {# epoch} --learning_rate {learning rate} ...
```

## Results

Our model achieves the following performance on the baseline models.

<img src = "https://github.com/PREDR/blob/main/figs/result.png" width="80%">

As the figure shows, our method significantly improves the existing methods with various criteria.


## Contributing

> Copyright on author. All rights reserved.
