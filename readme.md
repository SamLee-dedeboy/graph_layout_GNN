# Evaluating Graph Layouts using Graph Neural Networks
This repository implements the model describe in `report.pdf`.

## Motivation and Objectives
With the recent advances of deep learning, several works have been proposed to evaluate graph layouts using deep learning approaches.
However, they both use CNN-based models and thus have weak generality in terms of layout styles.
The model has to be retrained with new datasets if the targeted layouts is changed.
Even minors changes such as node color or edge width will require the retrain. 
Also, the training time is very long because the image size of the graph layouts often need to be big enough to contain a certain amount of nodes.
The researchers have to make a trade-off between image size(training time) and graph size.

To overcome the weak generality of CNN models, in this project I use Graph Neural Networks(GNNs) to evaluate the graph layouts.
Node coordinates are used to represent layouts instead of images. 
Using node coordinates have several benefits. 
The first is that the generality of the model is significantly improved, because the model does not require retraining if layout styles are changed.
Second is that using node coordinates enables the usage of GNN models, which usually have shorter training time comparing to CNNs, as the number of parameters is significantly less.
Moreover, previous CNN-based works can not explicitly take graph connectivity into account. 
Since GNNs are trained on the graph adjacent matrix, GNN-based approaches can evaluate the layouts based on the graph connectivity features. 

## Problem Formulation
The goal of the project is to give scores on different layouts on the save graph, based on the graph connectivity features and the layout features.
This is achieved by first generate a subgraph of the original graph, and then use the GAE model to reconstruct the original graph, using the node positions as node features. For mode detailed explanation, refer to `report.pdf`.

## Results
Sections below presents the results of this p   roject. 
More detailed explanation and insights are described in Section 5 in `report.pdf`. 
### Reconstruction Efficacy
The efficacy of reconstructing adjacent matrix from node coordinates is not shown by previous studies.
To prove the efficacy, a link prediction is conducted by the model. 
The model trys to reconstruct the edges from existing edges and node features, which are node coordinates in this case.
The missing edge rate is set to 50 percent, i.e. only 50 percent of edges is remaining, and the model will reconstruct the missing 50 percent edges.
Table below presents the link prediction accuracy on the test dataset. 
Left in the cell is accuracy on GAE and right is VGAE.
The average accuracy is **0.769** for GAE and **0.776** for VGAE, proving that reconstructing adjacency matrix from node coordinates is viable.
| graph \layout | MDS | spiral | spring | circular |
| :--: | :----: | :-------: | :----------: | :-------: |
| 0 | 0.827/0.834 | 0.832/0.837 | 0.826/0.832 | 0.847/0.851 |
|414 | 0.767/0.782 | 0.745/0.753 | 0.740/0.746 | 0.776/0.782 |
|3980 | 0.724/0.729 | 0.723/0.733 | 0.732/0.746 | 0.778/0.790 |
|698 | 0.687/0.671 | 0.752/0.758 | 0.785/0.794 | 0.774/0.782 |

### Correlation to Aesthetics Metrics
To evaluate the ranking result, and therefore the efficacy of the project, the correlation between test reconstruction loss and aesthetic metrics is investigated.
Three aesthetic metrics are used: (1)M_l: edge length variation (2) M_a: Minimum Angle (3) E_c: Edge crossings.

The correlation score is calculated by the `Pearson correlation coefficients`.
A score of 1 means strong positive correlation, -1 means strong negative correlation, and 0 means no correlation.
Since the three aesthetic metrics are expected to be minimized as well as the reconstruction loss,
a score of 1 should indicate a better performance.
Table below shows the correlation score between reconstuction loss and the three aesthetic metrics on four test graphs for GAE/VGAE.

|graph\metric | M_l | M_a | E_c |
| :--: | :----: | :-------: | :----------: |
| 0 | -0.648/**0.904** | -0.530/0.612 | 0.604/0.040|    
| 414 | -0.425/0.479 | 0.147/**0.861** | -0.425/-0.289  |
| 3980 | 0.816/0.437 | -0.943/-0.695 | -0.684/-0.985 |
| 698 | 0.873/**0.954** | 0.937/**0.984** | -0.428/-0.001  |

The result shows that the model is able to find the aesthetically best layout based on the graph connectivity features,
even though in the architecture no aesthetic metric is taken into account.

## Source Code Organization
The main implementation is contained in `model.py`, `preprocess.py` and `layout_metrics.py`. 

`model.py`: contains codes to initialize, train, and test the model.

`preprocess`: contains codes to generate dataset from raw data in `dataset/`, including generating layouts and node positions.

`layout_metrics`: contains codes to calculate aesthetic metrics. 

There are some pre-trained model and preprocessed data that can be directly load.

### Preprocessed Data
`train` and `test` contains the train dataset and test dataset.
To load, run:
```python
# init model
l = LayoutRater()
# load data from files.
train_data, test_data = l.loadData('train', "test")
```
Similarly, `test_data_w_metrics_n` contains test data along with the pre-computed aesthetic metrics, so that you don't need to re-compute the aesthetic metrics on each data (which is time-consuming).
You can run something like the following to load:
```python
train_data, test_data = l.loadData('train', "test_data_w_metrics_0")`
```
### pre-trained model
`GAE` and `VGAE` contains pre-trained model. It's not time-consuming to train the model, but it's more convenient this way. To load the model, run:
```python
model = l.load_model('GAE')
#
# or:
#
model = l.load_model('VGAE')
```

## Recreation of the project
To run the tests, first load the model and pre-processed data as mentioned above, then simply run:
```python
test_loader = DataLoader(test_data, 
                        batch_size=1, 
                        shuffle=False)
l.test(model=model,dataLoader=test_loader)
```
This method should print the loss on each data, as well as the aesthetic metrics. You can add any additional thing that you want to check, for example the reconstuction accuray can be collected using:
```python
auc, ap = model.test(z, data.pos_edge_index, data.neg_edge_index) 
```
By default, this method also computes the `pearson_coefficient_correlation` score between the reconstruction loss and each aesthetic metrics respectively. 