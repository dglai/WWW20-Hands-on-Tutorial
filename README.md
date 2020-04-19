Learning Graph Neural Networks with Deep Graph Library -- WWW 2020 Hands-on Tutorial
===

Presenters: George Karypis, Zheng Zhang, Minjie Wang, Da Zheng, Quan Gan

Time: (UTC/GMT +8) 09:00-16:30, April, 20, Monday

Abstract
---
Learning from graph and relational data plays a major role in many applications
including social network analysis, marketing, e-commerce, information retrieval,
knowledge modeling, medical and biological sciences, engineering, and others. In the
last few years, Graph Neural Networks (GNNs) have emerged as a promising new supervised
learning framework capable of bringing the power of deep representation learning to
graph and relational data. This ever-growing body of research has shown that GNNs
achieve state-of-the-art performance for problems such as link prediction, fraud
detection, target-ligand binding activity prediction, knowledge-graph completion,
and product recommendations.

The objective of this tutorial is twofold. First, it will provide an overview of the
theory behind GNNs, discuss the types of problems that GNNs are well suited for, and
introduce some of the most widely used GNN model architectures and problems/applications
that are designed to solve. Second, it will introduce the Deep Graph Library (DGL), a
new software framework that simplifies the development of efficient GNN-based training
and inference programs. To make things concrete, the tutorial will provide hands-on
sessions using DGL. This hands-on part will cover both basic graph applications (e.g.,
node classification and link prediction), as well as more advanced topics including
training GNNs on large graphs and in a distributed setting. In addition, it will provide
hands-on tutorials on using GNNs and DGL for real-world applications such as recommendation
and fraud detection.

Prerequisite
---

The attendees should have some experience with deep learning and have used deep learning
frameworks such as MXNet, Pytorch, and TensorFlow. Attendees should have experience with
the various problems and techniques arising and used in graph learning and analysis, but
it is not required.

Agenda
---

| Time | Session | Material | Presenter |
|:----:|:-------:|:--------:|:---------:|
| 9:00-9:45 | Overview of Graph Neural Networks | [slides](https://github.com/zheng-da/dgl-tutorial-full/blob/master/GNN_overview.pptx) | George Karypis |
| 9:45-10:30 | Overview of Deep Graph Library (DGL) | [slides](https://github.com/zheng-da/dgl-tutorial-full/blob/master/dgl_api/dgl-www-zz.pptx) | Zheng Zhang |
| 10:30-11:00 | Virtual Coffee Break | | |
| 11:00-12:30 | (Hands-on) GNN models for basic graph tasks | [notebook](https://github.com/dglai/WWW20-Hands-on-Tutorial/blob/master/_legacy/basic_apps/BasicTasks_pytorch.ipynb) | Minjie Wang |
| 12:30-14:00 | Virtual Lunch Break | | |
| 14:00-15:30 | (Hands-on) GNN training on large graphs | [notebook](https://github.com/dglai/WWW20-Hands-on-Tutorial/blob/master/large_graphs) | Da Zheng |
| 15:30-16:00 | Virtual Coffee Break | | |
| 16:00-17:30 | (Hands-on) GNN models for real-world applications | [notebook](https://github.com/dglai/WWW20-Hands-on-Tutorial/blob/master/_legacy/advanced_apps/rec/Recommendation.ipynb) | Quan Gan |

Section Content
---

* **Section 1: Overview of Graph Neural Networks.** This section describes how graph
  neural networks operate, their underlying theory, and their advantages over alternative
  graph learning approaches. In addition, it describes various learning problems on graphs
  and shows how GNNs can be used to solve them.
* **Section 2: Overview of Deep Graph Library (DGL).** This section describes the different
  abstractions and APIs that DGL provides, which are designed to simplify the implementation
  of GNN models, and explains how DGL interfaces with MXNet, Pytorch, and TensorFlow.
  It then proceeds to introduce DGL’s message-passing API that can be used to develop
  arbitrarily complex GNNs and the pre-defined GNN nn modules that it provides.
* **Section 3: GNN models for basic graph tasks.** This section demonstrates how to use
  GNNs to solve four key graph learning tasks: node classification, link prediction, graph
  classification, and network embedding pre-training. It will show how GraphSage, a popular
  GNN model, can be implemented with DGL’s nn module and show how the node embeddings
  computed by GraphSage can be used in different types of downstream tasks. In addition,
  it will demonstrate the implementation of a customized GNN model with DGL’s message passing
  interface.
* **Section 4: GNN training on large graphs.** This section uses some of the models described
  in Section 3 to demonstrate mini-batch training, multi-GPU training, and distributed
  training in DGL. It starts by describing how the concept of mini-batch training applies to
  GNNs and how mini-batch computations can be sped up by using various sampling techniques.
  It then proceeds to illustrate how one such sampling technique, called neighbor sampling,
  can be implemented in DGL using a Jupyter notebook. This notebook is then extended to show
  multi-GPU training and distributed training.
* **Section 5: GNN models for real-world applications.** This section uses the techniques
  described in the earlier sections to show how GNNs can be used to develop scalable solutions
  for recommendation and fraud detection. For recommendation, it develops a nearest-neighbor
  item-based recommendation method that employs a GNN model to learn item embeddings by
  following an end-to-end learning approach. For fraud detection, it extends the node
  classification model in the previous section to work on heterogeneous graphs and addresses
  the scenario where there exist few labelled samples.

## Community

Join our [Slack channel "WWW20-tutorial"](https://join.slack.com/t/deep-graph-library/shared_invite/zt-docxzmw2-9yMsL7rv9a2tpjzlLlVptg) for discussion.
