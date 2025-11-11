# Edge prediction


```
$ python ./dataset.py --relations "INVESTS_IN,IMPACT,POSITIVE_IMPACT_ON,NEGATIVE_IMPACT_ON" --limit 100000000        
Wrote:
  CSV:  data/ml_kgclient.csv
  EFEAT:data/ml_kgclient.npy
  NFEAT:data/ml_kgclient_node.npy



$ python ./train.py --data kgclient --use_memory --n_epoch 30 --prefix kg --bs 2000 --n_degree 10 --memory_dim 64

INFO:root:Namespace(data='kgclient', bs=2000, prefix='kg', n_degree=10, n_head=2, n_epoch=30, n_layer=1, lr=0.0001, patience=5, n_runs=1, drop_out=0.1, gpu=0, node_dim=100, time_dim=100, backprop_every=1, use_memory=True, embedding_module='graph_attention', message_function='identity', memory_updater='gru', aggregator='last', memory_update_at_end=False, message_dim=100, memory_dim=64, different_new_nodes=False, uniform=False, randomize_features=False, use_destination_embedding_in_message=False, use_source_embedding_in_message=False, dyrep=False)
The dataset has 42685 interactions, involving 22159 different nodes
The training dataset has 21752 interactions, involving 14615 different nodes
The validation dataset has 6404 interactions, involving 4769 different nodes
The test dataset has 6402 interactions, involving 4127 different nodes
The new node validation dataset has 5053 interactions, involving 4388 different nodes
The new node test dataset has 5320 interactions, involving 3834 different nodes
2215 nodes were used for the inductive testing, i.e. are never seen during training
INFO:root:num of training instances: 21752
INFO:root:num of batches per epoch: 11
INFO:root:start 0 epoch
INFO:root:epoch: 0 took 7.47s
INFO:root:Epoch mean loss: 1.3966865973039106
.....
INFO:root:epoch: 28 took 8.33s
INFO:root:Epoch mean loss: 1.08494910326871
INFO:root:val auc: 0.8513651515151516, new node val auc: 0.8443005896226415
INFO:root:val ap: 0.8533162820642668, new node val ap: 0.8409761955252291
INFO:root:start 29 epoch
INFO:root:epoch: 29 took 7.61s
INFO:root:Epoch mean loss: 1.0790771842002869
INFO:root:val auc: 0.849585606060606, new node val auc: 0.8463808573746472
INFO:root:val ap: 0.8545369161826597, new node val ap: 0.8440654336925396
INFO:root:Test statistics: Old nodes -- auc: 0.8887897727272729, ap: 0.8847071387618776
INFO:root:Test statistics: New nodes -- auc: 0.7382460905349795, ap: 0.7212069454542989
INFO:root:Saving TGN model
INFO:root:TGN model saved



$ python ./predict.py --data kgclient --prefix kg --use_memory --memory_dim 64 --topk 5 --candidates 200 --limit 500

The dataset has 42685 interactions, involving 22159 different nodes
The training dataset has 21752 interactions, involving 14615 different nodes
The validation dataset has 6404 interactions, involving 4769 different nodes
The test dataset has 6402 interactions, involving 4127 different nodes
The new node validation dataset has 5053 interactions, involving 4388 different nodes
The new node test dataset has 5320 interactions, involving 3834 different nodes
2215 nodes were used for the inductive testing, i.e. are never seen during training
/Users/richard/Projects/agentic/kgnn/./predict.py:191: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  dst = int(cand[j])
/Users/richard/Projects/agentic/kgnn/./predict.py:195: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  score = float(scores[j])
Wrote predictions: results/kg-kgclient/predicted_edges.csv
(kgnn)
```