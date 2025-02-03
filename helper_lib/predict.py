from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
import pandas as pd
from collections import Counter
import random
from IPython.display import clear_output

from helper_lib.graph import check_if_in_graph, build_triples_df

from datetime import datetime
import logging

from typing import Type, List, Optional, Union
from types import MethodType
import math
from ensmallen import Graph
from embiggen.embedding_transformers import EdgePredictionTransformer
from embiggen.utils import AbstractEdgeFeature

def run_model_parametrized(x, y, graph, 
    ModelClass, keep_top_classes, 
    parameter_name, parameter_values, 
    standardize = True, model_parameters={}, n_splits = 5, random_state = 42, test_size = 0.3,
    save_df_path=None,
  ):
  model_name = ModelClass.__name__
  df = pd.DataFrame(columns=['model','num_classes',f'{parameter_name}','balanced_acc_mean','balanced_acc_std','time'])
  if(save_df_path is not None):
    df.to_csv(save_df_path,index=False)
  for value in parameter_values:
    model_parameters[parameter_name] = value
    if standardize:
      model = make_pipeline(StandardScaler(), ModelClass(**model_parameters))
    else:
      model = ModelClass(**model_parameters)
    for k in keep_top_classes:
      logging.info(f"Parameter {parameter_name} = {value}, Number of classes: {k}")
      start_time = time.time()
      balanced_accuracy = calculate_balanced_accuracy(x, y, graph, model,
        n_splits=n_splits, 
        keep_top_classes=k,
        test_size=test_size,
        random_state=random_state
      )
      elapsed_time = time.time() - start_time
      logging.info(f"Balanced accuracy: {balanced_accuracy[0]*100:.2f}% ± {balanced_accuracy[1]*100:.2f}% ({elapsed_time:.3f} seconds) ({datetime.now().strftime('%d/%m/%y %H:%M:%S')})")
      df.loc[len(df.index)] = [model_name,k,value,balanced_accuracy[0],balanced_accuracy[1],elapsed_time]
      if(save_df_path is not None):
        df.to_csv(save_df_path,index=False)
  return df

def run_model(x, y, graph, 
    ModelClass, keep_top_classes, 
    standardize = True, model_parameters={}, n_splits = 5, random_state = 42, test_size = 0.3,
    save_df_path=None,
  ):
  model_name = ModelClass.__name__
  df = pd.DataFrame(columns=['model','num_classes','balanced_acc_mean','balanced_acc_std','time'])
  if(save_df_path is not None):
    df.to_csv(save_df_path,index=False)
  if standardize:
    model = make_pipeline(StandardScaler(), ModelClass(**model_parameters))
  else:
    model = ModelClass(**model_parameters)
  for k in keep_top_classes:
    
    logging.info(f"Number of classes: {k}")
    start_time = time.time()
    balanced_accuracy = calculate_balanced_accuracy(x, y, graph, model,
      n_splits=n_splits, 
      keep_top_classes=k,
      test_size=test_size,
      random_state=random_state
    )
    elapsed_time = time.time() - start_time
    logging.info(f"Balanced accuracy: {balanced_accuracy[0]*100:.2f}% ± {balanced_accuracy[1]*100:.2f}% ({elapsed_time:.3f} seconds) ({datetime.now().strftime('%d/%m/%y %H:%M:%S')})")
    df.loc[len(df.index)] = [model_name,k,balanced_accuracy[0],balanced_accuracy[1],elapsed_time]
    if(save_df_path is not None):
      df.to_csv(save_df_path,index=False)
  return df

def calculate_balanced_accuracy(nodes_embedded, node_types, graph, model, random_state=42, keep_top_classes=0, test_size=0.3, n_splits=5):
  if isinstance(nodes_embedded,pd.DataFrame):
    nodes_embedded = nodes_embedded.to_numpy()
  df = pd.DataFrame()
  #{'embeddingX':nodes_embedded[:,0],'embeddingY':nodes_embedded[:,1],'node_type': node_types}
  number_of_dimensions = nodes_embedded.shape[1]
  for dimension in nodes_embedded.T:
    df[f'embedding{len(df.columns)}'] = dimension
  df = df.copy()
  df['node_type'] = node_types
  

  if keep_top_classes > 0:
    # show only the top k types by number of nodes and group the rest as 'others'
    node_type_counts = df['node_type'].value_counts()
    # print(node_type_counts)
    node_type_counts = node_type_counts.sort_values(ascending=False)
    top_counts = node_type_counts.index[:keep_top_classes].to_list()
    # print(node_type_counts.index[:keep_top_classes])
    node_type_counts = node_type_counts[:keep_top_classes]

    # df['node_type'] = df['node_type'].apply(lambda x: x if x in node_type_counts.index else -1)
    df['node_type'] = df['node_type'].apply(lambda x: top_counts.index(x) if x in top_counts else keep_top_classes)

  # check if the class with the lowest number of nodes has more than 1 node
  if df['node_type'].value_counts().min() == 1:
    #print('Atleast one class has only one node. Using ShuffleSplit')
    SplitterClass = ShuffleSplit
  else: 
    #print('The class with the lowest number of nodes has more than 1 node. Using StratifiedShuffleSplit')
    SplitterClass = StratifiedShuffleSplit

  # convert data from df to numpy array of tuples
  # nodes_embedded_train_test = df[['embeddingX','embeddingY']].to_numpy()
  nodes_embedded_train_test = df[[f'embedding{i}' for i in range(number_of_dimensions)]].to_numpy()
  node_types_train_test = df['node_type'].to_numpy()

  test_accuracies = []
  train_accuracies = []

  for train_indices, test_indices in SplitterClass(
    n_splits=n_splits,
    test_size=test_size,
    random_state=random_state
  ).split(nodes_embedded_train_test, node_types_train_test):
    #model = DecisionTreeClassifier(max_depth=5)

    train_x, test_x = nodes_embedded_train_test[train_indices], nodes_embedded_train_test[test_indices]
    train_y, test_y = node_types_train_test[train_indices], node_types_train_test[test_indices]

    model.fit(train_x, train_y)

    train_accuracies.append(
      balanced_accuracy_score(train_y, model.predict(train_x))
    )

    test_accuracies.append(
      balanced_accuracy_score(test_y, model.predict(test_x))
    )

  mean_accuracy = np.mean(test_accuracies)
  std_accuracy = np.std(test_accuracies)
  
  return (mean_accuracy,std_accuracy,test_accuracies, train_accuracies)

def calculate_balanced_accuracy_grape(nodes_embedded, node_types, graph, random_state=42, keep_top_classes=0,n_splits=5):
  # Sets the top k most frequent node types to be the only ones, the rest are set to k
  node_types = np.array(node_types,copy=True)
  if(keep_top_classes>0):
    counts = np.bincount(node_types)
    node_type_names_iter = (
      graph.get_node_type_name_from_node_type_id(node_id)
      for node_id in range(graph.get_number_of_node_types())
    )
    node_type_names = np.array(
      list(node_type_names_iter),
      dtype=str,
    )
    top_counts = [
      index
      for index, _ in sorted(
          enumerate(zip(counts, node_type_names)), key=lambda x: x[1], reverse=True
      )[:keep_top_classes]
    ]
    for i, element_type in enumerate(node_types):
      if element_type not in top_counts:
        node_types[i] = keep_top_classes
      else:
        node_types[i] = top_counts.index(element_type)

  if min(Counter(node_types).values())==1:
    #print('Atleast one class has only one node. Using ShuffleSplit')
    SplitterClass = ShuffleSplit
  else: 
    #print('The class with the lowest number of nodes has more than 1 node. Using StratifiedShuffleSplit')
    SplitterClass = StratifiedShuffleSplit

  test_accuracies = []
  nodes_embedded_numpy = np.array(nodes_embedded)
  node_types_numpy = np.array(node_types)

  for train_indices, test_indices in SplitterClass(
    n_splits=n_splits,
    test_size=0.3,
    random_state=random_state
  ).split(nodes_embedded_numpy, node_types_numpy):
    model = DecisionTreeClassifier(max_depth=5)

    train_x, test_x = nodes_embedded_numpy[train_indices], nodes_embedded_numpy[test_indices]
    train_y, test_y = node_types_numpy[train_indices], node_types_numpy[test_indices]

    model.fit(train_x, train_y)

    test_accuracies.append(
      balanced_accuracy_score(test_y, model.predict(test_x))
    )

  mean_accuracy = np.mean(test_accuracies)
  std_accuracy = np.std(test_accuracies)

  return (mean_accuracy,std_accuracy)

def edge_prediction_pipeline(graph, model, embedder, pair_to_predict, train_on_filtered=True, train_size = 0.7, number_of_holdouts=5, seed=42, verbose=False, clear_output_holdout=True, use_scale_free_distribution=True):
  random.seed(seed)

  results = []

  for i in range(number_of_holdouts):
    # clean the cell output at each iteration to avoid huge cell outputs
    if clear_output_holdout:
      clear_output(wait=True) 
    # use connected monte carlo to obtain a training set that has the same connectivity guarantees as full graph
    logging.info(f'Generating holdout {i+1}/{number_of_holdouts}')
    random_state = random.randrange(0,100000)
    train_graph, positive_test_graph = graph.connected_holdout(train_size=train_size,random_state=random_state)
      
    # check if number of connected components is the same in the training set and full graph
    logging.debug(train_graph.get_number_of_connected_components())
    assert train_graph.get_number_of_connected_components() == graph.get_number_of_connected_components()
    
    logging.info('Filtering train and test graph by source/destination node type')
    # keep only the edges (source-destination node type) we are interested in
    train_graph_filtered = train_graph.filter_from_names(
      source_node_type_name_to_keep=[pair_to_predict[0]],
      destination_node_type_name_to_keep=[pair_to_predict[1]]
    )
    test_graph_filtered = positive_test_graph.filter_from_names(
      source_node_type_name_to_keep=[pair_to_predict[0]],
      destination_node_type_name_to_keep=[pair_to_predict[1]]
    )

    if verbose:
      df_train = build_triples_df(train_graph)
      # df_test = build_triples_df(positive_test_graph)
      df_train_filtered = build_triples_df(train_graph_filtered)
      # df_test_filtered = build_triples_df(test_graph_filtered)
      logging.info(f"Positive training set size: {len(df_train)}")
      logging.info(f"Positive training set FILTERED size: {len(df_train_filtered)}")
      # print relative frequencies of top 10 source_type - destination_type in train and test set compared to full graph
      # for i,(key,count) in enumerate(df_view0['complete_label'].value_counts().items()):
      #   logging.info(f"{key}: {df_train['complete_label'].value_counts()[key]/count} - {df_test['complete_label'].value_counts()[key]/count}")
      #   if i == 9:break

    # calculate the embedding on the not filtered train graph
    logging.info('Training embedding on unfiltered train graph')
    before = time.time()
    train_embedding = embedder.fit_transform(train_graph)
    logging.info(f"Embedding time:{time.time()-before}")
    # the embedding could be cached to avoid recalculating it
    
    logging.info('Training model using the filtered train graph')
    # train the model using the node embeddings and with the filtered train graph => train only on miRNA-Disease edges
    # the edge prediction models from grape generate their own negative edges for training
    model.fit(
      graph=train_graph_filtered if train_on_filtered else train_graph,
      node_features=train_embedding,
      support=train_graph,
    )

    logging.info('Evaluating model on positive train set')
    train_pred = model.predict_proba(
      graph=train_graph_filtered if train_on_filtered else train_graph,
      node_features=train_embedding,
      return_predictions_dataframe=True,
      support=train_graph
    )

    training_set_size = len(train_pred)

    pos_train_score = balanced_accuracy_score([True for _ in range(len(train_pred))], train_pred['prediction'].apply(lambda x:x>0.5))
    if verbose:
      # pred_train_edge_presence = train_pred.apply(lambda row:check_if_in_graph(graph,row['sources'],row['destinations'],pair_to_predict),axis=1)
      # train_score = balanced_accuracy_score(pred_train_edge_presence, train_pred['prediction'].apply(lambda x:x>0.5))
      logging.info(f"Balanced accuracy positive score TRAINING: {pos_train_score}")

    logging.info('Creating a graph with the negative edges for testing')
    # create graph with negative edges for testing 
    negative_test_graph = graph.sample_negative_graph(
      # number_of_negative_samples=test_graph_filtered.get_number_of_edges(), # this option creates only half the edges
      number_of_negative_samples=test_graph_filtered.get_number_of_directed_edges(),
      source_node_types_names=[pair_to_predict[0]],
      destination_node_types_names=[pair_to_predict[1]],
      random_state=random_state,
      use_scale_free_distribution=use_scale_free_distribution,
    )

    if verbose:
      logging.info('Positive test set:')
      df_test_positive = build_triples_df(test_graph_filtered)
      logging.info(df_test_positive['complete_label'].value_counts())
      logging.info('Negative test set:')
      df_test_negative = build_triples_df(negative_test_graph)
      logging.info(df_test_negative['complete_label'].value_counts())

    if verbose:
      logging.info(f"#edges in positive test graph: {test_graph_filtered.get_number_of_directed_edges()}")
      logging.info(f"#edges in negative test graph: {negative_test_graph.get_number_of_directed_edges()}")

    # use model to predict on the positive edges
    logging.info('Using the model to predict the existence of positive edges')
    pos_pred = model.predict_proba(
      graph=test_graph_filtered,
      node_features=train_embedding,
      return_predictions_dataframe=True,
      support=train_graph
    )
    
    if verbose:
      # check if all edges of positive test set are in the original graph
      pos_pred_edge_presence = pos_pred.apply(lambda row:check_if_in_graph(graph,row['sources'],row['destinations'],pair_to_predict),axis=1)
      logging.info(f'Are all positive edges present in the positive test set also in the original graph? {pos_pred_edge_presence.all()}')
      logging.info(pos_pred_edge_presence.value_counts())

    # use model to predict on the negative edges
    logging.info('Using the model to predict the non-existence of negative edges')
    neg_pred = model.predict_proba(
      graph=negative_test_graph,
      node_features=train_embedding,
      return_predictions_dataframe=True,
      support=train_graph
    )

    testing_set_size = len(pos_pred) + len(neg_pred)

    if verbose:
      # check if all edges of negative test set are not in the original graph
      neg_pred_edge_presence = neg_pred.apply(lambda row:check_if_in_graph(graph,row['sources'],row['destinations'],pair_to_predict),axis=1)
      logging.info(f'Are all negative edges present in the negative test set NOT in the original graph? {~neg_pred_edge_presence.all()}')
      logging.info(neg_pred_edge_presence.value_counts())

    # calculate balanced accuracy score for positive and negative predictions
    pos_score = balanced_accuracy_score([True for _ in range(len(pos_pred))], pos_pred['prediction'].apply(lambda x:x>0.5))
    neg_score = balanced_accuracy_score([False for _ in range(len(neg_pred))], neg_pred['prediction'].apply(lambda x:x>0.5))
    logging.info(f"Balanced accuracy positive score: {pos_score}")
    logging.info(f"Balanced accuracy negative score: {neg_score}")
    avg_score = (pos_score+neg_score)/2
    logging.info(f"Balanced accuracy mean score: {avg_score}")

    auc_score = roc_auc_score(
      [True for _ in range(len(pos_pred))] + [False for _ in range(len(neg_pred))],
      pd.concat([pos_pred['prediction'].apply(lambda x:x>0.5), neg_pred['prediction'].apply(lambda x:x>0.5)])
    )
    logging.info(f"AUC score: {auc_score}")

    results.append((graph.get_name(), embedder.model_name(), model.model_name(), pair_to_predict[0], pair_to_predict[1], 
                    train_on_filtered, training_set_size, testing_set_size, 
                    pos_train_score, pos_score, neg_score, avg_score, auc_score))
  return results

def _new_fit(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[ # type: ignore
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
    ):
        logging.debug("UPDATED _FIT")
        lpt = EdgePredictionTransformer(
            methods=self._edge_embedding_methods,
            aligned_mapping=True,
            include_both_undirected_edges=False,
        )
        lpt.fit(
            node_features,
            node_type_feature=node_type_features,
            edge_type_features=edge_type_features,
        )

        if support is None:
            support = graph

        if edge_features is None:
            edge_features: List[Type[AbstractEdgeFeature]] = []

        for edge_feature in edge_features: # type: ignore
            if not issubclass(type(edge_feature), AbstractEdgeFeature):
                raise NotImplementedError(
                    f"Edge features of type {type(edge_feature)} are not supported."
                    "We currently only support edge features of type AbstractEdgeFeature."
                )

        number_of_negative_samples = int(
            math.ceil(
                graph.get_number_of_directed_edges() * self._training_unbalance_rate
            )
        )
        logging.debug(graph.get_number_of_directed_edges())
        logging.debug(number_of_negative_samples)
        
        # custom code
        if self._original_graph:
            # generate the negative graph from the original graph instead of the negative graph
            negative_graph = self._original_graph.sample_negative_graph(
                number_of_negative_samples=number_of_negative_samples,
                only_from_same_component=True,
                random_state=self._random_state,
                use_scale_free_distribution=self._use_scale_free_distribution,
                sample_edge_types=len(edge_type_features) > 0, # type: ignore
                source_node_types_names = [self._pair_to_predict[0]] if self._pair_to_predict else None,
                destination_node_types_names = [self._pair_to_predict[1]] if self._pair_to_predict else None,
            )
        else:
            # default GRAPE code
            negative_graph = graph.sample_negative_graph(
                number_of_negative_samples=number_of_negative_samples,
                only_from_same_component=True,
                random_state=self._random_state,
                use_scale_free_distribution=self._use_scale_free_distribution,
                sample_edge_types=len(edge_type_features) > 0, # type: ignore
                source_node_types_names = [self._pair_to_predict[0]] if self._pair_to_predict else None,
                destination_node_types_names = [self._pair_to_predict[1]] if self._pair_to_predict else None,
            )
        
        # debug stuff
        # # create df with the negative graph edge data to explore what is inside
        # df_negative_graph_triples = helper_lib.graph.build_triples_df(negative_graph).drop_duplicates()
        
        # # calcualate the intersection between the view0 dataframe and the negative dataframe to check for overlap
        # intersection_neg = pd.merge(df_view, df_negative_graph_triples, how='inner', on=['source','destination'])
        # print(f"Percentage of negative train set that is actually positive: {len(intersection_neg)/len(df_negative_graph_triples)}")
        # print(f"Absolute value: {len(intersection_neg)}")
        # false_negatives.append(len(intersection_neg)/len(df_negative_graph_triples))
        
        
        #print(f"Positive training set size: {len(df_positive_graph_triples)}")
        # print relative frequencies of top 10 source_type - destination_type in train and test set compared to full graph
        #for i,(key,count) in enumerate(df_positive_graph_triples['complete_label'].value_counts().items()):
        #    print(f"{key}: {count}")
        #    if i == 9 :
        #        break
        #print(f"Negative training set size: {len(df_negative_graph_triples)}")
        # print relative frequencies of top 10 source_type - destination_type in train and test set compared to full graph
        #for i,(key,count) in enumerate(df_negative_graph_triples['complete_label'].value_counts().items()):
        #    print(f"{key}: {count}")
        #    if i == 9 :
        #        break
        
        # does not work properly
        # if(len(intersection_neg)>0):
        #     # remove positive edges from negative test set
        #     print("Removing positive edges from negative graph")
        #     # can't filter by edge id since it could make the graph directed (error from grape)
        #     edge_ids_to_remove = intersection_neg['edge_y'].unique()
        #     negative_graph = negative_graph.to_directed()
        #     negative_graph = negative_graph.filter_from_ids(edge_ids_to_remove=edge_ids_to_remove)
        #     negative_graph = negative_graph.to_undirected()
        #     # negative_graph = negative_graph.filter_from_ids(edge_ids_to_keep=edge_ids_to_keep)
            
        #     df_negative_graph_triples = helper_lib.graph.build_triples_df(negative_graph).drop_duplicates()
        #     intersection_neg = pd.merge(df_view, df_negative_graph_triples, how='inner', on=['source','destination'])
        #     print(f"[CLEANED] Percentage of negative train set that is actually positive: {len(intersection_neg)/len(df_negative_graph_triples)}")
        #     print(f"[CLEANED] Absolute value: {len(intersection_neg)}")

        assert negative_graph.has_edges()

        # the assert negative_graph.has_selfloops() from GRAPE almost always causes an exception 
        if negative_graph.has_selfloops():
            # assert graph.has_selfloops(), (
            #     "The negative graph contains self loops, "
            #     "but the positive graph does not."
            # )
            if not graph.has_selfloops():
                logging.warning("WARNING: The negative graph contains self loops, but the positive graph does not.")

        if self._training_unbalance_rate == 1.0:
            number_of_negative_edges = negative_graph.get_number_of_directed_edges()
            number_of_positive_edges = graph.get_number_of_directed_edges()
            self_loop_message = (
                ("The graph contains self loops.")
                if negative_graph.has_selfloops()
                else ("The graph does not contain self loops.")
            )
            if number_of_negative_edges not in (
                number_of_positive_edges + 1,
                number_of_positive_edges,
            ): 
                print(
                "The negative graph should have the same number of edges as the "
                "positive graph when using a training unbalance rate of 1.0. "
                "We expect the negative graph to have "
                f"{number_of_positive_edges} or {number_of_positive_edges + 1} edges, but found "
                f"{number_of_negative_edges}. {self_loop_message} "
                f"The exact number requested was {number_of_negative_samples}"
            )

        rasterized_edge_features = []

        for edge_feature in edge_features: # type: ignore
            for positive_edge_features, negative_edge_features in zip(
                edge_feature.get_edge_feature_from_graph(
                    graph=graph,
                    support=support,
                ).values(),
                edge_feature.get_edge_feature_from_graph(
                    graph=negative_graph,
                    support=support,
                ).values(),
            ):
                rasterized_edge_features.append(
                    np.vstack((positive_edge_features, negative_edge_features))
                )

        if self._use_edge_metrics:
            rasterized_edge_features.append(
                np.vstack(
                    (
                        support.get_all_edge_metrics( # type: ignore
                            normalize=True,
                            subgraph=graph,
                        ),
                        support.get_all_edge_metrics( # type: ignore
                            normalize=True,
                            subgraph=negative_graph,
                        ),
                    )
                )
            )

        self._model_instance.fit(
            *lpt.transform(
                positive_graph=graph,
                negative_graph=negative_graph,
                edge_features=rasterized_edge_features,
                shuffle=True,
                random_state=self._random_state,
            )
        )

def update_fit(edge_prediction_model, pair_to_predict, original_graph):
    edge_prediction_model._pair_to_predict = pair_to_predict
    edge_prediction_model._original_graph = original_graph
    edge_prediction_model._fit = MethodType(_new_fit, edge_prediction_model)
    return edge_prediction_model
  
def edge_pred_pairs(graph, embedder, edge_pred_model, pairs_to_predict, 
                    name_for_df=None, clear_output=False, 
                    train_size=0.7, number_of_holdouts=5, seed=42,
                    use_scale_free_distribution=True, train_on_filtered=True):
    df_results = pd.DataFrame()
    columns = ['Graph','Embedder','Model','Source Type','Destination Type', 
        'Train on filtered', 'Training set size', 'Testing set size', 
        'Positive training balanced accuracy', 'Positive balanced accuracy',
        'Negative balanced accuracy','Mean balanced accuracy','AUC'
    ]

    for i,pair_to_predict in enumerate(pairs_to_predict):
        print(f"Predicting pair: {pair_to_predict} ({i+1}/{len(pairs_to_predict)})")
        
        # change how the training behaves
        update_fit(edge_pred_model, pair_to_predict if train_on_filtered else None, graph)
        # negative graph extracted from the full graph to avoid false negatives
        # model will only be trained on data of the relevant type pair to predict
        results_custom_filtered_train = edge_prediction_pipeline(
            graph, edge_pred_model, embedder, pair_to_predict, 
            train_on_filtered=train_on_filtered,
            train_size=train_size, number_of_holdouts=number_of_holdouts, seed=seed, verbose=False,
            clear_output_holdout=clear_output,
            use_scale_free_distribution=use_scale_free_distribution
        )
        df_results_custom_filtered_train = pd.DataFrame(results_custom_filtered_train)
        df_results = pd.concat([df_results,df_results_custom_filtered_train])
            
    df_results.columns = columns
    df_results['edge_pred_model'] = edge_pred_model.model_name()
    df_results['embedding_model'] = embedder.model_name()
    df_results['name'] = f'{embedder.model_name()}-{edge_pred_model.model_name()}' if name_for_df is None else name_for_df
    # results[f'{embedder.model_name()}-{model.model_name()}'] = df_results
    return df_results