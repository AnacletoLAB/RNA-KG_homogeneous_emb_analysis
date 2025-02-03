from grape import GraphVisualizer
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

# plt.style.use('tableau-colorblind10')
cycler_colors = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cycler_colors)

scatter_kwargs = {'s':5, 'alpha':1, 'linewidths':0, 'rasterized':True}

def plot_node_types(graph, embedding_2d, node_types, axes, scatter_kwargs={}, show_legend=True, k=0, types_to_show=[],to_csv=None):
  df = pd.DataFrame({'embeddingX':embedding_2d[:,0],'embeddingY':embedding_2d[:,1],'node_type':node_types})
  if to_csv is not None:
    df_to_save = df.copy()
    df_to_save['node_type_label'] = df_to_save['node_type'].apply(lambda x: graph.get_node_type_name_from_node_type_id(x))
    df_to_save.to_csv(to_csv,index=False)
  #print(len(df)) 
  # graph.get_node_type_name_from_node_type_id(node_id)
  if len(types_to_show) > 0:
    types_to_show_ids = [graph.get_node_type_id_from_node_type_name(type_name) for type_name in types_to_show]
    df = df[df['node_type'].isin(types_to_show_ids)]
    # print warning if some types are not found
    types_found = set(df['node_type'].apply(lambda x: graph.get_node_type_name_from_node_type_id(x)).unique())
    types_not_found =  set(types_to_show) - types_found
    if len(types_not_found)>0:
      logging.warning(f"Some types you wanted to plot weren't found in the subsampled graph: {types_not_found}")


  #print(df)
  #print(len(df))
  initial_number_of_types = 0

  if k > 0:
    # show only the top k types by number of nodes and group the rest as 'others'
    node_type_counts = df['node_type'].value_counts()
    node_type_counts = node_type_counts.sort_values(ascending=False)
    #print(node_type_counts)
    #print('There are',len(node_type_counts),'node types.')
    #print(df['node_type'].unique())
    initial_number_of_types = len(node_type_counts)
    node_type_counts = node_type_counts[:k]
    df['node_type'] = df['node_type'].apply(lambda x: x if x in node_type_counts.index else -1)
  #print(df)

  num_colors_needed = len(df['node_type'].unique())
  # print(num_colors_needed)
  if num_colors_needed > 10:
    colors = cm.viridis(np.linspace(0, 1, num_colors_needed))

  plot_order = df['node_type'].value_counts().index

  for i in range(len(plot_order)):
    node_type_id = plot_order[i]
    if node_type_id == -1:
      # will be plotted last
      continue
    df_type = df[df['node_type']==node_type_id]
    if num_colors_needed <=10:
      axes.scatter(df_type['embeddingX'],df_type['embeddingY'],
        label=graph.get_node_type_name_from_node_type_id(node_type_id),
        # color=colors[i],
        **scatter_kwargs,
      )
    else:
      axes.scatter(df_type['embeddingX'],df_type['embeddingY'],
        label=graph.get_node_type_name_from_node_type_id(node_type_id),
        color=colors[i],
        **scatter_kwargs,
      )
  
  if k>0:
    node_type_label = 'Other'+f'({initial_number_of_types-k})'
    df_type = df[df['node_type']==-1]
    if len(df_type)>0:
      if num_colors_needed<=10:
        axes.scatter(df_type['embeddingX'],df_type['embeddingY'],
          label=node_type_label,
          # color=colors[-1],
          **scatter_kwargs,
        )
      else:
        axes.scatter(df_type['embeddingX'],df_type['embeddingY'],
          label=node_type_label,
          color=colors[-1],
          **scatter_kwargs,
        )

    


    # node_type_counts = df['node_type'].value_counts()
    # node_type_counts = node_type_counts.sort_values(ascending=False)
    # node_type_counts = node_type_counts[:k]
    # print(node_type_counts.index)
    # print(node_type_labels)
    # df['node_type'] = df['node_type'].apply(lambda x: x if x in node_type_counts.index else -1)
    # node_type_labels = [node_type_labels[i] for i in range(len(node_type_labels)) if i in node_type_counts.index]
    # node_type_ids = [node_type_ids[i] for i in range(len(node_type_labels)) if i in node_type_counts.index]

  # num_colors_needed = len(node_type_ids) if k==0 else len(node_type_ids)+1
  # colors = cm.viridis(np.linspace(0, 1, num_colors_needed))

  # for i in range(len(node_type_ids)):
  #   node_type_id = node_type_ids[i]
  #   node_type_label = node_type_labels[i]
  #   df_type = df[df['node_type']==node_type_id]
  #   axes.scatter(df_type['embeddingX'],df_type['embeddingY'],
  #     label=node_type_label,
  #     color=colors[i],
  #     **scatter_kwargs,
  #   )
  
  # if k>0:
  #   node_type_label = 'Other'+f'({initial_node_type_num-k})'
  #   df_type = df[df['node_type']==-1]
  #   if len(df_type)>0:
  #     axes.scatter(df_type['embeddingX'],df_type['embeddingY'],
  #       label=node_type_label,
  #       color=colors[-1],
  #       **scatter_kwargs,
  #     )

  if show_legend:
    axes.legend(markerscale=3)
  axes.axis('off')

def build_triples_df(embedding_2d, edge_types,subsampled_graph):
  df = pd.DataFrame({'embeddingX':embedding_2d[:,0],'embeddingY':embedding_2d[:,1],'edge_type':edge_types})
  if subsampled_graph.is_directed():
    edge_node_ids = (
      subsampled_graph.get_directed_source_node_ids(),
      subsampled_graph.get_directed_destination_node_ids(),
    )
  else:
    edge_node_ids = (
      subsampled_graph.get_source_node_ids(directed=False),
      subsampled_graph.get_destination_node_ids(directed=False),
    )
  sources = edge_node_ids[0]
  destinations = edge_node_ids[1]
  sources_types = [subsampled_graph.get_node_type_ids_from_node_id(node_id)[0] for node_id in sources]
  sources_types_labels = [subsampled_graph.get_node_type_name_from_node_type_id(node_type_id) for node_type_id in sources_types]
  destinations_types = [subsampled_graph.get_node_type_ids_from_node_id(node_id)[0] for node_id in destinations]
  destinations_types_labels = [subsampled_graph.get_node_type_name_from_node_type_id(node_type_id) for node_type_id in destinations_types]
  edge_types_labels = [subsampled_graph.get_edge_type_name_from_edge_type_id(edge_type_id) for edge_type_id in edge_types]
  df['source'] = sources
  df['destination'] = destinations
  df['source_type'] = sources_types
  df['edge_type'] = edge_types
  df['destination_type'] = destinations_types
  df['source_type_label'] = sources_types_labels
  df['edge_type_label'] = edge_types_labels
  df['destination_type_label'] = destinations_types_labels
  df['complete_label'] = df['source_type_label'] + ' - ' + df['edge_type_label'] + ' - ' + df['destination_type_label']
  return df


def plot_edge_types(graph, embedding_2d, edge_types, axes, subsampled_graph, scatter_kwargs={}, show_legend=True, k=0, types_to_show=[],to_csv=None, triples_filter=None):
  df = build_triples_df(embedding_2d, edge_types,subsampled_graph)
  # print(f"Dataframe len: {len(df)}")
  
  if to_csv is not None:
    df_to_save = df.copy()
    df_to_save['edge_type_label'] = df_to_save['edge_type'].apply(lambda x: graph.get_edge_type_name_from_edge_type_id(x))
    df_to_save.to_csv(to_csv,index=False)

  if triples_filter is not None:
    filtered_df = pd.DataFrame()
    for (source_type, edge_type, destination_type) in triples_filter:
      # print(f"Filtering triples with source type: {source_type}, edge type: {edge_type}, destination type: {destination_type}")
      # convert types to ids
      source_type_id = graph.get_node_type_id_from_node_type_name(source_type) if source_type is not None else None
      edge_type_id = graph.get_edge_type_id_from_edge_type_name(edge_type) if edge_type is not None else None
      destination_type_id = graph.get_node_type_id_from_node_type_name(destination_type) if destination_type is not None else None
      source_type_id_filter = df['source_type']==source_type_id if source_type_id is not None else True
      edge_type_id_filter = df['edge_type']==edge_type_id if edge_type_id is not None else True
      destination_type_id_filter = df['destination_type']==destination_type_id if destination_type_id is not None else True
      filtered_df = pd.concat([filtered_df,df[source_type_id_filter & edge_type_id_filter & destination_type_id_filter]])
    df = filtered_df
  # print(f"Filtered dataframe len: {len(df)}")

  
  if len(types_to_show) > 0:
    types_to_show_ids = [graph.get_edge_type_id_from_edge_type_name(type_name) for type_name in types_to_show]
    df = df[df['edge_type'].isin(types_to_show_ids)]
    # print warning if some types are not found
    types_found = set(df['edge_type'].apply(lambda x: graph.get_edge_type_name_from_edge_type_id(x)).unique())
    types_not_found =  set(types_to_show) - types_found
    if len(types_not_found)>0:
      logging.warning(f"Some types you wanted to plot weren't found in the subsampled graph: {types_not_found}")
    # print(df)

  initial_number_of_types = 0

  if k > 0:
    # show only the top k types by number of nodes and group the rest as 'others'
    edge_type_counts = df['edge_type'].value_counts()
    edge_type_counts = edge_type_counts.sort_values(ascending=False)
    initial_number_of_types = len(edge_type_counts)
    edge_type_counts = edge_type_counts[:k]
    df['edge_type'] = df['edge_type'].apply(lambda x: x if x in edge_type_counts.index else -1)

  num_colors_needed = len(df['edge_type'].unique())
  # print(num_colors_needed)
  if num_colors_needed > 10:
    colors = cm.viridis(np.linspace(0, 1, num_colors_needed))
  plot_order = df['edge_type'].value_counts().index

  for i in range(len(plot_order)):
    edge_type_id = plot_order[i]
    if edge_type_id == -1:
      # will be plotted last
      continue
    df_type = df[df['edge_type']==edge_type_id]
    if num_colors_needed > 10:
      axes.scatter(df_type['embeddingX'],df_type['embeddingY'],
        label=graph.get_edge_type_name_from_edge_type_id(edge_type_id),
        color=colors[i],
        **scatter_kwargs,
      )
    else:
      axes.scatter(df_type['embeddingX'],df_type['embeddingY'],
        label=graph.get_edge_type_name_from_edge_type_id(edge_type_id),
        **scatter_kwargs,
      )
  
  if k>0:
    edge_type_label = 'Other'+f'({initial_number_of_types-k})'
    df_type = df[df['edge_type']==-1]
    if len(df_type)>0:
      if num_colors_needed > 10:
        axes.scatter(df_type['embeddingX'],df_type['embeddingY'],
          label=edge_type_label,
          color=colors[i],
          **scatter_kwargs,
        )
      else:
        axes.scatter(df_type['embeddingX'],df_type['embeddingY'],
          label=edge_type_label,
          **scatter_kwargs,
        )

  if show_legend:
    axes.legend(markerscale=3)
  axes.axis('off')

def plot_triples_types(graph, embedding_2d, edge_types, axes, subsampled_graph, triples_filter=None, scatter_kwargs={}, show_legend=True):
  df = build_triples_df(embedding_2d, edge_types,subsampled_graph)
  # print(f"Dataframe len: {len(df)}")

  if triples_filter is not None:
    filtered_df = pd.DataFrame()
    for (source_type, edge_type, destination_type) in triples_filter:
      # print(f"Filtering triples with source type: {source_type}, edge type: {edge_type}, destination type: {destination_type}")
      # convert types to ids
      source_type_id = graph.get_node_type_id_from_node_type_name(source_type) if source_type is not None else None
      edge_type_id = graph.get_edge_type_id_from_edge_type_name(edge_type) if edge_type is not None else None
      destination_type_id = graph.get_node_type_id_from_node_type_name(destination_type) if destination_type is not None else None
      source_type_id_filter = df['source_type']==source_type_id if source_type_id is not None else True
      edge_type_id_filter = df['edge_type']==edge_type_id if edge_type_id is not None else True
      destination_type_id_filter = df['destination_type']==destination_type_id if destination_type_id is not None else True
      filtered_df = pd.concat([filtered_df,df[source_type_id_filter & edge_type_id_filter & destination_type_id_filter]])
    df = filtered_df
    
  # print(f"Filtered dataframe len: {len(df)}")

  num_colors_needed = len(df['complete_label'].unique())
  # print(num_colors_needed)
  if num_colors_needed > 10:
    colors = cm.viridis(np.linspace(0, 1, num_colors_needed))
  plot_order = df['complete_label'].value_counts().index

  for i in range(len(plot_order)):
    complete_type_id = plot_order[i]
    if edge_type_id == -1:
      # will be plotted last
      continue
    df_type = df[df['complete_label']==complete_type_id]
    label = df_type['complete_label'].iloc[0]
    if num_colors_needed > 10:
      axes.scatter(df_type['embeddingX'],df_type['embeddingY'],
        label=label,
        color=colors[i],
        **scatter_kwargs,
      )
    else:
      axes.scatter(df_type['embeddingX'],df_type['embeddingY'],
        label=label,
        **scatter_kwargs,
      )

  if show_legend:
    axes.legend(markerscale=3)
  axes.axis('off')

def plot_embedding(embeddings, sub_plot_titles, graphs, visualization_type,
    save_path=None, figsize=(15,7.5), formats=['pdf','jpeg'],
    to_csv=None,
    title=None, 
    show=True, 
    types_to_show=[], 
    k=0, 
    triples_filter=None,
    show_legend=False,
    number_of_subsampled_nodes=20_000,
    number_of_subsampled_edges=20_000,
    legend_cols=4,
):
  n_embeddings = len(embeddings)
  fig, axes = plt.subplots(1,n_embeddings,figsize=figsize)
  visualizers = []
  handles = None
  labels = None
  show_legend_subplot = True
  for i, embedding in enumerate(embeddings):
    if n_embeddings>1:
      ax = axes[i]
    else:
      ax = axes
    visualizer = GraphVisualizer(graphs[i], decomposition_method="TSNE", automatically_display_on_notebooks=False, number_of_subsampled_nodes=number_of_subsampled_nodes,number_of_subsampled_edges=number_of_subsampled_edges, edge_embedding_methods="Hadamard")
    # print(np.bincount(node_types))
    # print(len(np.bincount(node_types)))
    # print(np.sum(np.bincount(node_types)))
    if(visualization_type == 'node_types'):
      visualizer.fit_nodes(embedding)
      node_embeddings = visualizer._node_decomposition
      node_types = visualizer._get_flatten_multi_label_and_unknown_node_types()
      plot_node_types(graphs[i],node_embeddings, node_types, ax, show_legend=show_legend_subplot, types_to_show=types_to_show, k=k, to_csv=to_csv, scatter_kwargs=scatter_kwargs)
    elif(visualization_type == 'edge_types'):
      visualizer.fit_edges(embedding)
      node_embeddings = visualizer._positive_edge_decomposition
      node_types = visualizer._get_flatten_unknown_edge_types()
      subsampled_graph = visualizer._positive_graph
      plot_edge_types(graphs[i],node_embeddings, node_types, ax, subsampled_graph=subsampled_graph,
        show_legend=show_legend_subplot, 
        types_to_show=types_to_show, k=k, triples_filter=triples_filter,
        to_csv=to_csv, 
        scatter_kwargs=scatter_kwargs)
    elif(visualization_type == 'triples_types'):
      visualizer.fit_edges(embedding)
      node_embeddings = visualizer._positive_edge_decomposition
      node_types = visualizer._get_flatten_unknown_edge_types()
      subsampled_graph = visualizer._positive_graph
      plot_triples_types(graphs[i],node_embeddings, node_types, ax, subsampled_graph=subsampled_graph,
        show_legend=show_legend_subplot, triples_filter=triples_filter,
        scatter_kwargs=scatter_kwargs)
    else:
      logging.error(f"Visualization type {visualization_type} not supported, please use 'node_types', 'edge_types' or 'triples_types'")
    
    if(handles==None):
      show_legend_subplot=False
      handles = ax.get_legend().legend_handles
      labels = [text.get_text() for text in ax.get_legend().get_texts()]
      ax.get_legend().remove()
    ax.set_title(sub_plot_titles[i],fontsize=20)
    visualizers.append(visualizer)
  
  if show_legend: 
    # legend = fig.legend(handles, labels , loc='lower center', ncol=5)
    # print(labels)
    legend = fig.legend(handles, labels , loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=legend_cols)
    # fix the alpha of the legend
    for legend_handle in legend.legendHandles:
      legend_handle.set_alpha(1)
      try:
        legend_handle._legmarker.set_alpha(1)
      except AttributeError:
        pass

  if title is not None:
    plt.suptitle(title)

  plt.tight_layout()

  if save_path is not None:
    for format in formats:
      plt.savefig(f"{save_path}.{format}",dpi=300,bbox_inches='tight')
  
  if show:
    plt.show()
  else:
    plt.close()  
  
  return visualizers