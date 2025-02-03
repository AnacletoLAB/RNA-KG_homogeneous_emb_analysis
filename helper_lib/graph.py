from grape import Graph
import os 
import numpy as np
import pandas as pd
dir_path = os.path.dirname(os.path.realpath(__file__))

rnakg_node_path = "./RNA-KG/Zenodo2New/nodes.csv"
rnakg_edge_path = "./RNA-KG/Zenodo2New/edges.csv"

views_path = "./RNA-KG/Views/"

def load_rnakg(directed=False):
    print(os.getcwd())
    RNAKG = Graph.from_csv(
        # Edges related parameters
        ## The path to the edges list csv
        edge_path=rnakg_edge_path,
        ## Set the comma as the separator between values
        edge_list_separator=",",
        ## The first rows should be used as the columns names
        edge_list_header=True,
        ## The source nodes are in the subject column
        sources_column="subject",
        ## The source nodes are in the object column
        destinations_column="object",
        ## The source nodes are in the subject column
        edge_list_edge_types_column="type",
        load_edge_list_in_parallel=False,

        # Nodes related parameters
        ## The path to the nodes list csv
        node_path=rnakg_node_path,
        ## Set the comma as the separator between values
        node_list_separator=",",
        ## The first rows should be used as the columns names
        node_list_header=True,
        ## The column with the node name is the one with name "name".
        nodes_column="name",
        ## The column with the node type is the one with name "type".
        node_list_node_types_column="type",
        load_node_list_in_parallel=False,

        # Graph related parameters
        directed=directed,
        name="directedRNA-KG" if directed else "undirectedRNA-KG",
    )
    return RNAKG

def load_rnakg_fixed(directed=False):
    print(os.getcwd())
    RNAKG = Graph.from_csv(
        # Edges related parameters
        ## The path to the edges list csv
        edge_path='./RNA-KG/Zenodo2New2/edges.csv',
        ## Set the comma as the separator between values
        edge_list_separator=",",
        ## The first rows should be used as the columns names
        edge_list_header=True,
        ## The source nodes are in the subject column
        sources_column="subject",
        ## The source nodes are in the object column
        destinations_column="object",
        ## The source nodes are in the subject column
        edge_list_edge_types_column="type",
        load_edge_list_in_parallel=False,

        # Nodes related parameters
        ## The path to the nodes list csv
        node_path='./RNA-KG/Zenodo2New2/nodes.csv',
        ## Set the comma as the separator between values
        node_list_separator=",",
        ## The first rows should be used as the columns names
        node_list_header=True,
        ## The column with the node name is the one with name "name".
        nodes_column="name",
        ## The column with the node type is the one with name "type".
        node_list_node_types_column="type",
        load_node_list_in_parallel=False,

        # Graph related parameters
        directed=directed,
        name="directedRNA-KG" if directed else "undirectedRNA-KG",
    )
    return RNAKG

def _load_rnakg_fixed(version, directed=False):
    if version==2:
        edge_path='./RNA-KG/Zenodo2New2/edges.csv'
        node_path='./RNA-KG/Zenodo2New2/nodes.csv'
    elif version==9:
        edge_path='./RNA-KG/Zenodo9/edges.csv'
        node_path='./RNA-KG/Zenodo9/nodes.csv'

    RNAKG = Graph.from_csv(
        # Edges related parameters
        ## The path to the edges list csv
        edge_path=edge_path,
        ## Set the comma as the separator between values
        edge_list_separator=",",
        ## The first rows should be used as the columns names
        edge_list_header=True,
        ## The source nodes are in the subject column
        sources_column="subject",
        ## The source nodes are in the object column
        destinations_column="object",
        ## The source nodes are in the subject column
        edge_list_edge_types_column="type",
        load_edge_list_in_parallel=False,

        # Nodes related parameters
        ## The path to the nodes list csv
        node_path=node_path,
        ## Set the comma as the separator between values
        node_list_separator=",",
        ## The first rows should be used as the columns names
        node_list_header=True,
        ## The column with the node name is the one with name "name".
        nodes_column="name",
        ## The column with the node type is the one with name "type".
        node_list_node_types_column="type",
        load_node_list_in_parallel=False,

        # Graph related parameters
        directed=directed,
        name=f"directed_RNA-KG_{version}" if directed else f"undirected_RNA-KG_{version}",
    )
    return RNAKG

def load_view_rnakg(number, directed=False, train=False):
    view_location = views_path + "view" + str(number) + "/"
    if train:
        view_location+='train/'
    nodes_df_path = view_location + "nodes.pkl"
    edges_df_path = view_location + "edges.pkl"
    nodes_df = pd.read_pickle(nodes_df_path)
    edges_df = pd.read_pickle(edges_df_path)
    view_graph = Graph.from_pd(
        edges_df=edges_df,
        nodes_df=nodes_df,
        node_name_column="name",
        node_type_column="type",
        edge_src_column="subject",
        edge_dst_column="object",
        edge_type_column="predicate",
        directed=directed,
        name="RNA-KG view" + str(number)+ ("train" if train==True else ""),
    )
    return view_graph


def get_node_types_grape(graph) -> np.ndarray:
    """Returns flattened node type IDs adjusted for the current instance."""
    # The following is needed to normalize the multiple types
    node_types_counts = graph.get_node_type_id_counts_hashmap()
    top_10_node_types = {
        node_type: 50 - i
        for i, node_type in enumerate(
            sorted(node_types_counts.items(), key=lambda x: x[1], reverse=True)[:50]
        )
    }
    node_types_counts = {
        node_type: top_10_node_types.get(node_type, 0)
        for node_type in node_types_counts
    }
    node_types_number = graph.get_number_of_node_types()
    unknown_node_types_id = node_types_number

    # When we have multiple node types for a given node, we set it to
    # the most common node type of the set.
    return np.fromiter(
        (
            unknown_node_types_id
            if node_type_ids is None
            else sorted(
                node_type_ids,
                key=lambda node_type: node_types_counts[node_type],
                reverse=True,
            )[0]
            for node_type_ids in (
                graph.get_node_type_ids_from_node_id(node_id)
                for node_id in range(graph.get_number_of_nodes())
            )
        ),
        dtype=np.uint32,
    )

def get_edge_types_grape(graph) -> np.ndarray:
    """Returns flattened edge type IDs adjusted for the current instance."""
    # The following is needed to normalize the unknown types
    unknown_edge_types_id = graph.get_number_of_edge_types()
    # When we have multiple node types for a given node, we set it to
    # the most common node type of the set.
    return np.fromiter(
        (
            unknown_edge_types_id if edge_type_id is None else edge_type_id
            for edge_type_id in (
                graph.get_directed_edge_type_ids()
                if graph.is_directed()
                else graph.get_upper_triangular_edge_type_ids()
            )
        ),
        dtype=np.uint32,
    )

def check_if_in_graph(graph,src_node,dst_node,pair_to_predict):
  try:
    # check if the node type match
    src_node_type = graph.get_node_type_name_from_node_name(src_node)[0]
    dst_node_type = graph.get_node_type_name_from_node_name(dst_node)[0]
    # print(src_node_type)
    # print(dst_node_type)
    if src_node_type not in pair_to_predict:
        print('Wrong source type')
        return False
    if dst_node_type not in pair_to_predict:
        print('Wrong destiantion type')
        return False
    if src_node_type==pair_to_predict[0] and dst_node_type !=pair_to_predict[1]:
        print('Wrong type combination 1')
        return False
    if src_node_type==pair_to_predict[1] and dst_node_type !=pair_to_predict[0]:
        print('Wrong type combination 2')
        return False
    # print('Type combination: OK!')
    # check if the edge exists
    graph.get_edge_id_from_node_names(src_node,dst_node)
    return True
  except:
    return False
  
def build_triples_df(graph):
  df = pd.DataFrame()
  if graph.is_directed():
    edge_node_ids = (
      graph.get_directed_source_node_ids(),
      graph.get_directed_destination_node_ids(),
    )
  else:
    edge_node_ids = (
      graph.get_source_node_ids(directed=False),
      graph.get_destination_node_ids(directed=False),
    )
  sources = edge_node_ids[0]
  destinations = edge_node_ids[1]
  sources_types = [graph.get_node_type_ids_from_node_id(node_id)[0] for node_id in sources]
  sources_types_labels = [graph.get_node_type_name_from_node_type_id(node_type_id) for node_type_id in sources_types]
  destinations_types = [graph.get_node_type_ids_from_node_id(node_id)[0] for node_id in destinations]
  destinations_types_labels = [graph.get_node_type_name_from_node_type_id(node_type_id) for node_type_id in destinations_types]

  edge_node_ids = graph.get_edge_node_ids(directed=graph.is_directed())
  edge_ids = [graph.get_edge_id_from_node_ids(node_ids[0],node_ids[1]) for node_ids in edge_node_ids]
  edge_types = [graph.get_edge_type_id_from_edge_id(edge_id) for edge_id in edge_ids]
  edge_type_labels = [graph.get_edge_type_name_from_edge_id(edge_id) for edge_id in edge_ids]
  
  df['source'] = sources
  df['destination'] = destinations
  df['edge'] = edge_ids
  df['source_type'] = sources_types
  df['destination_type'] = destinations_types
  df['edge_type'] = edge_types
  df['source_type_label'] = sources_types_labels
  df['destination_type_label'] = destinations_types_labels
  df['edge_type_label'] = edge_type_labels
  df['complete_label'] = df['source_type_label'] + ' - ' + df['destination_type_label']
  return df