import pandas as pd
from grape import EmbeddingResult
import os

embedding_cache_dir = './cache/embeddings/'
def cache_embedding(embedding:EmbeddingResult,filename:str):
  embedding.embedding_method_name
  for i in range(embedding.number_of_embeddings()):
    emb_i = embedding.get_node_embedding_from_index(i)
    emb_i.to_pickle(embedding_cache_dir + f'{filename}_{i}.pkl')

def load_embedding(filename:str,embedding_index:int=0):
  #print(embedding_cache_dir + f'{filename}_{embedding_index}.pkl')
  return pd.read_pickle(embedding_cache_dir + f'{filename}_{embedding_index}.pkl')

def load_embeddings(filename:str):
  embeddings = []
  i = 0
  while True:
    try:
      embeddings.append(load_embedding(filename,i))
      i += 1
    except:
      break
  if len(embeddings) == 0:
    raise Exception('No embeddings found')
  return embeddings

def load_or_embed(filename:str,embedding_method,graph):
  if is_embedding_cached(filename):
    print('Loading cached embedding')
    return load_embeddings(filename)
  else:
    print('Embedding graph')
    embedding = embedding_method.fit_transform(graph)
    print('Caching embedding')
    cache_embedding(embedding,filename)
    return embedding

def is_embedding_cached(filename:str):
  #print(embedding_cache_dir + f'{filename}_0.pkl')
  return os.path.isfile(embedding_cache_dir + f'{filename}_0.pkl')

def set_embedding_cache_dir(path:str):
  global embedding_cache_dir
  embedding_cache_dir = path