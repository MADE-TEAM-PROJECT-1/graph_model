# Description
Graph embedding model is built using [AmpliGraph](https://docs.ampligraph) library.

The graph consist from:
- entities (vertices): articles and authors.
- relations (edges):
    - REFERENCE: edge between $article_1$ and $article_2$ if $article_1$ has reference to $article_2$;
    - AUTHOR: edge between $author$ and $article$ if $author$ is and author of $article$;
    - COAUTHOR: edge between $author_1$ and $author_2$ if they are coauthors (exist an article where they are authors).

Dataset for model training consist of triples of three types as described above.

After model is trained we can explore knowledge graph:
- get embeddings of entities (of article and author);
- discover new facts: to find new relations (edges) in graph;
- queries the model with two elements of a triple and returns the `top_n` results of all possible completions ordered by score predicted by the model;
- return the nearest neighbors of entities.

# Final model
- is trained for 100 epochs with embedding size 10;
- knowledge graph is extracted from:
    - 10763244 REFERENCE edges;
    - 3414268 COAUTHOR edges;
    - 3236682 AUTHOR edges;
- consists of 17414194 embeddings of articles and authors;
- example of retrieving embedding for articles and authors you can find in [get_embeddings.py](get_embeddings.py)
- can be downloaded from [gdrive](https://drive.google.com/file/d/1NQsRtoii30h-MkkFbipGKDsrPmQTEY8Y)

    or using `gdown`
    ```shell
    gdown --id 1NQsRtoii30h-MkkFbipGKDsrPmQTEY8Y
    ```

# Embeddings
You can download embeddings (dictionary with ids and emb as keys and values respectively) with `emb_dim=10` from gdrive:
- [articles_embeddings](https://drive.google.com/file/d/1T6qhVNpnzhOcJzM5CIH4sZ52OAumIXs3)
- [authors_embeddings](https://drive.google.com/file/d/1XWZYopltvEPUv6-p5FLMCxQAIjzlqJYP)

To load embeddings you can use the following function:
```python
def load_numpy(filename: str) -> Dict[str, np.ndarray]:
    with open(filename, 'rb') as f:
        return np.load(f)
```