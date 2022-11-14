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