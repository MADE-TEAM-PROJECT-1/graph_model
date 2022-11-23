from ampligraph.utils import restore_model
from ampligraph.discovery import find_nearest_neighbours
from numpy import ndarray
from typing import List, Tuple


def get_embeddings(
    model, entities: List[str], n_neighbors: int, metric: str = "cosine"
) -> Tuple[ndarray, ndarray]:
    """
    @model: fitted graph model
    @entities: list of entities id's (article of author)
    @n_neighbors: number of nearest neighbors to be returned
    @metric: metric for nearest neighboors search
    return: (neighbors, dist), where neighbors and dist are ndarrays of shape (len(entities), n_neighbors)
            where neighboors is ndarray of n_neighbors for each entity from entities
            and ndarray of distances
    """
    neighbors, dist = find_nearest_neighbours(
        model, entities=entities, n_neighbors=n_neighbors, metric=metric
    )
    return neighbors, dist


def main():
    model_path = "graph_model.pkl"
    model = restore_model(model_path)
    articles_ids = ["53e99784b7602d9701f3e151", "53e99784b7602d9701f3e15d"]
    authors_ids = ["53f46797dabfaeb22f542630", "53f42e8cdabfaee1c0a4274e"]

    neighbors, dist = get_embeddings(
        model, entities=articles_ids + authors_ids, n_neighbors=5
    )
    print(neighbors, dist)


if __name__ == "__main__":
    main()
