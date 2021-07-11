import openTSNE
import pylab
import sklearn.manifold


def visualize_tsne(x, y, visualize=True, opentsne=False):
    """
        Parameters
        ----------
        x : numpy.ndarray
            feature data
        y : int
            domain labels

        Returns
        -------
        T-SNE result : numpy.ndarray
                       data_num * 2
    """
    x = x.reshape(x.shape[0], -1)
    if opentsne:
        embedding = openTSNE.TSNE().fit(x)
    else:
        embedding = sklearn.manifold.TSNE(n_components=2, random_state=1).fit_transform(x)

    if visualize:
        pylab.scatter(embedding[:, 0], embedding[:, 1], 20, y)
        pylab.show()

    return embedding
