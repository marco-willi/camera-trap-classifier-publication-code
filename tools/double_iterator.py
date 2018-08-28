"""
Double Iterator
- Outer (slower) ImageGenerator that serves large batches of data that just
  fit into memory
- Inner (numpy) ImageGenerator that serves smaller batches of data
"""

# import modules
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import Iterator


class DoubleIterator(Iterator):
    """ Outer / Inner data generators to optimize image serving
        - batch_size: int
            the number of images returned by the Iterator
        - outer_generator: Iterator that returns images
           typically ImageDataGenerator.flow_from_directory()
    """
    def __init__(self, outer_generator, batch_size, seed=None,
                 inner_shuffle=True):
        self.outer_generator = outer_generator
        self.batch_size = batch_size
        self.n_on_stack = 0
        self.inner = None
        self.n = outer_generator.n
        self.seed = seed
        self.inner_shuffle = inner_shuffle

    def next(self):
        """ Get next batch """
        if (self.n_on_stack == 0) or (self.inner is None):
            # get next batch of outer generator
            X_outer, y_outer = self.outer_generator.next()
            # calculate stack size for inner generator
            self.n_on_stack = (self.outer_generator.batch_size //
                               self.batch_size)

            # Create inner data generator (no data agumentation - this is
            # done by the outer generator)
            self.inner = ImageDataGenerator().flow(
                X_outer, y_outer,
                batch_size=self.batch_size,
                seed=self.seed, shuffle=self.inner_shuffle)

        # get next batch
        X_inner, y_inner = self.inner.next()
        self.n_on_stack -= 1
        # print("N on stack: %s, batches_seen: %s" %
        #       (self.n_on_stack, self.outer_generator.total_batches_seen))

        return X_inner, y_inner
