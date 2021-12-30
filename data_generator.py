from keras.preprocessing.image import Iterator, ImageDataGenerator
import numpy as np

class CustomDataGenerator(Iterator):
    def __init__(self, X, y, target_size, preprocessing_fn, batch_size=32, shuffle=True, seed=None, **kwargs):
        self.X = X
        self.Y = y
        self.target_size = target_size
        self.preprocessing_fn = preprocessing_fn
        super().__init__(X.shape[0], batch_size, shuffle, seed)
        self.generator = ImageDataGenerator(**kwargs)
        
    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples from array of indices"""

        # Get a batch of image data
        
        copy_x = self.X[index_array].copy()
        copy_y = self.Y[index_array].copy()
        batch_x = np.zeros((len(index_array), *self.target_size, 1))
        batch_y = np.zeros((len(index_array), *self.target_size, 1))
        # Transform the inputs and correct the outputs accordingly
        for i, (x, y) in enumerate(zip(copy_x, copy_y)):
            transform_params = self.generator.get_random_transform(x.shape)
            t_x = self.generator.apply_transform(x, transform_params)
            t_y = self.generator.apply_transform(y, transform_params)
            batch_x[i], batch_y[i] = self.preprocessing_fn(t_x, denoise=True), self.preprocessing_fn(t_y)
        return batch_x, batch_y