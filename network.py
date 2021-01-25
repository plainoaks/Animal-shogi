from tensorflow.python import keras as K

class SLPolicyNetwork:
    
    def __init__(self):
        self.model = K.Sequential()
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True, input_shape = (22, 4, 3)))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.BatchNormalization())
        self.model.add(K.layers.Flatten())
        self.model.add(K.layers.Dense(units = 144, activation = "softmax"))
        self.model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=[K.metrics.Accuracy()])

    def __call__(self, x, y):
        return self.model.fit(x, y, batch_size=32)

class ValueNetwork:

    def __init__(self):
        self.model = K.Sequential()
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True, input_shape = (22, 4, 3)))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu", use_bias = True))
        self.model.add(K.layers.BatchNormalization())
        self.model.add(K.layers.Flatten())
        self.model.add(K.layers.Dense(units = 1, activation = "sigmoid"))
        self.model.compile(optimizer="Adam", loss="mean_squared_error", metrics=[K.metrics.Accuracy()])

    def __call__(self, x, y):
        return self.model.fit(x, y, batch_size=32)

class RolloutPolicy:
    def __init__(self):
        super().__init__()
        self.model = K.Sequential()
        self.model.add(K.layers.Conv2D(filters = 16, kernel_size = 3, padding = "same", activation = "relu", use_bias = True, input_shape = (22, 4, 3)))
        self.model.add(K.layers.Flatten())
        self.model.add(K.layers.Dense(units = 144, activation = "softmax"))
        self.model.compile(optimizer="Adam", loss="categorical_crossentropy")
    
    def __call__(self, x, y):
        return self.model.fit(x, y, batch_size=32)
