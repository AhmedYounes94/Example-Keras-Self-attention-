import keras
from keras import backend as K
from keras import models, layers, regularizers, optimizers, objectives, regularizers, initializers, constraints
from keras.layers import Layer

class SelfAttention(Layer):
    def __init__(self, 
                 aspect_size,
                 hidden_dim, 
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        self.aspect_size = aspect_size
        self.hidden_dim = hidden_dim
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (None, Sequence_size, Sequence_hidden_dim)
        assert len(input_shape) >= 3
        sequence_size = input_shape[1]
        sequence_hidden_dim = input_shape[-1]
        
        self.Ws1 = self.add_weight(shape=(self.hidden_dim, sequence_hidden_dim),
                                      initializer=self.kernel_initializer,
                                      name='SelfAttention-Ws1',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
        self.Ws2 = self.add_weight(shape=(self.aspect_size, self.hidden_dim), 
                                      initializer=self.kernel_initializer,
                                      name='SelfAttention-Ws2',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        batch_size = K.cast(K.shape(inputs)[0], K.floatx())
        inputs_t = K.permute_dimensions(inputs, (1,2,0)) # H.T
        d1 = K.tanh(K.permute_dimensions(K.dot(self.Ws1, inputs_t), (2,0,1))) # d1 = tanh(dot(Ws1, H.T))
        d1 = K.permute_dimensions(d1, (2,1,0))
        A = K.softmax(K.permute_dimensions(K.dot(self.Ws2, d1), (2,0,1))) # A = softmax(dot(Ws2, d1))
        H = K.permute_dimensions(inputs, (0,2,1))
        outputs = K.batch_dot(A, H, axes=2) # M = AH

        A_t = K.permute_dimensions(A, (0,2,1))
        I = K.eye(self.aspect_size)
        P = K.square(self._frobenius_norm(K.batch_dot(A, A_t) - I)) # P = (frobenius_norm(dot(A, A.T) - I))**2
        self.add_loss(P/batch_size)
        return outputs

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 3
        assert input_shape[-1]
        batch_size = input_shape[0]
        sequence_hidden_dim = input_shape[-1]
        output_shape = [batch_size, self.aspect_size, sequence_hidden_dim]
        return tuple(output_shape)

    def get_config(self):
        config = {
            'aspect_size': self.aspect_size,
            'hidden_dim': self.hidden_dim,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def _frobenius_norm(self, inputs):
        outputs = K.sqrt(K.sum(K.square(inputs)))
        return outputs
        
