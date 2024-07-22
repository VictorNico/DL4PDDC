import tensorflow as tf

class CustomSeparableConv2D(tf.keras.layers.SeparableConv2D):
    def __init__(self, *args, **kwargs):
        if 'kernel_initializer' in kwargs:
            kwargs['depthwise_initializer'] = kwargs.pop('kernel_initializer')
            kwargs['pointwise_initializer'] = kwargs['depthwise_initializer']
        if 'kernel_regularizer' in kwargs:
            kwargs['depthwise_regularizer'] = kwargs.pop('kernel_regularizer')
            kwargs['pointwise_regularizer'] = kwargs['depthwise_regularizer']
        if 'kernel_constraint' in kwargs:
            kwargs['depthwise_constraint'] = kwargs.pop('kernel_constraint')
            kwargs['pointwise_constraint'] = kwargs['depthwise_constraint']
        super(CustomSeparableConv2D, self).__init__(*args, **kwargs)

    def get_config(self):
        config = super(CustomSeparableConv2D, self).get_config()
        if 'depthwise_initializer' in config:
            config['kernel_initializer'] = config.pop('depthwise_initializer')
            config['kernel_initializer'] = config['kernel_initializer']
        if 'depthwise_regularizer' in config:
            config['kernel_regularizer'] = config.pop('depthwise_regularizer')
            config['kernel_regularizer'] = config['kernel_regularizer']
        if 'depthwise_constraint' in config:
            config['kernel_constraint'] = config.pop('depthwise_constraint')
            config['kernel_constraint'] = config['kernel_constraint']
        return config

# Chargez votre modèle Keras
model = tf.keras.models.load_model('my_leaf_classification_model_Xception.h5', custom_objects={'SeparableConv2D': CustomSeparableConv2D})

# Sauvegardez le modèle modifié en tant que SavedModel
model.save('saved_model/my_leaf_classification_model_Xception')
