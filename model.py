import tensorflow as tf

def add_conv_stage(dim_out, kernel_size=3, strides=1, padding='same', use_bias=False, use_BN=False):
    if  use_BN:
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=dim_out, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(filters=dim_out, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1)
        ])
    else:
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=dim_out, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=dim_out, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias),
            tf.keras.layers.ReLU()
        ])
    
def add_merge_stage(ch_coarse, ch_fine, in_coarse, in_fine, upsample):
    conv = tf.keras.layers.Conv2DTranspose(filters=ch_fine, kernel_size=4, strides=2, padding='same', output_padding=1)
    
def upsample(ch_fine):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2DTranspose(filters=ch_fine, kernel_size=4, strides=2, padding='same', use_bias=False),
        tf.keras.layers.ReLU()
    ])

class UNet(tf.keras.models.Model):
    def __init__(self, use_BN):
        super().__init__()
        
        self.conv1 = add_conv_stage(32, use_BN=use_BN)
        self.conv2 = add_conv_stage(64, use_BN=use_BN)
        self.conv3 = add_conv_stage(128, use_BN=use_BN)
        self.conv4 = add_conv_stage(256, use_BN=use_BN)
        self.conv5 = add_conv_stage(512, use_BN=use_BN)

        self.aux = tf.keras.layers.Conv2D(filters=1, kernel_size=14,
                        strides=1, kernel_initializer="he_normal", activation='sigmoid')
        self.aux_flatten = tf.keras.layers.Flatten(name='aux_output')

        self.conv4m = add_conv_stage(256, use_BN=use_BN)
        self.conv3m = add_conv_stage(128, use_BN=use_BN)
        self.conv2m = add_conv_stage(64, use_BN=use_BN)
        self.conv1m = add_conv_stage(32, use_BN=use_BN)
        
        self.conv0 = tf.keras.layers.Conv2D(1, 3, 1, padding='same')
        
        self.max_pool = tf.keras.layers.MaxPool2D()
        
        self.upsample54 = upsample(256)
        self.upsample43 = upsample(128)
        self.upsample32 = upsample(64)
        self.upsample21 = upsample(32)
            
        self.dropout = tf.keras.layers.Dropout(rate=0.1)
        
    def call(self, x, **kwargs):
        conv1_out = self.conv1(x)
        #return self.upsample21(conv1_out)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv2_out = self.dropout(conv2_out)
        conv3_out = self.conv3(self.max_pool(conv2_out))
        conv3_out = self.dropout(conv3_out)
        conv4_out = self.conv4(self.max_pool(conv3_out))
        conv4_out = self.dropout(conv4_out)
        conv5_out = self.conv5(self.max_pool(conv4_out))
        conv5_out = self.dropout(conv5_out)
        
        aux = self.aux(conv5_out)
        aux = self.aux_flatten(aux)

        conv5m_out = tf.concat([self.upsample54(conv5_out), conv4_out], -1)
        conv4m_out = self.conv4m(conv5m_out)
        conv4m_out = self.dropout(conv4m_out)
        
        conv4m_out_ = tf.concat([self.upsample43(conv4m_out), conv3_out], -1)
        conv3m_out = self.conv3m(conv4m_out_)
        conv3m_out = self.dropout(conv3m_out)
        
        conv3m_out_ = tf.concat([self.upsample32(conv3m_out), conv2_out], -1)
        conv2m_out = self.conv2m(conv3m_out_)
        conv2m_out = self.dropout(conv2m_out)

        conv2m_out_ = tf.concat([self.upsample21(conv2m_out), conv1_out], -1)
        conv1m_out = self.conv1m(conv2m_out_)
        conv1m_out = self.dropout(conv1m_out)

        conv0_out = self.conv0(conv1m_out)

        return tf.keras.activations.sigmoid(conv0_out), aux
