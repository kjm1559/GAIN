import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf

def make_lossing_data(batch_data, loss_rate=0.2):
    lossing_data = np.zeros(batch_data.shape)
    for i in range(batch_data.shape[0]):
        mark = np.zeros(len(batch_data[i].flatten()))
        mark[:int(len(mark) * loss_rate)] = 1
        np.random.shuffle(mark)
        mark = mark.reshape(batch_data[i].shape)
        lossing_data[i, mark == 1] = -1
        lossing_data[i, mark != 1] = batch_data[i, mark != 1]
    return lossing_data 

def make_lossing_box_data(batch_data, box_size=4):
    lossing_data = batch_data.copy()
    for i in range(batch_data.shape[0]):
        # st = int(lossing_data.shape[1] / 2 - box_size/2) + np.random.randint(-3, 3)
        # st2 = int(lossing_data.shape[2] / 2 - box_size/2) + np.random.randint(-3, 3)
        st = np.random.randint(0, batch_data.shape[1] - box_size)
        st2 = np.random.randint(0, batch_data.shape[2] - box_size)
        lossing_data[i, st:st+box_size, st2:st2+box_size] = -1
    return lossing_data 


def make_input(batch_data, hint_rate=0.3):
    # make mask matrix
    mask_matrix = np.ones(batch_data.shape)
    mask_matrix[batch_data == -1] = 0

    # modifiy data matrix
    batch_data[batch_data == -1] = 0

    # make random matrix
    random_matrix = batch_data.copy()
    # random_matrix[np.where(mask_matrix == 0)] = np.random.normal(0, 1, len(np.where(mask_matrix == 0)[0].flatten()))
    random_matrix = (1 - mask_matrix) * np.random.normal(0, 0.1, mask_matrix.shape)

    hint_matrix = []
    for i in range(len(mask_matrix)):
        m = mask_matrix[i]
        b_index = np.arange(len(m.flatten()))
        b_rand = np.random.randint(0, len(b_index), len(b_index))
        b = np.zeros(len(b_index))
        b[b_rand != b_index] = 1
        b = b.reshape(m.shape)
        hint_matrix.append(m * b + 0.5 * (1 - b))
    
    return batch_data, random_matrix, mask_matrix, np.array(hint_matrix)


def tf_make_data_matrix(batch_data, tf_mask_matrix, box_size=12):
    if type(tf_mask_matrix.shape[0]) == type(None):
        return tf_mask_matrix, tf_mask_matrix
    # make random matrix
    tf_random_noise = tf.random.uniform(batch_data.shape, maxval=0.1, dtype=tf.float32)
    tf_random_matrix = tf_random_noise * (1 - tf_mask_matrix)

    # make hint_matrix
    tf_hint_matrix = tf_mask_matrix
    b = tf.random.shuffle(1 - tf_mask_matrix)
    tf_hint_matrix = b * tf_mask_matrix + 0.5*(1 - b)

    return tf_random_matrix, tf_hint_matrix


def make_base_model(input_dim, output_dim, sequence_len, hidden_size, name='base_model'):
    inputs = keras.Input(shape=(sequence_len, input_dim))

    if type(hidden_size) == type([]):
        for i, hidden in enumerate(hidden_size):
            if i == 0:
                x = keras.layers.LSTM(hidden, return_sequences=True)(inputs)
            else:
                x = keras.layers.LSTM(hidden, return_sequences=True)(x)
    else:
        x = keras.layers.LSTM(hidden_size, return_sequences=True)(inputs)
    
    outputs = keras.layers.TimeDistributed(keras.layers.Dense(output_dim, activation='sigmoid'))(x)

    return keras.Model(inputs, outputs, name=name)

def make_generator(input_dim, output_dim, sequence_len, hidden_size, name='gen'):
    data_matrix = keras.Input(shape=(sequence_len, input_dim), name='data_matrix')
    random_matrix = keras.Input(shape=(sequence_len, input_dim), name='random_matrix')
    mask_matrix = keras.Input(shape=(sequence_len, input_dim), name='mask_matrix')

    # x = keras.layers.Concatenate(axis=-1)([data_matrix, random_matrix, mask_matrix])
    x = mask_matrix * data_matrix + (1 - mask_matrix) * random_matrix
    x = keras.layers.Concatenate(axis=-1)([x, mask_matrix])

    if type(hidden_size) == type([]):
        for hidden in hidden_size:
            x = keras.layers.LSTM(hidden, return_sequences=True)(x)
    else:
        x = keras.layers.LSTM(hidden_size, return_sequences=True)(x)
    
    outputs = keras.layers.TimeDistributed(keras.layers.Dense(output_dim, activation='sigmoid'))(x)

    return keras.Model([data_matrix, random_matrix, mask_matrix], outputs, name=name)

def make_discriminator(input_dim, output_dim, sequence_len, hidden_size, name='gen'):
    gen_matrix = keras.Input(shape=(sequence_len, input_dim), name='gen_matrix')
    hint_matrix = keras.Input(shape=(sequence_len, input_dim), name='hint_matrix')

    x = keras.layers.Concatenate(axis=-1)([gen_matrix, hint_matrix])
    # x = gen_matrix

    if type(hidden_size) == type([]):
        for hidden in hidden_size:
            x = keras.layers.LSTM(hidden, return_sequences=True)(x)
    else:
        x = keras.layers.LSTM(hidden_size, return_sequences=True)(x)
    
    outputs = keras.layers.TimeDistributed(keras.layers.Dense(output_dim, activation='sigmoid'))(x)
    # outputs = (outputs + hint_matrix) / 2
    return keras.Model([gen_matrix, hint_matrix], outputs, name=name)


# GAIN_tf
# tensorflow version 2.0 
class Gain(keras.Model):
    def __init__(self, generator, discriminator, train_data, random_std=1, batch_size=64, validation_split=0.2):
        super(Gain, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.random_std = random_std
        self.train_data = train_data
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.train_steps = int(int(len(self.train_data) * (1-self.validation_split)) / self.batch_size)
        self.validation_steps = int(int(len(self.train_data) * self.validation_split) / self.batch_size)

    def train_data_generator(self,):
        i = 0
        max_index = int(len(self.train_data) * (1-self.validation_split))
        while True:
            if i == max_index - self.batch_size:
                i = 0
            y = self.train_data[i:i+self.batch_size]
            x = make_lossing_box_data(y)
            data_matrix, random_matrix, mask_matrix, hint_matrix = make_input(x)
            i += self.batch_size
            yield [data_matrix, random_matrix, mask_matrix, hint_matrix], y

    def validation_data_generator(self,):
        i = int(len(self.train_data) * (1 - self.validation_split))
        max_index = len(self.train_data)
        while True:
            if i == max_index - self.batch_size:
                i = int(len(self.train_data) * self.validation_split)
            y = self.train_data[i:i+self.batch_size]
            x = make_lossing_box_data(y)
            data_matrix, random_matrix, mask_matrix, hint_matrix = make_input(x)
            i += self.batch_size
            yield [data_matrix, random_matrix, mask_matrix, hint_matrix], y
            
    # def compile(self, gen_optimizer, dis_optimizer, **args):#optimizer=None, loss=None, metrics=None, weighted_metrics=None, loss_weights=None):
    def compile(self, optimizer=None, **args):#optimizer=None, loss=None, metrics=None, weighted_metrics=None, loss_weights=None):
        super(Gain, self).compile()
        self.gen_optimizer = optimizer[0]
        self.dis_optimizer = optimizer[1]
              
    def train_step(self, batch_data):
        # x, y = batch_data
        # data_matrix, random_matrix, mask_matrix, hint_matrix = x
        x, y = batch_data
        data_matrix, mask_matrix = x
        # x = make_lossing_box_data(batch_data)

        random_matrix, hint_matrix = tf_make_data_matrix(data_matrix, mask_matrix)
        
        with tf.GradientTape() as tape:
            x_gen = self.generator([data_matrix, random_matrix, mask_matrix], training=True)
            x_hat = mask_matrix * data_matrix + (1 - mask_matrix) * x_gen
            g_dis = self.discriminator([x_hat, hint_matrix], training=True)
                
            loss_g = 100*K.mean(mask_matrix * K.square(y - x_gen))/K.mean(mask_matrix) - K.mean((1-mask_matrix)*K.log(g_dis + 1e-4))
            
        grads_G = tape.gradient(loss_g, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grads_G, self.generator.trainable_variables))
        
        with tf.GradientTape() as tape:
            x_gen = self.generator([data_matrix, random_matrix, mask_matrix], training=True)
            x_hat = mask_matrix * data_matrix + (1 - mask_matrix) * x_gen
            g_dis = self.discriminator([x_hat, hint_matrix], training=True)

            loss_d = keras.losses.BinaryCrossentropy()(mask_matrix, g_dis)#- K.mean(mask_matrix * K.log(K.clip(g_dis, 1e-4, 1)) + (1-mask_matrix) * K.log(K.clip(1-g_dis, 1e-4, 1)))
        grads_D = tape.gradient(loss_d, self.discriminator.trainable_variables)    
        self.dis_optimizer.apply_gradients(zip(grads_D, self.discriminator.trainable_variables))

        return {
            'gen_loss': loss_g,
            'dis_loss': loss_d,
            'gen_lr': self.gen_optimizer.lr,
            'dis_lr': self.dis_optimizer.lr
        } 

    def test_step(self, batch_data):
        # batch_data, y = batch_data
        # data_matrix, random_matrix, mask_matrix, hint_matrix = batch_data#make_input(batch_data)
        x, y = batch_data
        data_matrix, mask_matrix = x
        # x = make_lossing_box_data(batch_data)
        random_matrix, hint_matrix = tf_make_data_matrix(data_matrix, mask_matrix)

        x_gen = self.generator([data_matrix, random_matrix, mask_matrix], training=False)
        x_hat = mask_matrix * data_matrix + (1 - mask_matrix) * x_gen
        g_dis = self.discriminator([x_hat, hint_matrix], training=False)

        # loss_d = keras.losses.BinaryCrossentropy()(g_dis*mask_matrix, mask_matrix)        
        # loss_g = keras.losses.MeanSquaredError()(x_gen * mask_matrix, y * mask_matrix) + loss_d

        loss_g = 100*K.mean(mask_matrix * K.square(y - x_gen))/K.mean(mask_matrix) - K.mean((1-mask_matrix)*K.log(g_dis + 1e-4))
        loss_d = keras.losses.BinaryCrossentropy()(mask_matrix, g_dis)#- K.mean(mask_matrix * K.log(K.clip(g_dis, 1e-4, 1)) + (1-mask_matrix) * K.log(K.clip(1-g_dis, 1e-4, 1)))

        return {
            'gen_loss': loss_g,
            'dis_loss': loss_d
        }  


# tensorflow version 1.14
class Gain_tf_114(Gain):
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, optimizer, **args):
        self.gen_optimizer = optimizer[0]
        self.dis_optimizer = optimizer[1]
    
    def fit(self, train_data, batch_size=64, epochs=100, verbose=1, validation_split=.2, shuffle=True):
        from tqdm import tqdm
        shuffle_index = np.arange(len(train_data))
        np.random.shuffle(shuffle_index)
        train_x = train_data[shuffle_index[:int(len(train_data) * validation_split)]]
        vail_x = train_data[shuffle_index[int(len(train_data) * validation_split):]]
        hist = {
            'gen_loss':[],
            'dis_loss':[],
            'val_gen_loss':[],
            'val_dis_loss':[]
            }
        for i in range(epochs):
            # batch size train
            print('Epoch ' + str(i+1) + '/' + str(epochs))
            gen_losses = []
            dis_losses = []
            pbar = tqdm(range(0, len(train_x), batch_size), position=0)
            for j in pbar:
                target_x = train_x[j: j+batch_size]
                loss = self.train_step(tf.convert_to_tensor(target_x, dtype='float32'))
      
                gen_losses.append(K.eval(loss['gen_loss']))
                dis_losses.append(K.eval(loss['dis_loss']))
                
                pbar.set_postfix({
                    'gen_loss':np.mean(gen_losses), 
                    'dis_loss':np.mean(dis_losses)
                    })
            val_loss = self.test_step(tf.convert_to_tensor(vail_x, dtype='float32'))
            print('val_gen_loss :', K.eval(val_loss['gen_loss']), ', val_dis_loss :', K.eval(val_loss['dis_loss']))
            hist['gen_loss'].append(np.mean(gen_losses))
            hist['dis_loss'].append(np.mean(dis_losses))
            hist['val_gen_loss'].append(K.eval(val_loss['gen_loss']))
            hist['val_gen_loss'].append(K.eval(val_loss['dis_loss']))

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()    
    x_train = x_train/255
    x_test = x_test/255
    x_train_lossing = make_lossing_data(x_train, 0.3)    
    x_train_data_matrix, x_train_random_matrix, x_train_mask_matrix, x_train_hint_matrix = make_input(x_train_lossing)
    
    gen = make_generator(x_train.shape[-1], x_train.shape[-1], x_train.shape[-2], [128, 256], 'gen')
    dis = make_discriminator(x_train.shape[-1], x_train.shape[-1], x_train.shape[-2], 128, 'dis')
    x_train = x_train/x_train.max()
    gain_model = Gain(gen, dis)

    gain_model.compile(optimizer=[keras.optimizers.Adam(lr=1e-4, beta_1=0.5), keras.optimizers.Adam(lr=1e-4, beta_1=0.5)])#, metrics=[get_lr_metric])
    gain_model.fit([x_train_data_matrix, x_train_random_matrix, x_train_mask_matrix, x_train_hint_matrix], x_train, batch_size=64, epochs=100, verbose=1, validation_split=0.2)