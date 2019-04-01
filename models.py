import copy
import json

import tensorflow as tf
import numpy as np

from data import pair_examples


class Config(object):

    @classmethod
    def from_dict(cls, json_object):
        config = cls()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, 'r', encoding='utf-8') as rf:
            con = json.load(rf)
        return cls.from_dict(con)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'


class GDMModelConfig(Config):
    def __init__(self):
        self.x_dim = 91
        self.a_dim = 8
        self.y_dim = 1

        self.alpha = 1
        self.beta = 10
        self.l2 = 0.01

        self.act_fn = 'sigmoid'

        self.learning_rate = 0.001
        self.batch_size = 256
        self.epochs = 1000

    @property
    def xh_dim(self):
        return self.x_dim // 2

    @property
    def ah_dim(self):
        return self.a_dim // 2

    @property
    def buffer_size(self):
        return self.batch_size * 10


class GanDaeMLPModel(object):
    def __init__(self,
                 config: GDMModelConfig):
        self._config = config
        self._build()

    def _build(self):
        self._placeholder_def()
        self._set_def()
        self._main_graph()
        self._pred_def()
        self._loss_def()
        self._train_def()
        self._init_sess()

    def _placeholder_def(self):
        with tf.variable_scope('input'):
            self._input_x = tf.placeholder(tf.float32, [None, self._config.x_dim], 'input_x')
            self._input_a = tf.placeholder(tf.float32, [None, self._config.a_dim], 'input_a')
            self._input_y = tf.placeholder(tf.float32, [None, self._config.y_dim], 'input_y')

    def _set_def(self):
        with tf.variable_scope('data'):
            self._data_set = tf.data.Dataset.from_tensor_slices((self._input_x, self._input_a, self._input_y))
            self._data_set = self._data_set.repeat().shuffle(self._config.buffer_size).batch(self._config.batch_size)

            self._iter = self._data_set.make_initializable_iterator()
            self._x, self._a, self._y = self._iter.get_next()

    def _main_graph(self):
        self._regularizer = tf.keras.regularizers.l2(self._config.l2)

        if self._config.act_fn == 'relu':
            self._act_fn = tf.nn.relu
        else:
            self._act_fn = tf.nn.sigmoid

        def x_ae():
            wx = tf.get_variable('wx', [self._config.x_dim, self._config.xh_dim],
                                 initializer=tf.keras.initializers.glorot_uniform())
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self._regularizer(wx))
            bx = tf.get_variable('bx', [self._config.xh_dim],
                                 initializer=tf.keras.initializers.zeros())
            zx = tf.get_variable('zx', [self._config.x_dim],
                                 initializer=tf.keras.initializers.zeros())
            xh = self._act_fn(self._x @ wx + bx)
            x_rec = xh @ tf.transpose(wx) + zx
            return xh, x_rec

        self._xh, self._x_rec = x_ae()

        def a_ae():
            wa = tf.get_variable('wa', [self._config.a_dim, self._config.ah_dim],
                                 initializer=tf.keras.initializers.glorot_uniform())
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self._regularizer(wa))
            ba = tf.get_variable('ba', [self._config.ah_dim],
                                 initializer=tf.keras.initializers.zeros())
            za = tf.get_variable('za', [self._config.a_dim],
                                 initializer=tf.keras.initializers.zeros())
            ah = self._act_fn(self._a @ wa + ba)
            a_rec = ah @ tf.transpose(wa) + za
            return ah, a_rec

        self._ah, self._a_rec = a_ae()

        def _a_generator():
            i = tf.keras.layers.Input(shape=(self._config.xh_dim + self._config.y_dim,))
            x = tf.layers.Dense(self._config.a_dim, activation=self._act_fn,
                                kernel_regularizer=self._regularizer)(i)
            o = tf.layers.Dense(self._config.a_dim, activation=None,
                                kernel_regularizer=self._regularizer)(x)
            g = tf.keras.models.Model(inputs=i, outputs=o)
            return g

        def _a_discriminator():
            i = tf.keras.layers.Input(shape=(self._config.a_dim,))
            x = tf.layers.Dense(self._config.ah_dim, activation=self._act_fn,
                                kernel_regularizer=self._regularizer)(i)
            o = tf.layers.Dense(1)(x)
            d = tf.keras.models.Model(inputs=i, outputs=o)
            return d

        self._ag = _a_generator()
        self._ad = _a_discriminator()

        self._a_gen = self._ag(tf.concat((self._xh, self._y), -1))
        self._a_fake_logits = self._ad(self._a_gen)
        self._a_real_logits = self._ad(self._a)

    def _pred_def(self):
        self._logits = tf.layers.dense(tf.concat((self._xh, self._ah), -1), self._config.y_dim,
                                       kernel_regularizer=tf.keras.regularizers.l2())
        self._pred = tf.nn.sigmoid(self._logits)

    def _loss_def(self):
        # self._x_rec_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self._x,
        #                                                    logits=self._x_rec)
        self._a_rec_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self._a,  # a(treatment) 确保是0或1的
                                                           logits=self._a_rec)
        self._x_rec_loss = tf.losses.mean_squared_error(labels=self._x,  # x(feature) 是实数
                                                        predictions=self._x_rec)
        # self._a_rec_loss = tf.losses.mean_squared_error(labels=self._a,
        #                                                 predictions=self._a_rec)
        self._pred_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self._y,
                                                          logits=self._logits)
        self._gen_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self._a_fake_logits),
                                                         logits=self._a_fake_logits)
        self._dis_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self._a_real_logits),
                                                         logits=self._a_real_logits) + \
                         tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(self._a_fake_logits),
                                                         logits=self._a_fake_logits)

        self._reg_loss = tf.losses.get_regularization_loss()
        self._total_loss = self._pred_loss \
                           + self._config.alpha * (self._x_rec_loss + self._a_rec_loss) \
                           + self._config.beta * self._gen_loss \
                           + self._reg_loss  # 定义regularizer的时候已经定义好L2超参数了，这里不再乘l2

    def _train_def(self):
        gen_vars = [w for w in tf.trainable_variables() if w not in self._ad.trainable_variables]

        self._train_gen = tf.train.AdamOptimizer(0.0001).minimize(self._total_loss, var_list=gen_vars)
        self._train_dis = tf.train.AdamOptimizer(0.0001).minimize(self._dis_loss, var_list=self._ad.trainable_variables)

    def _init_sess(self):
        c = tf.ConfigProto()
        c.gpu_options.allow_growth = True
        self._sess = tf.Session(config=c)
        self._init = tf.global_variables_initializer()
        self._sess.run(self._init)

    def fit(self, data_set, retrain=True):
        if retrain:
            self._sess.run(self._init)

        self._sess.run(self._iter.initializer, feed_dict={self._input_x: data_set.x,
                                                          self._input_a: data_set.a,
                                                          self._input_y: data_set.y})

        epochs = []
        losses = []
        for i in range(self._config.epochs * data_set.examples // self._config.batch_size):
            self._sess.run(self._train_gen)
            self._sess.run(self._train_dis)

            if i % 10 == 0:
                epochs.append(i)
                losses.append(self._sess.run(self._pred_loss))

    def predict(self, data_set):
        return self._sess.run(self._pred, feed_dict={self._x: data_set.x,
                                                     self._a: data_set.a})

    def predict_proba(self, data_set):
        return self._sess.run(self._pred, feed_dict={self._x: data_set.x,
                                                     self._a: data_set.a})


class FewShotConfig(Config):
    def __init__(self):
        self.x_dim = 100
        self.rep_layers = 2

        self.l2 = 0.001
        self.k = 5

        self.batch_size = 256
        self.learning_rate = 0.001
        self.epochs = 1

    @property
    def buffer_size(self):
        return self.batch_size * 10

    @property
    def rep_dim(self):
        return self.x_dim // 2


class FewShotModel(object):
    def __init__(self, config: FewShotConfig):
        self._config = config
        self._build()
        self._trained = False
        self._x_set = None
        self._y_set = None

    def _build(self):
        self._placeholder_def()
        self._main_graph()
        self._loss_def()
        self._train_def()
        self._init_sess()

    def _placeholder_def(self):
        self._x1 = tf.placeholder(tf.float32, [None, self._config.x_dim], 'x1')
        self._x2 = tf.placeholder(tf.float32, [None, self._config.x_dim], 'x2')
        self._y = tf.placeholder(tf.float32, [None, 1], 'y')

    def _main_graph(self):
        self._regularizer = tf.keras.regularizers.l2(self._config.l2)

        def rep_net():
            inp = tf.keras.layers.Input((self._config.x_dim, ))
            x = inp
            for _ in range(self._config.rep_layers):
                x = tf.keras.layers.Dense(self._config.rep_dim, activation=tf.nn.sigmoid,
                                          kernel_regularizer=self._regularizer)(x)
            model = tf.keras.models.Model(inputs=inp, outputs=x)
            return model

        self._rep_net = rep_net()
        self._rep1 = tf.math.l2_normalize(self._rep_net(self._x1), -1)
        self._rep2 = tf.math.l2_normalize(self._rep_net(self._x2), -1)

        self._distance = tf.sqrt(tf.reduce_sum(tf.square(self._rep1 - self._rep2), 1))

    def _loss_def(self):
        self._cl = contrastive_loss(self._y, self._rep1, self._rep2)
        self._reg_loss = tf.reduce_sum(self._rep_net.losses)
        self._total_loss = self._cl + self._reg_loss

    def _train_def(self):
        self._train_op = tf.train.AdamOptimizer(self._config.learning_rate).minimize(self._total_loss)

    def _init_sess(self):
        c = tf.ConfigProto()
        c.gpu_options.allow_growth = True
        self._sess = tf.Session(config=c)
        self._init = tf.global_variables_initializer()
        self._sess.run(self._init)

    def fit(self, data_set):
        self._sess.run(self._init)

        self._x_set = np.concatenate((data_set.x, data_set.a), -1)
        self._y_set = np.reshape(data_set.y, -1)
        self._trained = True

        for (x1, x2), y in pair_examples(data_set, self._config.batch_size, self._config.epochs):
            self._sess.run(self._train_op, feed_dict={self._x1: x1,
                                                      self._x2: x2,
                                                      self._y: y})

    def predict(self, data_set):
        preds = np.zeros((data_set.examples, 1))

        for i in range(data_set.examples):
            feat = np.concatenate((data_set.x[i], data_set.a[i]), -1)
            pred = self.predict_one(feat)
            preds[i][0] = pred
        return preds

    def predict_one(self, x):
        x = np.tile(x, (self._x_set.shape[0], 1))

        distances = self._sess.run(self._distance, feed_dict={self._x1: self._x_set,
                                                              self._x2: x})
        distances = np.reshape(distances, -1)
        first_k = np.argsort(distances)[:self._config.k]
        first_k_y = self._y_set[first_k]
        return np.argmax(np.bincount(first_k_y))


def contrastive_loss(labels, pos_embedding, neg_embedding, margin=1.0):
    """

    :param labels: two examples are from the same class ? 1 : 0
    :param pos_embedding: 2-D vector embedding of first input
    :param neg_embedding: 2-D vector embedding of second input
    :param margin:
    :return:
    """
    distances = tf.sqrt(tf.reduce_sum(tf.square(pos_embedding - neg_embedding), 1))
    return tf.reduce_mean(
               tf.to_float(labels) * tf.square(distances) +
               (1 - tf.to_float(labels) * tf.square(tf.maximum(margin - distances, 0)))
           )
