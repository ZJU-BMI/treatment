import copy
import json

import tensorflow as tf


class ModelConfig(object):
    def __init__(self):
        self.x_dim = 91
        self.a_dim = 8
        self.y_dim = 1

        self.alpha = 1
        self.beta = 1
        self.gamma = 1
        self.l2 = 0.001

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


class BaseModel(object):
    def __init__(self,
                 config: ModelConfig):
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

        def x_ae():
            wx = tf.get_variable('wx', [self._config.x_dim, self._config.xh_dim],
                                 initializer=tf.keras.initializers.glorot_uniform())
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self._regularizer(wx))
            bx = tf.get_variable('bx', [self._config.xh_dim],
                                 initializer=tf.keras.initializers.zeros())
            zx = tf.get_variable('zx', [self._config.x_dim],
                                 initializer=tf.keras.initializers.zeros())
            xh = self._x @ wx + bx
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
            ah = self._a @ wa + ba
            a_rec = ah @ tf.transpose(wa) + za
            return ah, a_rec

        self._ah, self._a_rec = a_ae()

        def _a_generator():
            i = tf.keras.layers.Input(shape=(self._config.xh_dim + self._config.y_dim,))
            x = tf.layers.Dense(10, activation=tf.nn.relu,
                                kernel_regularizer=self._regularizer)(i)
            o = tf.layers.Dense(self._config.a_dim, activation=tf.nn.sigmoid,
                                kernel_regularizer=self._regularizer)(x)
            g = tf.keras.models.Model(inputs=i, outputs=o)
            return g

        def _a_discriminator():
            i = tf.keras.layers.Input(shape=(self._config.a_dim,))
            x = tf.layers.Dense(self._config.ah_dim, activation=tf.nn.relu,
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

        self._total_loss = self._config.alpha * self._x_rec_loss \
            + self._config.beta * self._a_rec_loss \
            + self._config.gamma * self._pred_loss \
            + self._reg_loss  # 定义regularizer的时候已经定义好L2超参数了，这里不再乘l2

    def _train_def(self):
        gen_vars = [w for w in tf.trainable_variables() if w not in self._ad.trainable_variables]

        self._train_gen = tf.train.AdamOptimizer().minimize(self._total_loss, var_list=gen_vars)
        self._train_dis = tf.train.AdamOptimizer().minimize(self._dis_loss, var_list=self._ad.trainable_variables)

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

        for i in range(self._config.epochs * data_set.examples // self._config.batch_size):
            self._sess.run(self._train_gen)
            self._sess.run(self._train_dis)

    def predict(self, data_set):
        return self._sess.run(self._pred, feed_dict={self._x: data_set.x,
                                                     self._a: data_set.a})
