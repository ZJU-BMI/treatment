import tensorflow as tf


class ModelConfig(object):
    def __init__(self):
        self.x_dim = 91
        self.a_dim = 8
        self.y_dim = 1

        self.xh_dim = 50
        self.ah_dim = 4

        self.alpha = 1
        self.beta = 1
        self.gamma = 1

        self.batch_size = 128
        self.epochs = 1000

    @property
    def buffer_size(self):
        return self.batch_size * 10


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
        def x_ae():
            wx = tf.get_variable('wx', [self._config.x_dim, self._config.xh_dim],
                                 initializer=tf.contrib.layers.xavier_initializer())
            xh = self._x @ wx
            x_rec = xh @ tf.transpose(wx)
            return xh, x_rec

        self._xh, self._x_rec = x_ae()

        def a_ae():
            wa = tf.get_variable('wa', [self._config.a_dim, self._config.ah_dim],
                                 initializer=tf.contrib.layers.xavier_initializer())
            ah = self._a @ wa
            a_rec = ah @ tf.transpose(wa)
            return ah, a_rec

        self._ah, self._a_rec = a_ae()

        def _a_generator():
            i = tf.keras.layers.Input(shape=(self._config.xh_dim + self._config.y_dim,))
            x = tf.keras.layers.Dense(10, activation=tf.nn.relu)(i)
            o = tf.keras.layers.Dense(self._config.a_dim, activation=tf.nn.sigmoid)(x)
            g = tf.keras.models.Model(inputs=i, outputs=o)
            return g

        def _a_discriminator():
            i = tf.keras.layers.Input(shape=(self._config.a_dim,))
            x = tf.keras.layers.Dense(self._config.a_dim // 2, activation=tf.nn.relu)(i)
            o = tf.keras.layers.Dense(1)(x)
            d = tf.keras.models.Model(inputs=i, outputs=o)
            return d

        self._ag = _a_generator()
        self._ad = _a_discriminator()

        self._a_gen = self._ag(tf.concat((self._xh, self._y), -1))
        self._a_fake_logits = self._ad(self._a_gen)
        self._a_real_logits = self._ad(self._a)

    def _pred_def(self):
        self._logits = tf.layers.dense(tf.concat((self._xh, self._ah), -1), self._config.y_dim)
        self._pred = tf.nn.sigmoid(self._logits)

    def _loss_def(self):
        self._x_rec_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self._x,
                                                           logits=self._x_rec)
        self._a_rec_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self._a,
                                                           logits=self._a_rec)
        self._pred_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self._y,
                                                          logits=self._logits)
        self._gen_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self._a_fake_logits),
                                                         logits=self._a_fake_logits)
        self._dis_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self._a_real_logits),
                                                         logits=self._a_real_logits) + \
            tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(self._a_fake_logits),
                                            logits=self._a_fake_logits)

        self._total_loss = self._gen_loss \
            + self._config.alpha * self._x_rec_loss \
            + self._config.beta * self._a_rec_loss \
            + self._config.gamma * self._pred_loss

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
