import tensorflow as tf
import warnings
from abc import abstractmethod
from stable_baselines.common.distributions import make_proba_dist_type, CategoricalProbabilityDistribution, \
    MultiCategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution, BernoulliProbabilityDistribution
from stable_baselines.common.policies import nature_cnn,  mlp_extractor, BasePolicy
from stable_baselines.a2c.utils import linear, conv, conv_to_fc, deconv, linear_general
import numpy as np


class POMEPolicy(BasePolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """


    """
    Policy object that implements actor critic

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        super(POMEPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        self._pdtype = make_proba_dist_type(ac_space)
        self._proba_distribution = None
        self._value_fn = None
        self._action = None
        self._deterministic_action = None
        self._policy = None
        try:
            self.n_actions = ac_space.n
            self.n_ob = (ob_space.shape[0], ob_space.shape[1])
        except AttributeError:
            self.n_actions = ac_space.shape[0]
            self.n_ob = ob_space.shape[0]
        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        def a3c_cnn(scaled_images, **kwargs):
            """
            CNN from A3C paper.
            :param scaled_images: (TensorFlow Tensor) Image input placeholder
            :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
            :return: (TensorFlow Tensor) The CNN output layer
            """
            activ = tf.nn.relu
            layer_1 = activ(
                conv(scaled_images, 'c1', n_filters=16, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
            layer_2 = activ(conv(layer_1, 'c2', n_filters=32, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
            layer_3 = conv_to_fc(layer_2)
            return activ(linear(layer_3, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))

        def dynamics(scaled_images, action, **kwargs):
            """
            Dynamic function
            :param scaled_images: (TensorFlow Tensor) Image input placeholder
            :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
            :return: (TensorFlow Tensor) The CNN output layer
            """
            action_one_hot = tf.one_hot(action, self.n_actions)
            single_action_map = tf.reshape(action_one_hot, [-1, 1, 1, self.n_actions])
            action_map = tf.tile(single_action_map, (1, self.n_ob[0], self.n_ob[1], 1))
            merge_map = tf.concat([scaled_images, action_map], axis=-1)

            activ = tf.nn.relu
            layer_1 = activ(conv(merge_map, 'c3', n_filters=32, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
            output_height, output_width = layer_1.shape[1], layer_1.shape[2]
            layer_2 = activ(conv(layer_1, 'c4', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
            layer_de_2 = activ(deconv(layer_2, 'c5', n_filters=64, filter_size=4, stride=2, output_height=output_height, output_width=output_width, init_scale=np.sqrt(2), **kwargs))
            layer_de_1 = activ(deconv(layer_de_2, 'c6', n_filters=256, filter_size=4, stride=2, output_height=self.n_ob[0], output_width=self.n_ob[1], init_scale=np.sqrt(2), **kwargs))
            layer_de_0 = tf.nn.softmax(layer_de_1, axis=-1)

            next_frame = tf.reshape(layer_de_0, [-1, 256])
            next_frame = tf.matmul(next_frame, tf.constant(np.arange(0, 256).reshape(256, 1).astype(np.float32)/255))
            next_frame = tf.reshape(next_frame, [-1, self.n_ob[0], self.n_ob[1]])
            rf_latent = conv_to_fc(layer_2)
            return next_frame, linear(rf_latent, 'rf', 1)

        with tf.variable_scope("model", reuse=reuse):
            pi_latent = vf_latent = a3c_cnn(self.processed_obs, **kwargs)

            self._value_fn = linear(vf_latent, 'vf', 1)
            self._next_state_fn = linear(vf_latent, 'tf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

        with tf.variable_scope("world_model", reuse=reuse):
            self._next_frame, self._reward_fn = dynamics(self.processed_obs, self.action, **kwargs)
            self._reward_flat = self.reward_fn[:, 0]

        self._pdtype = make_proba_dist_type(ac_space)

    def _setup_init(self):
        """Sets up the distributions, actions, and value."""
        with tf.variable_scope("output", reuse=True):
            assert self.policy is not None and self.proba_distribution is not None and self.value_fn is not None
            self._action = self.proba_distribution.sample()
            self._deterministic_action = self.proba_distribution.mode()
            self._neglogp = self.proba_distribution.neglogp(self.action)
            if isinstance(self.proba_distribution, CategoricalProbabilityDistribution):
                self._policy_proba = tf.nn.softmax(self.policy)
            elif isinstance(self.proba_distribution, DiagGaussianProbabilityDistribution):
                self._policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]
            elif isinstance(self.proba_distribution, BernoulliProbabilityDistribution):
                self._policy_proba = tf.nn.sigmoid(self.policy)
            elif isinstance(self.proba_distribution, MultiCategoricalProbabilityDistribution):
                self._policy_proba = [tf.nn.softmax(categorical.flatparam())
                                     for categorical in self.proba_distribution.categoricals]
            else:
                self._policy_proba = []  # it will return nothing, as it is not implemented
            self._value_flat = self.value_fn[:, 0]

    @property
    def pdtype(self):
        """ProbabilityDistributionType: type of the distribution for stochastic actions."""
        return self._pdtype

    @property
    def policy(self):
        """tf.Tensor: policy output, e.g. logits."""
        return self._policy

    @property
    def proba_distribution(self):
        """ProbabilityDistribution: distribution of stochastic actions."""
        return self._proba_distribution

    @property
    def value_fn(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, 1)"""
        return self._value_fn

    @property
    def value_flat(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, )"""
        return self._value_flat

    @property
    def reward_fn(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, 1)"""
        return self._reward_fn

    @property
    def reward_flat(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, )"""
        return self._reward_flat

    @property
    def action(self):
        """tf.Tensor: stochastic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._action

    @property
    def next_frame(self):
        """tf.Tensor: stochastic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._next_frame

    @property
    def deterministic_action(self):
        """tf.Tensor: deterministic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._deterministic_action

    @property
    def neglogp(self):
        """tf.Tensor: negative log likelihood of the action sampled by self.action."""
        return self._neglogp

    @property
    def policy_proba(self):
        """tf.Tensor: parameters of the probability distribution. Depends on pdtype."""
        return self._policy_proba

    def step(self, obs, state=None, mask=None, deterministic=False, need_model=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                {self.obs_ph: obs})
        next_frame, reward = self.sess.run([self.next_frame, self.reward_flat], {self.obs_ph: obs, self.action: action})
        # print(action)
        # print(self.processed_obs)
        # print(next_frame)
        if need_model:
            return action, value, next_frame, reward, self.initial_state, neglogp
        else:
            return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    def value_simple(self, obs):
        return self.sess.run(self.value_flat, {self.processed_obs: obs})

    def scale_obs(self, obs):
        return self.sess.run(self.processed_obs, {self.obs_ph: obs})