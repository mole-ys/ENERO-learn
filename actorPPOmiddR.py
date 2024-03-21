# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from keras import regularizers

class myModel(tf.keras.Model):
    def __init__(self, hparams, hidden_init_actor, kernel_init_actor):
        super(myModel, self).__init__()
        self.hparams = hparams

        # Define layers here
        # 定义消息传递层Message，使用Sequential模型，便于后续添加多个层。
        self.Message = tf.keras.models.Sequential()
        # 向消息传递层添加一个全连接层（Dense层），其输出维度由hparams中的link_state_dim确定，激活函数为SELU，权重初始化器为hidden_init_actor。
        self.Message.add(keras.layers.Dense(self.hparams['link_state_dim'],
                                            kernel_initializer=hidden_init_actor,
                                            activation=tf.nn.selu, name="FirstLayer"))

        # 定义状态更新层Update，使用GRUCell，输出维度同样由link_state_dim确定，指定数据类型为float32。
        self.Update = tf.keras.layers.GRUCell(self.hparams['link_state_dim'], dtype=tf.float32)

        # 定义读出层Readout，使用Sequential模型。
        self.Readout = tf.keras.models.Sequential()
        # 向读出层添加第一个全连接层，输出维度由readout_units确定，激活函数为SELU，权重初始化器为hidden_init_actor，并添加L2正则化。
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_initializer=hidden_init_actor,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout1"))
        #self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        # 向读出层再添加一个与之前类似的全连接层。
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_initializer=hidden_init_actor,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout2"))
        #self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        # 向读出层添加最后一个全连接层，用于输出最终结果，输出维度为1，权重初始化器为kernel_init_actor。
        self.Readout.add(keras.layers.Dense(1, kernel_initializer=kernel_init_actor, name="Readout3"))

    def build(self, input_shape=None):
        # Create the weights of the layer
        self.Message.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim']*2]))
        self.Update.build(input_shape=tf.TensorShape([None,self.hparams['link_state_dim']]))
        self.Readout.build(input_shape=[None, self.hparams['link_state_dim']])
        self.built = True

    #@tf.function
    # 这是模型的前向传播方法，其中link_state是当前边的状态，states_graph_ids用于读出阶段以聚合图中所有边的信息，
    # states_first和states_second分别代表边的起点和终点（用于消息传递），sates_num_edges表示图中边的总数，training标志用于指示是否是训练阶段。
    def call(self, link_state, states_graph_ids, states_first, states_second, sates_num_edges, training=False):

        # Execute T times
        for _ in range(self.hparams['T']):
            # We have the combination of the hidden states of the main edges with the neighbours
            # 通过tf.gather，基于states_first和states_second从link_state中获取主边和相邻边的隐藏状态
            mainEdges = tf.gather(link_state, states_first)
            neighEdges = tf.gather(link_state, states_second)

            # 将主边和相邻边的状态沿着第二维拼接，为消息传递准备。
            edgesConcat = tf.concat([mainEdges, neighEdges], axis=1)

            ### 1.a Message passing for link with all it's neighbours
            # 对拼接的边状态执行消息传递，即通过Message层处理。
            outputs = self.Message(edgesConcat)

            ### 1.b Sum of output values according to link id index
            # 利用tf.math.unsorted_segment_sum对Message层的输出按照边的终点（或某种索引）进行求和，为每个边生成一个更新消息。
            edges_inputs = tf.math.unsorted_segment_sum(data=outputs, segment_ids=states_second,
                                                        num_segments=sates_num_edges)

            ### 2. Update for each link
            # GRUcell needs a 3D tensor as state because there is a matmul: Wrap the link state
            # 使用GRUCell更新每条边的状态。GRUCell接受当前边的状态和通过消息传递生成的更新消息，输出新的边状态。
            outputs, links_state_list = self.Update(edges_inputs, [link_state])

            # 更新边的状态为GRUCell的输出。
            link_state = links_state_list[0]

        # Perform sum of all hidden states
        # 通过tf.math.segment_sum对所有边的状态进行聚合，基于states_graph_ids将属于同一图的边状态求和。
        edges_combi_outputs = tf.math.segment_sum(link_state, states_graph_ids, name=None)

        # 最后，将聚合后的边状态传递给Readout层以生成最终的输出。
        # 这个输出可以用于图分类、边预测等任务。training参数用于指示是否应用如Dropout等仅在训练时使用的技术。
        r = self.Readout(edges_combi_outputs,training=training)
        return r
