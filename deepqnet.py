# Cecilia Aponte
# AI - Reinforcement Learning
# Breakout

import os
import numpy as np
import tensorflow as tf



class DeepQNetwork(object):
    # two networks, one to select the action and one for value of action
    # inputDims = size image (height x width), number of frames
    def __init__(self, learnRate, gamma, epsilon, nActions, netName, fullConLDim = 512,
                inputDims = (210, 160, 4), ckptDir = '/tmp/deepqnet'):
        self.learnRate = learnRate
        self.gamma = gamma
        self.epsilon = epsilon
        self.netName = netName
        self.nActions = nActions
        self.fullConLDim = fullConLDim
        self.inputDims = inputDims
        self.sess = tf.Session()
        self.buildNet()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()
        self.checkpoint_file = os.path.join(ckptDir, 'deepqnet.ckpt')
        # keep track of all trainable variables from the corresponding network
        # used to copy one network to the other
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope = self.netName)


    def buildNet(self):
        with tf.variable_scope(self.netName):
            # stack of images
            self.input = tf.placeholder(tf.float32, shape=[None, *self.inputDims],
                                        name='inputs')
            # actions that agent took
            self.actions = tf.placeholder(tf.float32, shape=[None, self.nActions],
                                        name='actionTaken')
            # target values for Q network
            self.Qtarget = tf.placeholder(tf.float32, shape=[None, self.nActions],
                                        name='qTargets')

            # build CNN layers
            # DeepMind experts used kernel_initializer of variance_scaling_initializer
            conv1 = tf.contrib.layers.conv2d(self.input, 32, 8, 4, activation_fn=tf.nn.relu)
            conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
            conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)

            # flatten output layer and pass through dense network to get q-values
            # (state,action) pair
            flatten = tf.contrib.layers.flatten(conv3)
            FullConLay = tf.contrib.layers.fully_connected(flatten, self.fullConLDim)
            self.Q = tf.contrib.layers.fully_connected(FullConLay, 4, activation_fn=None)


            # Calcualte the loss
            self.losses = tf.math.squared_difference(self.Qtarget, self.Q)
            self.loss = tf.reduce_mean(self.losses)
            # Optimizer Parameters from original paper
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learnRate, epsilon=self.epsilon)
            self.trainOper = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())


# Contains the learning, memory, etc
class Agent(object):
    # replaceTarget: how often to replace target network
    def __init__(self, alpha, gamma, MemSize, nActions, epsilon, batchSize,
                replaceTarget=5000, inputDims=(210,160,4),
                qNextDir='/tmp/qNext', qEvalDir='/tmp/qEval'):

        self.nActions = nActions
        self.gamma = gamma
        self.MemSize = MemSize
        self.memCntr = 0
        self.epsilon = epsilon
        self.batchSize = batchSize
        self.replaceTarget = replaceTarget
        self.action_space = [i for i in range(self.nActions)]

        # Two Networks:
        self.qNext = DeepQNetwork(alpha, gamma, epsilon, nActions,inputDims=inputDims,
                                netName='qNext', ckptDir=qNextDir)

        self.qEval = DeepQNetwork(alpha,gamma, epsilon, nActions,inputDims=inputDims,
                                netName='qEval', ckptDir=qEvalDir)

        # Memory save: state, action, reward, and next state transition
        # Terminal flags to tell the agent if the game is done to calculate
        # the reward at this time
        # One hot encoding for these
        self.stateMemory = np.zeros((self.MemSize, *inputDims))
        self.newStateMemory = np.zeros((self.MemSize, *inputDims))
        self.actionMemory = np.zeros((self.MemSize, self.nActions), dtype=np.int8)
        self.rewardMemory = np.zeros(self.MemSize)
        self.terminalMemory = np.zeros(self.MemSize, dtype=np.int8)


    def storeTransition(self, state, action, reward, newState, terminalState):
        # counter to keep track of number of memories stored
        index = self.memCntr % self.MemSize
        self.stateMemory[index] = state
        actions = np.zeros(self.nActions)
        actions[action] = 1.0 # one hot encoding
        self.actionMemory[index] = actions
        self.rewardMemory[index] = reward
        self.newStateMemory[index] = newState
        self.terminalMemory[index] = terminalState

        self.memCntr += 1


    # Epsilon greedy parameter: how often to choose a random action
    # Initially will be purely random (high epsilon), and then will
    # decrease so agent will become more greedy to choose the highest
    # value of the next state
    def chooseAction(self, state):
        state = state.reshape(1, *state.shape)
        random = np.random.random()
        if random < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            # find next highest valued action
            actions = self.qEval.sess.run(self.qEval.Q, feed_dict= {self.qEval.input : state})
            action = np.argmax(actions)
        return action


    # Learning
    def learning(self):
        # 1- Check to see if want to update target network
        if self.memCntr % self.replaceTarget == 0:
            self.updateGraph()
        maxMem = self.memCntr if self.memCntr < self.MemSize else self.MemSize

        # 2- Select a batch of random memory (non-sequential) to get
        #    different transitions throughout all the memory
        batch = np.random.choice(maxMem, self.batchSize)
        stateBatch = self.stateMemory[batch]
        actionBatch = self.actionMemory[batch]

        # return to integer encoding froom One Hot Encoding
        actionValues = np.array([0,1,2,3], dtype=np.int8)
        actionIndices = np.dot(actionBatch, actionValues)
        rewardBatch = self.rewardMemory[batch]
        newStateBatch = self.newStateMemory[batch]
        terminalBatch = self.terminalMemory[batch]

        # 3- Calculate current action and next maximum action
        qEval = self.qEval.sess.run(self.qEval.Q,
                            feed_dict={self.qEval.input : stateBatch})
        qNext = self.qNext.sess.run(self.qNext.Q,
                            feed_dict={self.qNext.input : newStateBatch})

        # 4- Insert into Q-learning algorithm
        Qtarget = qEval.copy()
        id = np.arange(self.batchSize)
        # If end of run, only want reward. Otherwise, the discount future reward
        Qtarget[id, actionIndices] = rewardBatch + self.gamma * np.max(qNext, axis=1) * terminalBatch


        # 5- Run update function on the loss

        # Run all through NN
        _ = self.qEval.sess.run(self.qEval.trainOper,
                            feed_dict={self.qEval.input: stateBatch,
                            self.qEval.actions: actionBatch,
                            self.qEval.Qtarget: Qtarget})

        # Decrease epsilon during time
        if self.memCntr > 40000:
            if self.epsilon > 0.01:
                # decrease linearly, slowly so there is a lot of exploration
                self.epsilon -= 0.0001
            else:
                self.epsilon = 0.01
