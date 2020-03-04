import numpy as np

class nn():

    # nn properties
    input_nodes_num = 0
    hidden_nodes_num = 0
    M = 0

    # input (CNN) layer
    W_in = []
    e = []

    # hidden (LSTM) layer
    W_h_f = []
    W_h_f_1 = []
    W_h_f_2 = []

    W_h_i = []
    W_h_i_1 = []
    W_h_i_2 = []

    W_h_o = []
    W_h_o_1 = []
    W_h_o_2 = []

    W_h_C = []
    W_h_C_1 = []
    W_h_C_2 = []

    # memory values
    h_last = []
    C_last = []
    h_cur = []
    C_cur = []

    # output layer
    W_out = []



    def __init__(self, i_num, hidden_num, M, X_len):

        # nn properties
        self.input_nodes_num = i_num
        self.hidden_nodes_num = hidden_num
        self.M = M

        # input (CNN) layer
        self.W_in = np.array([np.hamming(M) for n in range(i_num)])
        self.e = np.array([np.exp(1j*2*np.pi*n*np.linspace(0, M-1, M)/M) for n in range(i_num)])

        # hidden (LSTM) layer
        self.W_h_f = np.transpose(np.array([np.random.normal(0, 1, hidden_num) for n in range(hidden_num)]))
        self.W_h_f_1 = np.transpose(np.array([np.random.normal(0, .01, i_num) for n in range(hidden_num)]))
        self.W_h_f_2 = np.array([np.random.normal(0, .01, X_len + M - 1) for n in range(hidden_num)])

        self.W_h_i = np.transpose(np.array([np.random.normal(0, 1, hidden_num) for n in range(hidden_num)]))
        self.W_h_i_1 = np.transpose(np.array([np.random.normal(0, .01, i_num) for n in range(hidden_num)]))
        self.W_h_i_2 = np.array([np.random.normal(0, .01, X_len + M - 1) for n in range(hidden_num)])

        self.W_h_o = np.transpose(np.array([np.random.normal(0, 1, hidden_num) for n in range(hidden_num)]))
        self.W_h_o_1 = np.transpose(np.array([np.random.normal(0, .01, i_num) for n in range(hidden_num)]))
        self.W_h_o_2 = np.array([np.random.normal(0, .01, X_len + M - 1) for n in range(hidden_num)])

        self.W_h_C = np.transpose(np.array([np.random.normal(0, 1, hidden_num) for n in range(hidden_num)]))
        self.W_h_C_1 = np.transpose(np.array([np.random.normal(0, .01, i_num) for n in range(hidden_num)]))
        self.W_h_C_2 = np.array([np.random.normal(0, .01, X_len + M - 1) for n in range(hidden_num)])

        # memory values
        self.h_last = np.zeros(hidden_num)
        self.C_last = np.zeros(hidden_num)
        self.h_cur = np.zeros(hidden_num)
        self.C_cur = np.zeros(hidden_num)

        # output layer
        self.W_out = np.array([np.random.normal(0, 1, hidden_num) for n in range(2)])

    def feedforward(self, X):
        # set last values in memory
        self.h_last = self.h_cur
        self.C_last = self.C_cur

        # input layer
        conv_out = np.array([np.convolve(X, self.e[i] + self.W_in[i]) for i in range(self.W_in.shape[0])])
        H = np.square(np.absolute(conv_out))

        # hidden layer

        # f_t = forget gate activation
        # o_t = forget gate activation
        # i_t = forget gate activation
        f_t = np.array([sigmoid(np.inner(np.matmul(self.W_h_f_1[i], H), self.W_h_f_2[i]) + np.inner(self.W_h_f[i], self.h_last)) \
               for i in range(self.hidden_nodes_num)])
        i_t = np.array([sigmoid(np.inner(np.matmul(self.W_h_i_1[i], H), self.W_h_i_2[i]) + np.inner(self.W_h_i[i], self.h_last)) \
               for i in range(self.hidden_nodes_num)])
        o_t = np.array([sigmoid(np.inner(np.matmul(self.W_h_o_1[i], H), self.W_h_o_2[i]) + np.inner(self.W_h_o[i], self.h_last)) \
               for i in range(self.hidden_nodes_num)])

        # C_in_t = memory in
        C_in_t = np.array([np.tanh(np.inner(np.matmul(self.W_h_f_1[i], H), self.W_h_f_2[i]) + np.inner(self.W_h_f[i], self.h_last)) \
               for i in range(self.hidden_nodes_num)])

        # new hidden layer activation and memory
        C_t = (f_t*self.C_last) + (i_t*C_in_t)
        h_t = o_t*np.tanh(C_t)

        # set new values in memory
        self.C_cur = C_t
        self.h_cur = h_t

        # output layer
        h_out = np.array([sigmoid(np.inner(self.W_out[i], h_t)) for i in range(2)])

        return h_out

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

if __name__ == '__main__':
    # nn properties
    in_node_num = 10
    hidden_node_num = 10
    M = 5

    # data, dimension, and number of points
    dim = 100
    X_len = 52
    X = np.array([np.random.normal(0, 1, dim) for n in range(X_len)])

    test = nn(in_node_num, hidden_node_num, M, dim)

    for i in range(X_len):
        print(test.feedforward(X[i]))