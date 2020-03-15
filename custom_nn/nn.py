import numpy as np
from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.timeseries import TimeSeries


class nn():
    # nn properties
    input_nodes_num = 0
    hidden_nodes_num = 0
    M = 0

    # input (CNN) layer
    W_in = []
    e = []

    # hidden (LSTM) layer
    W_f_1 = []
    W_f_2 = []
    W_f_h = []

    W_i_1 = []
    W_i_2 = []
    W_i_h = []

    W_o_1 = []
    W_o_2 = []
    W_o_h = []

    W_C_1 = []
    W_C_2 = []
    W_C_h = []

    # memory values
    h_last = []
    C_last = []
    h_cur = []
    C_cur = []

    # output layer
    W_out = []

    # previous outputs
    z_k = []
    y_j = []
    f_t = []
    i_t = []
    o_t = []
    C_in_t = []
    C_t = []
    h_t = []
    H_t = []

    out_1 = []

    def __init__(self, i_num, hidden_num, M, X_len):

        # nn properties
        self.input_nodes_num = i_num
        self.hidden_nodes_num = hidden_num
        self.M = M

        # input (CNN) layer
        # self.W_in = np.array([np.hamming(M) for n in range(i_num)])
        self.W_in = np.array([np.random.normal(0, 1, M) for n in range(i_num)])
        self.e = np.array([np.exp(1j * 2 * np.pi * n * np.linspace(0, M - 1, M) / M) for n in range(i_num)])

        # hidden (LSTM) layer
        self.W_f_1 = np.array([np.random.normal(0, 1, i_num) for n in range(hidden_num)])
        self.W_f_2 = np.array([np.random.normal(0, 1, X_len + M - 1) for n in range(hidden_num)])
        self.W_f_h = np.array([np.random.normal(0, 1, hidden_num) for n in range(hidden_num)])

        self.W_i_1 = np.array([np.random.normal(0, 1, i_num) for n in range(hidden_num)])
        self.W_i_2 = np.array([np.random.normal(0, 1, X_len + M - 1) for n in range(hidden_num)])
        self.W_i_h = np.array([np.random.normal(0, 1, hidden_num) for n in range(hidden_num)])

        self.W_o_1 = np.array([np.random.normal(0, 1, i_num) for n in range(hidden_num)])
        self.W_o_2 = np.array([np.random.normal(0, 1, X_len + M - 1) for n in range(hidden_num)])
        self.W_o_h = np.array([np.random.normal(0, 1, hidden_num) for n in range(hidden_num)])

        self.W_C_1 = np.array([np.random.normal(0, 1, i_num) for n in range(hidden_num)])
        self.W_C_2 = np.array([np.random.normal(0, 1, X_len + M - 1) for n in range(hidden_num)])
        self.W_C_h = np.array([np.random.normal(0, 1, hidden_num) for n in range(hidden_num)])

        # memory values
        self.clear_memory()

        # output layer
        self.W_out = np.array([np.random.normal(0, 1, hidden_num) for n in range(2)])

    def feedforward(self, X):
        # set last values in memory
        self.h_last = self.h_cur
        self.C_last = self.C_cur

        # input layer
        conv_out = np.array([np.convolve(X, self.e[i] * self.W_in[i]) for i in range(self.W_in.shape[0])])
        H = np.square(np.absolute(conv_out))
        self.H_t.append(H)
        # print(1)
        # print(H.shape)
        # print(self.W_f_1.shape)
        # print(self.W_f_1[0].shape)

        # hidden layer

        # f_t = forget gate activation
        # o_t = out gate activation
        # i_t = input gate activation
        # print("hmmmm")
        # print(self.W_f_h[0])
        # print(self.hidden_nodes_num)
        f_t = np.array(
            [sigmoid(np.inner(np.matmul(self.W_f_1[i], H), self.W_f_2[i]) + np.inner(self.W_f_h[i], self.h_last)) \
             for i in range(self.hidden_nodes_num)])
        i_t = np.array(
            [sigmoid(np.inner(np.matmul(self.W_i_1[i], H), self.W_i_2[i]) + np.inner(self.W_i_h[i], self.h_last)) \
             for i in range(self.hidden_nodes_num)])
        o_t = np.array(
            [sigmoid(np.inner(np.matmul(self.W_o_1[i], H), self.W_o_2[i]) + np.inner(self.W_o_h[i], self.h_last)) \
             for i in range(self.hidden_nodes_num)])

        # C_in_t = memory in
        C_in_t = np.array(
            [np.tanh(np.inner(np.matmul(self.W_f_1[i], H), self.W_f_2[i]) + np.inner(self.W_f_h[i], self.h_last)) \
             for i in range(self.hidden_nodes_num)])

        # add to previous outputs
        # print(f_t)
        self.f_t = np.append(self.f_t, [f_t], axis=0)
        # print(1)
        self.i_t = np.append(self.i_t, [i_t], axis=0)
        self.o_t = np.append(self.o_t, [o_t], axis=0)
        self.C_in_t = np.append(self.C_in_t, [C_in_t], axis=0)

        # new hidden layer activation and memory
        C_t = (f_t * self.C_last) + (i_t * C_in_t)
        h_t = o_t * np.tanh(C_t)

        # set new values in memory
        self.C_cur = C_t
        self.h_cur = h_t

        # add to previous outputs
        # self.C_t = np.append(self.C_t, [C_t], axis=0)
        self.C_t.append(C_t)
        self.h_t = np.append(self.h_t, [h_t], axis=0)

        # add to previous outputs
        self.y_j = np.append(self.y_j, [h_t], axis=0)

        # output layer
        # h_out = np.array([sigmoid(np.inner(self.W_out[i], h_t)) for i in range(2)])
        h_out = np.array([np.inner(self.W_out[i], h_t) for i in range(2)])
        a_out = self.softmax(h_out)

        # add to previous outputs
        self.z_k = np.append(self.z_k, [a_out], axis=0)

        return a_out

    def backprop(self, X, Y):
        # print(self.z_k[:,0])
        L = len(Y)
        eta_out = 2e-3
        eta_hidden = 1e-6
        eta_in = 0

        #
        # output layer
        #
        grad1 = np.sum(np.array([self.y_j[i] * ((1 * (Y[i] == 0) - 0) - self.z_k[i][0]) for i in range(L)]),
                       axis=0).tolist()
        grad2 = np.sum(np.array([self.y_j[i] * ((1 * (Y[i] == 1) - 0) - self.z_k[i][1]) for i in range(L)]),
                       axis=0).tolist()

        grad = []
        grad.append(grad1)
        grad.append(grad2)

        # self.W_out = np.array(grad)
        self.W_out = self.W_out + eta_out * np.array(grad)

        #
        # hidden layer
        #
        dhdc_t = np.array([self.o_t[i] * (1 - np.square(np.tanh(self.C_t[i]))) for i in range(L)])
        h_shift = np.array([self.h_t[i - 1] if i > 0 else np.zeros(self.hidden_nodes_num) for i in range(L)])
        dc_tdf_t = np.array([self.C_t[i - 1] if i > 0 else np.zeros(self.hidden_nodes_num) for i in range(L)])
        dc_tdi_t = self.C_in_t  # previously named dc_tdc_i_t
        dc_tdc_in_t = self.i_t
        dh_to_t = np.tanh(self.C_t)  # previously named dh_tdw_ho

        new_W_f_1 = []
        new_W_f_2 = []
        new_W_f_h = []

        new_W_i_1 = []
        new_W_i_2 = []
        new_W_i_h = []

        new_W_o_1 = []
        new_W_o_2 = []
        new_W_o_h = []

        new_W_C_1 = []
        new_W_C_2 = []
        new_W_C_h = []
        for j in range(self.hidden_nodes_num):
            dendh_t = np.array([(2 * Y[i] - 1) * (1 - self.z_k[i][Y[i]]) * self.W_out[Y[i]][j] for i in range(L)])

            # f gradients
            df_tdw_f_1 = np.array(
                [np.matmul(self.H_t[i], self.W_f_2[j]) * self.f_t[i][j] * (1 - self.f_t[i][j]) for i in range(L)])
            df_tdw_f_2 = np.array(
                [np.matmul(self.H_t[i].T, self.W_f_1[j]) * self.f_t[i][j] * (1 - self.f_t[i][j]) for i in range(L)])
            df_tdw_f_h = np.array([h_shift[i - 1][j] * self.f_t[i][j] * (1 - self.f_t[i][j]) for i in range(L)])

            temp1 = dhdc_t[:, j] * dc_tdf_t[:, j]
            dh_tw_f_1 = (df_tdw_f_1.T * temp1).T
            dh_tw_f_2 = (df_tdw_f_2.T * temp1).T
            dh_tw_f_h = (df_tdw_f_h.T * temp1).T

            new_W_f_1.append(np.sum((dh_tw_f_1.T * dendh_t).T, axis=0))
            new_W_f_2.append(np.sum((dh_tw_f_2.T * dendh_t).T, axis=0))
            new_W_f_h.append(np.sum((dh_tw_f_h.T * dendh_t).T, axis=0))

            # i gradients
            df_tdw_i_1 = np.array(
                [np.matmul(self.H_t[i], self.W_i_2[j]) * self.i_t[i][j] * (1 - self.i_t[i][j]) for i in range(L)])
            df_tdw_i_2 = np.array(
                [np.matmul(self.H_t[i].T, self.W_i_1[j]) * self.i_t[i][j] * (1 - self.i_t[i][j]) for i in range(L)])
            df_tdw_i_h = np.array([h_shift[i - 1][j] * self.i_t[i][j] * (1 - self.i_t[i][j]) for i in range(L)])

            temp2 = dhdc_t[:, j] * dc_tdi_t[:, j]
            dh_tw_i_1 = (df_tdw_i_1.T * temp2).T
            dh_tw_i_2 = (df_tdw_i_2.T * temp2).T
            dh_tw_i_h = (df_tdw_i_h.T * temp2).T

            new_W_i_1.append(np.sum((dh_tw_i_1.T * dendh_t).T, axis=0))
            new_W_i_2.append(np.sum((dh_tw_i_2.T * dendh_t).T, axis=0))
            new_W_i_h.append(np.sum((dh_tw_i_h.T * dendh_t).T, axis=0))

            # c_in gradients
            dc_in_td_w_c_in_1 = np.array(
                [np.matmul(self.H_t[i], self.W_C_2[j]) * (1 - np.square(self.C_in_t[i][j])) for i in range(L)])
            dc_in_td_w_c_in_2 = np.array(
                [np.matmul(self.H_t[i].T, self.W_C_1[j]) * (1 - np.square(self.C_in_t[i][j])) for i in range(L)])
            dc_in_td_w_c_in_h = np.array([h_shift[i - 1][j] * (1 - np.square(self.C_in_t[i][j])) for i in range(L)])

            temp3 = dhdc_t[:, j] * dc_tdc_in_t[:, j]
            dh_tw_c_in_1 = (dc_in_td_w_c_in_1.T * temp2).T
            dh_tw_c_in_2 = (dc_in_td_w_c_in_2.T * temp2).T
            dh_tw_c_in_h = (dc_in_td_w_c_in_h.T * temp2).T

            new_W_C_1.append(np.sum((dh_tw_c_in_1.T * dendh_t).T, axis=0))
            new_W_C_2.append(np.sum((dh_tw_c_in_2.T * dendh_t).T, axis=0))
            new_W_C_h.append(np.sum((dh_tw_c_in_h.T * dendh_t).T, axis=0))

            # o gradients
            do_td_w_o_1 = np.array(
                [np.matmul(self.H_t[i], self.W_o_2[j]) * self.o_t[i][j] * (1 - self.o_t[i][j]) for i in range(L)])
            do_td_w_o_2 = np.array(
                [np.matmul(self.H_t[i].T, self.W_C_1[j]) * self.o_t[i][j] * (1 - self.o_t[i][j]) for i in range(L)])
            do_td_w_o_h = np.array([h_shift[i - 1][j] * self.o_t[i][j] * (1 - self.o_t[i][j]) for i in range(L)])

            temp4 = dh_to_t[:, j]
            dh_tw_o_1 = (do_td_w_o_1.T * temp4).T
            dh_tw_o_2 = (do_td_w_o_2.T * temp4).T
            dh_tw_o_h = (do_td_w_o_h.T * temp4).T

            new_W_o_1.append(np.sum((dh_tw_o_1.T * dendh_t).T, axis=0))
            new_W_o_2.append(np.sum((dh_tw_o_2.T * dendh_t).T, axis=0))
            new_W_o_h.append(np.sum((dh_tw_o_h.T * dendh_t).T, axis=0))

        # set new values
        # print(np.array(new_W_f_1))
        # print(self.W_f_1)
        # self.W_f_1 = self.W_f_1 + np.array(new_W_f_1)
        # self.W_f_2 = self.W_f_2 + np.array(new_W_f_2)
        # self.W_f_h = self.W_f_h + np.array(new_W_f_h)
        # self.W_i_1 = self.W_i_1 + np.array(new_W_i_1)
        # self.W_i_2 = self.W_i_2 + np.array(new_W_i_2)
        # self.W_i_h = np.array(new_W_i_h)
        # self.W_C_1 = np.array(new_W_C_1)
        # self.W_C_2 = np.array(new_W_C_2)
        # self.W_C_h = np.array(new_W_C_h)
        # self.W_o_1 = np.array(new_W_C_1)
        # self.W_o_2 = np.array(new_W_C_2)
        # self.W_o_h = np.array(new_W_C_h)

        self.W_f_1 = self.W_f_1 - eta_hidden * np.array(new_W_f_1)
        self.W_f_2 = self.W_f_2 - eta_hidden * np.array(new_W_f_2)
        self.W_f_h = self.W_f_h - eta_hidden * np.array(new_W_f_h)
        self.W_i_1 = self.W_i_1 - eta_hidden * np.array(new_W_i_1)
        self.W_i_2 = self.W_i_2 - eta_hidden * np.array(new_W_i_2)
        self.W_i_h = self.W_i_h - eta_hidden * np.array(new_W_i_h)
        self.W_C_1 = self.W_C_1 - eta_hidden * np.array(new_W_C_1)
        self.W_C_2 = self.W_C_2 - eta_hidden * np.array(new_W_C_2)
        self.W_C_h = self.W_C_h - eta_hidden * np.array(new_W_C_h)
        self.W_o_1 = self.W_o_1 - eta_hidden * np.array(new_W_C_1)
        self.W_o_2 = self.W_o_2 - eta_hidden * np.array(new_W_C_2)
        self.W_o_h = self.W_o_h - eta_hidden * np.array(new_W_C_h)

        # print(self.W_f_1)

        #
        # input layer
        #
        cos_a = np.array(
            [np.cos(2 * np.pi * n * np.linspace(0, self.M - 1, self.M) / self.M) for n in range(self.input_nodes_num)])
        sin_a = np.array(
            [np.sin(2 * np.pi * n * np.linspace(0, self.M - 1, self.M) / self.M) for n in range(self.input_nodes_num)])

        tj_out = np.empty((L, self.M))
        L_array = []
        for t in range(L):
            in_array = []
            for j in range(self.input_nodes_num):
                pad_cos_a = np.pad(cos_a[j], (len(X[0]) - 1, 0), 'constant', constant_values=0)
                pad_sin_a = np.pad(cos_a[j], (len(X[0]) - 1, 0), 'constant', constant_values=0)
                dendh_t = np.array([(2 * Y[t] - 1) * (1 - self.z_k[t][Y[t]]) * self.W_out[Y[t]][j]])
                # print(dendh_t.shape)

                # print(do_tdg_t.shape)
                # df_tdg_t =
                # di_tdg_t =
                # dc_indg_t =

                # conv_cos_out = np.array([np.convolve(X[t], cos_a[i] * self.W_in[i]) for i in range(self.W_in.shape[0])])
                # conv_sin_out = np.array([np.convolve(X[t], sin_a[i] * self.W_in[i]) for i in range(self.W_in.shape[0])])
                # conv_cos_out = np.array([np.convolve(X[i], cos_a[j] * self.W_in[j]) for i in range(L)])
                # conv_sin_out = np.array([np.convolve(X[i], sin_a[j] * self.W_in[j]) for i in range(L)])
                conv_cos_out = np.convolve(X[t], cos_a[j] * self.W_in[j])
                conv_sin_out = np.convolve(X[t], sin_a[j] * self.W_in[j])

                # print(conv_cos_out.shape)

                # dg_tdw_j = np.array([2*cos_a[j]*X[i]*conv_cos_out[i][self.M - 1] + 2*sin_a[j]*X[i]*conv_sin_out[i][self.M - 1] \
                #           for i in range(L)])

                # pad_X_t = np.pad(X[t], (M - 1, 0), 'constant', constant_values=0)
                dg_tdw_j = np.array(
                    [2 * pad_cos_a * np.pad(X[t], (i, M - 1 - i), 'constant', constant_values=0) * conv_cos_out[i] \
                     + 2 * pad_sin_a * np.pad(X[t], (i, M - 1 - i), 'constant', constant_values=0) * conv_sin_out[i] for
                     i in range(M)]).T
                # print(dg_tdw_j.shape)

                do_tdg_t_temp = []
                term_2 = []
                for n in range(self.hidden_nodes_num):
                    do_tdg_tj = self.W_o_1[n, j] * self.W_o_2[n]

                    df_tdg_tj = self.W_f_1[n, j] * self.W_f_2[n]
                    di_tdg_tj = self.W_i_1[n, j] * self.W_i_2[n]
                    dc_in_tdg_tj = self.W_C_1[n, j] * self.W_C_2[n]

                    temp = dc_tdf_t[t, n] * df_tdg_tj + dc_tdi_t[t, n] * di_tdg_tj + dc_tdc_in_t[t, n] * dc_in_tdg_tj
                    term_2.append(dg_tdw_j.T @ temp)

                    do_tdg_t_temp.append(dg_tdw_j.T @ do_tdg_tj)

                do_tdg_t_sum = np.sum(np.array(do_tdg_t_temp), axis=0)
                term_2_sum = np.sum(np.array(term_2), axis=0)

                in_array.append(dendh_t * (dh_to_t[t, j] * do_tdg_t_sum + dhdc_t[t, j] * term_2_sum))
            L_array.append(in_array)
        L_array = np.array(L_array)

        weight_updates = np.array(
            [np.sum(np.array([L_array[i, j] for i in range(L)]), axis=0) for j in range(self.input_nodes_num)])

        # print(self.W_in)
        # print(weight_updates)
        # self.W_in = weight_updates
        # self.W_in = self.W_in - eta*weight_updates
        # print(self.W_out)

        # print("Output Layer Change: " + str(np.linalg.norm(eta_out*np.array(grad))))
        # print("Hidden Layer Change: " + str(np.linalg.norm(eta_hidden*np.array(new_W_f_1))))
        # print("Input Layer Change: " + str(np.linalg.norm(eta_in*weight_updates)))

        self.W_in = self.W_in - eta_in * weight_updates

    def clear_memory(self):
        # memory values
        self.h_last = np.zeros(self.hidden_nodes_num)
        self.C_last = np.zeros(self.hidden_nodes_num)
        self.h_cur = np.zeros(self.hidden_nodes_num)
        self.C_cur = np.zeros(self.hidden_nodes_num)
        self.z_k = np.empty((0, 2))
        self.y_j = np.empty((0, self.hidden_nodes_num))

        self.f_t = np.empty((0, self.hidden_nodes_num))
        self.i_t = np.empty((0, self.hidden_nodes_num))
        self.o_t = np.empty((0, self.hidden_nodes_num))
        self.C_in_t = np.empty((0, self.hidden_nodes_num))
        # self.C_t = np.empty((0, self.hidden_nodes_num))
        self.C_t = []
        self.h_t = np.empty((0, self.hidden_nodes_num))
        self.H_t = []

        self.out_1 = np.empty((0, self.hidden_nodes_num))

    def softmax(self, h):
        exp_h = np.exp(h)
        return exp_h / np.sum(exp_h)

    def acc(self, Y):
        preds = self.classify()
        return np.sum(preds == Y) / len(Y)

    def classify(self):
        preds = []
        for i in range(self.z_k.shape[0]):
            preds.append(self.z_k[i, 0] > self.z_k[i, 1])
        return preds


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def decimate_series(data, R, M):
    x = np.array(data)
    L = len(x)
    X = np.array([x[R * i: R * i + M] for i in range(int(np.floor((L - M) / R)))])
    Y = np.array(
        [int(max(x[R * i: R * i + M]) < max(x[R * i + M: R * i + M + 5])) for i in range(int(np.floor((L - M) / R)))])
    return X, Y


if __name__ == '__main__':
    ts = TimeSeries(key='8VHYYYZR0BTCNJ2F', output_format='pandas')
    df, _ = ts.get_daily(symbol='GOOG', outputsize='full')
    minmax = MinMaxScaler().fit(df.iloc[0:253, 3:4].astype('float32'))  # Close index
    df_log = minmax.transform(df.iloc[0:253, 3:4].astype('float32'))  # Close index
    data = df_log

    # df_log = pd.DataFrame(df_log)

    # data = df.iloc[0:253, 3:4].values.tolist()

    scale = 10
    x = [point[0] / scale for point in data]
    # nn properties
    in_node_num = 10
    hidden_node_num = 10
    M = 5

    # data, dimension, and number of points
    dim = 5
    R = 1
    X, Y = decimate_series(x, R, dim)
    test = nn(in_node_num, hidden_node_num, M, dim)

    num_batches = 1
    batch_size = int(np.floor(X.shape[0] / num_batches));
    iterations = 300
    for it in range(iterations):
        for j in range(num_batches):
            for i in range(batch_size):
                test.feedforward(X[batch_size * j + i])
            test.backprop(X[batch_size * j:batch_size * (j + 1)], Y[batch_size * j:batch_size * (j + 1)])
            test.clear_memory()
        for j in range(X.shape[0]):
            test.feedforward(X[i])
        print("Epoch: " + str(it + 1) + ". Accuracy: " + str(test.acc(Y)))
        print('-------')
        test.clear_memory()
#     for it in range(iterations):
#         for i in range(X.shape[0]):
#             test.feedforward(X[i])
#         print("Iteration: " + str(it + 1) + " " + str(test.acc(Y)))
#         print('-------')
#         test.backprop(X, Y)
#         test.clear_memory()
