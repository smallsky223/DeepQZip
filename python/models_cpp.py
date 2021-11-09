import torch as t

# pytorch
class QVRNNCpp(t.nn.Module):
    def __init__(self, alphabet_size):
        super(QVRNNCpp, self).__init__()
        self.embed1 = t.nn.Embedding(alphabet_size, 16)
        self.embed2 = t.nn.Embedding(94, 32)
        self.birnn = t.nn.LSTM(48, 32, 2, batch_first=True, bidirectional = True)
        self.fc1 = t.nn.Linear(64, 64)
        self.fc2 = t.nn.Linear(64, alphabet_size)
        self.ac1 = t.nn.ReLU()

    def forward(self, inp_x, inp_q): # (bs, 64)
        x = self.embed1(inp_x) # (bs, 64, 16)
        q = self.embed2(inp_q) # (bs, 64, 32)
        xq = t.cat((x, q), -1) # (bs, 64, 48)
        self.birnn.flatten_parameters()
        outputs,(h_n,c_n) = self.birnn(xq) # (bs, 64, 64)
        outputs = outputs[:, -1, :] # (bs, 64)
        outputs = self.fc1(outputs) # (bs, 64)
        outputs = self.ac1(outputs)
        outputs = self.fc2(outputs) # (bs, 4)
        ts=t.nn.functional.softmax(outputs, dim=1)
        tff=t.floor(ts * 1e8)
        outputs = tff.int()
        outputs[outputs == 0] = 1
        
        for i in range(outputs.shape[1] - 1):
            outputs[:, i + 1] += outputs[:, i]
        return outputs.reshape((-1))


"""
# keras
def biLSTM(bs,time_steps,alphabet_size):
        inputs_bits = Input(shape=(time_steps, 2))
        x = Lambda(lambda tensor: tensor[:,:,0])(inputs_bits)
        q = Lambda(lambda tensor: tensor[:,:,1])(inputs_bits)
        x = Embedding(alphabet_size, 32,)(x)
        q = Reshape((time_steps, 1))(q)
        x = Concatenate(axis=-1)([x, q])
        x = Bidirectional(CuDNNLSTM(32, stateful=False, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(32, stateful=False, return_sequences=False))(x)
        x = Dense(64, activation='relu')(x)
        y = Dense(alphabet_size, activation='softmax')(x)
        model = Model(inputs_bits, y)
        return model
"""
