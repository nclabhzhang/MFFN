import torch
import torch.nn as nn
from model.dataset import embedding_dim, hidden_size

class MFFN(nn.Module):
    def __init__(self):
        super(MFFN, self).__init__()
        self.conv0 = nn.Sequential(  # input_dim-->[batch*3, 1024, 500]
            nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # output_dim-->[batch*3, 128, 500]
            nn.Conv1d(in_channels=128, out_channels=80, kernel_size=49, stride=1, padding=30),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),  # output_dim-->[batch*3, 80, 256]
            nn.Conv1d(in_channels=80, out_channels=40, kernel_size=31, stride=1, padding=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),  # output_dim-->[batch*3, 40, 128]
            nn.Conv1d(in_channels=40, out_channels=16, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )  # output_dim-->[batch*3, 16, 64]
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=96, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=96, out_channels=64, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.pooler = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Tanh())
        self.relu = nn.ReLU()

        # Fully connected layer to shink AB
        self.fc_AB = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(embedding_dim * 2, int(embedding_dim / 2)),
            nn.BatchNorm1d(int(embedding_dim / 2)),
            nn.ReLU()
        )

        # Multi-Head Attention layer to aggregate information from proteins A,B
        self.attention_AB = nn.MultiheadAttention(embed_dim=2, num_heads=1, batch_first=True)
        # Transformer Encoder layer to aggregate information from protein complex pair P,Q
        self.attention_PQ = nn.MultiheadAttention(embed_dim=2, num_heads=1, batch_first=True)

        self.GRU = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.gru_query = nn.Linear(hidden_size*2, 1, bias=False)
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=96, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=96, out_channels=64, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(embedding_dim*2 + embedding_dim*2 + embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, data, _):
        data = data.view(-1, 500, embedding_dim)  # (batch_size*3, 500, embedding_dim)
        CLS = data[:, 0].squeeze()  # (batch_size*3, embedding_dim)
        CLS = self.pooler(CLS)  # the equivalent implementation of Bert-pooler moudle, (batch_size*3, embedding_dim)
        CLS = CLS.view(-1, 3, embedding_dim)  # (batch_size, 3, embedding_dim)
        CLS = CLS / torch.mean(self.relu(CLS), dim=2, keepdim=True) * 2
        A = CLS[:, 0]  # (batch_size, embedding_dim)
        B = CLS[:, 1]
        B_MUT = CLS[:, 2]
        # Present in the form of [A-B,A-B',A-B,A-B'...], and the aggregate information from protein A and B
        AB__AB_MUT = torch.stack((torch.stack((A, B), dim=2), torch.stack((A, B_MUT), dim=2)), dim=1).view(-1, 2*embedding_dim, 2)  # (batch_size, 2*embedding_dim, 2)
        AB__AB_MUT = self.attention_AB(AB__AB_MUT, AB__AB_MUT, AB__AB_MUT, need_weights=False)[0]
        AB__AB_MUT = AB__AB_MUT.reshape(-1, 2 * embedding_dim)  # (batch_size*2, 2*embedding_dim)
        AB__AB_MUT = self.fc_AB(AB__AB_MUT)  # (batch_size*2, embedding_dim/2)
        PQ = AB__AB_MUT.view(-1, 2, int(embedding_dim / 2))  # (batch_size, 2, embedding_dim/2)
        PQ = PQ.permute(0, 2, 1)  # (batch_size, embedding_dim/2 , 2)
        # Aggregate information from protein pairs p and q
        PQ = PQ / torch.mean(self.relu(PQ), dim=1, keepdim=True)  # (batch_size, embedding_dim/2 , 2)
        PQ = self.attention_PQ(PQ, PQ, PQ, need_weights=False)[0]  # (batch_size, embedding_dim/2,  2)
        PQ = PQ.reshape(-1, embedding_dim)  # (batch_size, embedding_dim)

        data = data.permute(0, 2, 1)  # [batch*3, 500, 1024] --> [batch*3, 1024, 500]
        conv1_out = self.conv0(data)  # output_dim --> [batch*3, 16, 64]

        conv2_in = conv1_out.view(-1, 3, 16, 64)  # [batch, 3, 16, 64]
        conv2_in = conv2_in.permute(1, 0, 3, 2)  # [3, batch, 64, 16]
        conv2_in = torch.stack([torch.cat([conv2_in[0], conv2_in[1]], dim=1),
                                torch.cat([conv2_in[0], conv2_in[2]], dim=1)], dim=1)
        conv2_in = conv2_in.view(-1, 128, 16)  # [batch*2, 128 ,16]
        conv2_out = self.conv2(conv2_in)   # [batch*2, 64, 16]

        data = data.permute(0, 2, 1)  # [batch*3, 1024, 500]-->[batch*3, 500, 1024]
        gru_out, _ = self.GRU(data)  # [batch*3, 500, 2*hidden_size]
        A = self.gru_query(gru_out)
        A = torch.softmax(gru_out, 1)
        gru_out = torch.sum(gru_out * A, 1)  # [batch*3, 2*hidden_size]
        conv3_in = gru_out.view(-1, 3, 16, 64)
        conv3_in = conv3_in.permute(1, 0, 3, 2)  # [3, batch, 64, 16]
        conv3_in = torch.stack([torch.cat([conv3_in[0], conv3_in[1]], dim=1),
                                torch.cat([conv3_in[0], conv3_in[2]], dim=1)], dim=1)
        conv3_in = conv3_in.view(-1, 128, 16)
        conv3_out = self.conv3(conv3_in)

        fc_in = torch.cat((conv2_out.view(-1, embedding_dim * 2), conv3_out.view(-1, embedding_dim * 2), PQ), dim=1)  # [batch, embedding_dim*5]
        fc_out = self.fc(fc_in)
        return fc_out

