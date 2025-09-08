# === Regression Results ===
# MAE: 1.435, RMSE: 2.211
# === Classification Report ===
#              precision    recall  f1-score   support
#
#         0.0      0.974     0.949     0.961        39
#         1.0      0.889     0.941     0.914        17
#
#    accuracy                          0.946        56
#   macro avg      0.931     0.945     0.938        56
#weighted avg      0.948     0.946     0.947        56

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
import spacy

# 常量设置
DEFAULT_AUDIO_FEAT_DIM = 102
DEFAULT_MFCC_FEAT_DIM = 13
TOTAL_AUDIO_FEAT_DIM = DEFAULT_AUDIO_FEAT_DIM + DEFAULT_MFCC_FEAT_DIM

DEFAULT_VISUAL_FEAT_DIM = 53  
TFIDF_DIM = 100             
MAX_TEXT_LENGTH = 50
SCORE_FEAT_DIM = 16       
FUSION_DIM = 128  # 融合后公共维度  

#########################################
# 数据集：读取文本、音频和视觉（视频）模态数据（支持数据增强）
#########################################
class UnzippedDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_df, audio_seq_len=60, visual_seq_len=60, tfidf_vectorizer=None, augment=False):
        self.root_dir = root_dir
        self.audio_seq_len = audio_seq_len
        self.visual_seq_len = visual_seq_len
        self.data_info = data_df.reset_index(drop=True)
        self.tfidf_vectorizer = tfidf_vectorizer
        self.augment = augment  # 是否进行数据增强

    def read_csv_df(self, csv_path):
        if not os.path.isfile(csv_path):
            print(f"警告：未找到文件 {csv_path}")
            return None
        df = pd.read_csv(csv_path)
        return df

    def process_timeseries(self, df, seq_len, feat_dim):
        df.replace('unknown', np.nan, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.fillna(0, inplace=True)
        data = df.values.astype(np.float32)
        if data.shape[1] > feat_dim:
            data = data[:, :feat_dim]
        elif data.shape[1] < feat_dim:
            pad_cols = feat_dim - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad_cols)), mode='constant')
        if data.shape[0] > seq_len:
            data = data[:seq_len, :]
        elif data.shape[0] < seq_len:
            pad_time = seq_len - data.shape[0]
            data = np.pad(data, ((0, pad_time), (0, 0)), mode='constant')
        return data

    def augment_sample(self, text, audio_data, tfidf):
        # 随机删除 10% 的单词；音频与 TF-IDF 加入轻微高斯噪声
        words = text.split()
        if len(words) > 0:
            keep_words = [w for w in words if np.random.rand() > 0.1]
            if len(keep_words) == 0:
                keep_words = words
            aug_text = " ".join(keep_words)
        else:
            aug_text = text
        noise_audio = np.random.normal(0, 0.01, audio_data.shape).astype(np.float32)
        aug_audio = audio_data + noise_audio
        noise_tfidf = np.random.normal(0, 0.01, tfidf.shape).astype(np.float32)
        aug_tfidf = tfidf + noise_tfidf
        return aug_text, aug_audio, aug_tfidf

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        participant_id = str(row["Participant_ID"])
        score = row["PHQ_Score"]
        # 新增二分类标签
        phq_binary = row["PHQ_Binary"]
        p_dir = os.path.join(self.root_dir, f"{participant_id}_P")
        
        # 文本：从 transcript 文件中读取文本
        transcript_path = os.path.join(p_dir, f"{participant_id}_Transcript.csv")
        transcript_df = self.read_csv_df(transcript_path)
        if transcript_df is not None and "Text" in transcript_df.columns:
            text = " ".join(transcript_df["Text"].astype(str).tolist())
        else:
            text = ""
        
        # 计算 TF-IDF 特征
        if self.tfidf_vectorizer is not None:
            tfidf = self.tfidf_vectorizer.transform([text]).toarray()[0]
        else:
            tfidf = np.zeros((TFIDF_DIM,), dtype=np.float32)
        
        # 音频：读取预提取的特征
        audio_csv = os.path.join(p_dir, "features", f"{participant_id}_BoAW_openSMILE_2.3.0_eGeMAPS.csv")
        audio_df = self.read_csv_df(audio_csv)
        if audio_df is not None:
            audio_data_geMAPS = self.process_timeseries(audio_df, self.audio_seq_len, DEFAULT_AUDIO_FEAT_DIM)
        else:
            audio_data_geMAPS = np.zeros((self.audio_seq_len, DEFAULT_AUDIO_FEAT_DIM), dtype=np.float32)
        
        mfcc_csv = os.path.join(p_dir, "features", f"{participant_id}_BoAW_openSMILE_2.3.0_MFCC.csv")
        mfcc_df = self.read_csv_df(mfcc_csv)
        if mfcc_df is not None:
            audio_data_mfcc = self.process_timeseries(mfcc_df, self.audio_seq_len, DEFAULT_MFCC_FEAT_DIM)
        else:
            audio_data_mfcc = np.zeros((self.audio_seq_len, DEFAULT_MFCC_FEAT_DIM), dtype=np.float32)
        audio_data = np.concatenate([audio_data_geMAPS, audio_data_mfcc], axis=1)
        
        # 视觉（视频）：读取预提取的视频特征
        visual_csv = os.path.join(p_dir, "features", f"{participant_id}_OpenFace2.1.0_Pose_gaze_AUs.csv")
        visual_df = self.read_csv_df(visual_csv)
        if visual_df is not None:
            visual_data = self.process_timeseries(visual_df, self.visual_seq_len, DEFAULT_VISUAL_FEAT_DIM)
        else:
            visual_data = np.zeros((self.visual_seq_len, DEFAULT_VISUAL_FEAT_DIM), dtype=np.float32)
        
        # 数据增强（仅对少数类）
        if self.augment:
            text, audio_data, tfidf = self.augment_sample(text, audio_data, tfidf)
        
        sample = {
            "audio": torch.tensor(audio_data),
            "text": text,
            "visual": torch.tensor(visual_data),
            "tfidf": torch.tensor(tfidf, dtype=torch.float32),
            "score": torch.tensor(score, dtype=torch.float32),
            "PHQ_Binary": torch.tensor(phq_binary, dtype=torch.float32),
            "pid": participant_id
        }
        return sample

#########################################
# 多头自注意力模块（用于后续注意力聚合）
#########################################
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
    
    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        return attn_output

#########################################
# 结合多头自注意力与线性投影进行加权聚合
#########################################
class EnhancedAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(EnhancedAttention, self).__init__()
        self.mha = MultiHeadSelfAttention(input_dim, num_heads)
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        attn_output = self.mha(x)
        scores = self.fc(attn_output)  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context

#########################################
# 基本 GCN 层
#########################################
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)
        self.activation = nn.ReLU()
        if in_features != out_features:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = None

    def forward(self, x, adj):
        out = torch.bmm(adj, x)
        out = self.linear(out)
        out = self.activation(out)
        out = self.dropout(out)
        x_res = self.residual(x) if self.residual is not None else x
        out = self.layer_norm(out + x_res)
        return out

#########################################
# 堆叠 GCN 模块
#########################################
class StackedGCN(nn.Module):
    def __init__(self, in_features, out_features, num_layers=2, dropout=0.1):
        super(StackedGCN, self).__init__()
        layers = []
        for i in range(num_layers):
            in_dim = in_features if i == 0 else out_features
            layers.append(GCNLayer(in_dim, out_features, dropout))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x

#########################################
# 文本分支：基于 GCN 和 Transformer 编码器
#########################################
class TextBranch(nn.Module):
    def __init__(self, vocab_size=50000, embedding_dim=300, max_length=50, 
                 gcn_hidden=256, tfidf_dim=100, dropout=0.1, transformer_layers=2):
        super(TextBranch, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.max_length = max_length
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError("未找到 spaCy 模型 'en_core_web_sm'，请运行 'python -m spacy download en_core_web_sm'")
        self.gcn = GCNLayer(embedding_dim + tfidf_dim, gcn_hidden, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=gcn_hidden, nhead=4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.attention = EnhancedAttention(gcn_hidden, num_heads=4)
        self.fusion_layer_norm = nn.LayerNorm(gcn_hidden)

    def build_dependency_adj(self, tokens, T, device):
        adj = torch.zeros(T, T)
        for token in tokens:
            idx = token.i
            if idx >= T:
                break
            adj[idx, idx] = 1
            if token.head.i < T:
                adj[idx, token.head.i] = 1
                adj[token.head.i, idx] = 1
        return adj.to(device)

    def forward(self, text_list, tfidf):
        device = tfidf.device
        token_embeddings = []
        adj_matrices = []
        for text in text_list:
            doc = self.nlp(text)
            tokens = list(doc)[:self.max_length]
            token_ids = [token.orth % 50000 for token in tokens]
            if len(token_ids) < self.max_length:
                token_ids += [0] * (self.max_length - len(token_ids))
            token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
            emb = self.embedding(token_ids_tensor)
            token_embeddings.append(emb)
            adj = self.build_dependency_adj(tokens, self.max_length, device)
            adj_matrices.append(adj)
        token_embeddings = torch.stack(token_embeddings, dim=0)
        A = torch.stack(adj_matrices, dim=0)
        tfidf_expanded = tfidf.unsqueeze(1).repeat(1, self.max_length, 1)
        token_cat = torch.cat([token_embeddings, tfidf_expanded], dim=-1)
        D = torch.sum(A, dim=-1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag_embed(D_inv_sqrt)
        A_norm = torch.bmm(torch.bmm(D_inv_sqrt, A), D_inv_sqrt)
        gcn_out = self.gcn(token_cat, A_norm)
        gcn_out = self.transformer_encoder(gcn_out)
        text_repr = self.attention(gcn_out)
        text_repr = self.fusion_layer_norm(text_repr)
        return text_repr

#########################################
# 音频分支：使用堆叠 GCN + BiGRU + Transformer 编码器
#########################################
class AudioBranch(nn.Module):
    def __init__(self, input_dim=TOTAL_AUDIO_FEAT_DIM, gcn_channels=64, gru_hidden=32, num_gru_layers=1, 
                 bidirectional=True, dropout=0.1, transformer_layers=1):
        super(AudioBranch, self).__init__()
        self.gcn = StackedGCN(input_dim, gcn_channels, num_layers=2, dropout=dropout)
        self.bigru = nn.GRU(input_size=gcn_channels, hidden_size=gru_hidden, num_layers=num_gru_layers, 
                            batch_first=True, bidirectional=bidirectional)
        self.gru_output_dim = gru_hidden * 2 if bidirectional else gru_hidden
        self.attention = EnhancedAttention(self.gru_output_dim, num_heads=4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.gru_output_dim, nhead=4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

    def forward(self, x):
        B, T, _ = x.size()
        device = x.device
        A = torch.eye(T, device=device).unsqueeze(0).repeat(B, 1, 1)
        gcn_out = self.gcn(x, A)
        gru_out, _ = self.bigru(gcn_out)
        gru_out = self.transformer_encoder(gru_out)
        audio_feat = self.attention(gru_out)
        return audio_feat

#########################################
# 视频分支：结构与音频分支类似
#########################################
class VideoBranch(nn.Module):
    def __init__(self, input_dim=DEFAULT_VISUAL_FEAT_DIM, gcn_channels=64, gru_hidden=32, num_gru_layers=1, 
                 bidirectional=True, dropout=0.1, transformer_layers=1):
        super(VideoBranch, self).__init__()
        self.gcn = StackedGCN(input_dim, gcn_channels, num_layers=2, dropout=dropout)
        self.bigru = nn.GRU(input_size=gcn_channels, hidden_size=gru_hidden, num_layers=num_gru_layers, 
                            batch_first=True, bidirectional=bidirectional)
        self.gru_output_dim = gru_hidden * 2 if bidirectional else gru_hidden
        self.attention = EnhancedAttention(self.gru_output_dim, num_heads=4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.gru_output_dim, nhead=4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

    def forward(self, x):
        B, T, _ = x.size()
        device = x.device
        A = torch.eye(T, device=device).unsqueeze(0).repeat(B, 1, 1)
        gcn_out = self.gcn(x, A)
        gru_out, _ = self.bigru(gcn_out)
        gru_out = self.transformer_encoder(gru_out)
        video_feat = self.attention(gru_out)
        return video_feat

#########################################
# 局部两两融合
#########################################
class PairwiseFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, fusion_dim=FUSION_DIM, dropout=0.1):
        super(PairwiseFusion, self).__init__()
        self.proj1 = nn.Sequential(
            nn.Linear(input_dim1, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj2 = nn.Sequential(
            nn.Linear(input_dim2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.attention = EnhancedAttention(fusion_dim, num_heads=4)
    
    def forward(self, feat1, feat2):
        x1 = self.proj1(feat1).unsqueeze(1)
        x2 = self.proj2(feat2).unsqueeze(1)
        x = torch.cat([x1, x2], dim=1)
        x = self.transformer(x)
        fused = self.attention(x)
        return fused

#########################################
# 两两模态融合模型（多任务：回归与二分类）
#########################################
class PairwiseFusionNet(nn.Module):
    def __init__(self,
                 audio_input_dim=TOTAL_AUDIO_FEAT_DIM,
                 audio_hidden_dim=32,
                 text_hidden_dim=768,  
                 max_text_length=50,
                 gcn_hidden=256,
                 tfidf_dim=TFIDF_DIM,
                 fusion_dim=FUSION_DIM,
                 dropout=0.1):
        super(PairwiseFusionNet, self).__init__()
        self.audio_branch = AudioBranch(input_dim=audio_input_dim, gru_hidden=audio_hidden_dim, transformer_layers=1)
        self.text_branch = TextBranch(max_length=max_text_length, gcn_hidden=gcn_hidden, tfidf_dim=tfidf_dim, dropout=dropout, transformer_layers=2)
        # 将文本分支输出进行投影
        self.text_proj = nn.Linear(gcn_hidden, text_hidden_dim)
        self.video_branch = VideoBranch(input_dim=DEFAULT_VISUAL_FEAT_DIM, gru_hidden=32, transformer_layers=1)
        
        self.score_audio = nn.Linear(1, SCORE_FEAT_DIM)
        self.score_text = nn.Linear(1, SCORE_FEAT_DIM)
        self.score_video = nn.Linear(1, SCORE_FEAT_DIM)
        
        # 定义三对两两融合模块
        self.pairwise_ta = PairwiseFusion(self.audio_branch.gru_output_dim + SCORE_FEAT_DIM, text_hidden_dim + SCORE_FEAT_DIM, fusion_dim=fusion_dim, dropout=dropout)
        self.pairwise_tv = PairwiseFusion(text_hidden_dim + SCORE_FEAT_DIM, self.video_branch.gru_output_dim + SCORE_FEAT_DIM, fusion_dim=fusion_dim, dropout=dropout)
        self.pairwise_av = PairwiseFusion(self.audio_branch.gru_output_dim + SCORE_FEAT_DIM, self.video_branch.gru_output_dim + SCORE_FEAT_DIM, fusion_dim=fusion_dim, dropout=dropout)
        # 最终将三个融合结果拼接后输出最终预测：分别增加回归与分类两个头
        self.final_fc = nn.Linear(fusion_dim * 3, 1)      # 回归输出
        self.final_cls = nn.Linear(fusion_dim * 3, 1)     # 分类输出（输出 logits，用于 BCEWithLogitsLoss）
    
    def forward(self, audio, text_list, tfidf, visual, score):
        audio_feat = self.audio_branch(audio)
        text_feat = self.text_branch(text_list, tfidf)
        text_proj = self.text_proj(text_feat)
        video_feat = self.video_branch(visual)
        
        score = score.unsqueeze(-1)
        score_audio_feat = self.score_audio(score)
        score_text_feat = self.score_text(score)
        score_video_feat = self.score_video(score)
        
        audio_feat_cat = torch.cat([audio_feat, score_audio_feat], dim=1)
        text_feat_cat = torch.cat([text_proj, score_text_feat], dim=1)
        video_feat_cat = torch.cat([video_feat, score_video_feat], dim=1)
        
        fused_ta = self.pairwise_ta(audio_feat_cat, text_feat_cat)
        fused_tv = self.pairwise_tv(text_feat_cat, video_feat_cat)
        fused_av = self.pairwise_av(audio_feat_cat, video_feat_cat)
        
        fused_all = torch.cat([fused_ta, fused_tv, fused_av], dim=1)
        reg_output = self.final_fc(fused_all)
        cls_output = self.final_cls(fused_all)
        return reg_output, cls_output

#########################################
# 模型评估（多任务：回归任务 + 二分类任务）
#########################################
def evaluate_model(model, dataloader, device):
    model.eval()
    all_reg_preds, all_scores = [], []
    all_cls_preds, all_cls_labels = [], []
    with torch.no_grad():
        for sample in dataloader:
            text_list = sample["text"]
            tfidf = sample["tfidf"].to(device)
            audio = sample["audio"].to(device)
            visual = sample["visual"].to(device)
            scores = sample["score"].to(device)
            cls_labels = sample["PHQ_Binary"].to(device)
            reg_output, cls_output = model(audio, text_list, tfidf, visual, scores)
            reg_output = reg_output.squeeze(-1)
            cls_output = cls_output.squeeze(-1)
            all_reg_preds.extend(reg_output.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            # 二分类预测：先经过 sigmoid 再以 0.5 为阈值
            cls_preds = (torch.sigmoid(cls_output) > 0.5).int().cpu().numpy()
            all_cls_preds.extend(cls_preds)
            all_cls_labels.extend(cls_labels.cpu().numpy())
    mae = mean_absolute_error(all_scores, all_reg_preds)
    rmse = np.sqrt(mean_squared_error(all_scores, all_reg_preds))
    print(f"\n=== Regression Results ===\nMAE: {mae:.3f}, RMSE: {rmse:.3f}")
    print("\n=== Classification Report ===")
    report = classification_report(all_cls_labels, all_cls_preds, digits=3)
    print(report)

#########################################
# 模型训练（多任务：回归任务 + 二分类任务）
#########################################
def train_model(model, train_loader, dev_loader, device, epochs, lr, patience=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    epochs_without_improve = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for sample in train_loader:
            optimizer.zero_grad()
            text_list = sample["text"]
            tfidf = sample["tfidf"].to(device)
            audio = sample["audio"].to(device)
            visual = sample["visual"].to(device)
            scores = sample["score"].to(device)
            cls_labels = sample["PHQ_Binary"].to(device)
            reg_output, cls_output = model(audio, text_list, tfidf, visual, scores)
            reg_output = reg_output.squeeze(-1)
            cls_output = cls_output.squeeze(-1)
            loss_reg = reg_criterion(reg_output, scores)
            loss_cls = cls_criterion(cls_output, cls_labels)
            loss = loss_reg + loss_cls  # 两个任务的损失简单相加，可根据需要调整权重
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for sample in dev_loader:
                text_list = sample["text"]
                tfidf = sample["tfidf"].to(device)
                audio = sample["audio"].to(device)
                visual = sample["visual"].to(device)
                scores = sample["score"].to(device)
                cls_labels = sample["PHQ_Binary"].to(device)
                reg_output, cls_output = model(audio, text_list, tfidf, visual, scores)
                reg_output = reg_output.squeeze(-1)
                cls_output = cls_output.squeeze(-1)
                loss_reg = reg_criterion(reg_output, scores)
                loss_cls = cls_criterion(cls_output, cls_labels)
                loss = loss_reg + loss_cls
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{epochs} - Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Validation loss improved, saving model.")
        else:
            epochs_without_improve += 1
            print(f"No improvement for {epochs_without_improve} epoch(s).")
            if epochs_without_improve >= patience:
                print("Early stopping triggered!")
                break

        print("Development set evaluation:")
        evaluate_model(model, dev_loader, device)

#########################################
# 辅助函数
#########################################
def load_all_texts(root_dir, data_df):
    texts = []
    for _, row in data_df.iterrows():
        participant_id = str(row["Participant_ID"])
        p_dir = os.path.join(root_dir, f"{participant_id}_P")
        transcript_path = os.path.join(p_dir, f"{participant_id}_Transcript.csv")
        if os.path.isfile(transcript_path):
            try:
                df = pd.read_csv(transcript_path)
                if "Text" in df.columns:
                    text = " ".join(df["Text"].astype(str).tolist())
                else:
                    text = ""
            except Exception as e:
                print(f"读取 {transcript_path} 出错: {e}")
                text = ""
        else:
            text = ""
        texts.append(text)
    return texts

#########################################
# 主流程：三折交叉验证 + 独立评估
#########################################
def main():
    parser = argparse.ArgumentParser(description="Pairwise MultiModal Fusion for Depression Recognition (Multi-Task: Regression + Classification)")
    parser.add_argument("--root_dir", type=str, default="/root/Desktop/我的网盘/depression/e_", help="各 participant 数据所在目录")
    parser.add_argument("--train_csv", type=str, default="/root/Desktop/我的网盘/depression/e_/labels/train_split.csv", help="训练集 CSV 文件")
    parser.add_argument("--dev_csv", type=str, default="/root/Desktop/我的网盘/depression/e_/labels/dev_split.csv", help="验证集 CSV 文件")
    parser.add_argument("--test_csv", type=str, default="/root/Desktop/我的网盘/depression/e_/labels/test_split.csv", help="测试集 CSV 文件")
    parser.add_argument("--checkpoint", type=str, default="", help="预训练模型 checkpoint (.pth) 文件路径")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_text_length", type=int, default=50, help="文本最大长度")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    train_df = pd.read_csv(args.train_csv)
    train_texts = load_all_texts(args.root_dir, train_df)
    tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_DIM)
    tfidf_vectorizer.fit(train_texts)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold = 1
    for train_index, val_index in kf.split(train_df):
        print(f"\n===== Fold {fold} =====")
        train_fold_df = train_df.iloc[train_index].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_index].reset_index(drop=True)
        train_dataset = UnzippedDataset(root_dir=args.root_dir, data_df=train_fold_df, tfidf_vectorizer=tfidf_vectorizer, augment=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataset = UnzippedDataset(root_dir=args.root_dir, data_df=val_fold_df, tfidf_vectorizer=tfidf_vectorizer)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        model = PairwiseFusionNet(
            audio_input_dim=TOTAL_AUDIO_FEAT_DIM,
            audio_hidden_dim=32,
            text_hidden_dim=768,
            max_text_length=args.max_text_length,
            gcn_hidden=256,
            tfidf_dim=TFIDF_DIM,
            fusion_dim=FUSION_DIM,
            dropout=1e-1
        ).to(device)

        if args.checkpoint and os.path.isfile(args.checkpoint):
            print("加载 checkpoint:", args.checkpoint)
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        else:
            print("未提供或未找到指定的 checkpoint 文件，将从头开始训练。")
        
        train_model(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, patience=3)
        print(f"\n=== Fold {fold} 在验证集上的评估 ===")
        evaluate_model(model, val_loader, device)
        fold += 1

    print("\n===== 独立验证集评估 =====")
    dev_df = pd.read_csv(args.dev_csv)
    dev_dataset = UnzippedDataset(root_dir=args.root_dir, data_df=dev_df, tfidf_vectorizer=tfidf_vectorizer)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    evaluate_model(model, dev_loader, device)

    print("\n===== 测试集评估 =====")
    test_df = pd.read_csv(args.test_csv)
    test_dataset = UnzippedDataset(root_dir=args.root_dir, data_df=test_df, tfidf_vectorizer=tfidf_vectorizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
