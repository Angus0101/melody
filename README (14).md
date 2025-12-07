# Stage 2 – Pure DL (CNN) for PTM Site Prediction

本專案實作一個 **Pure Deep Learning** 架構，使用字元層級的一維卷積神經網路（1D CNN）來預測蛋白質序列是否具有三種翻譯後修飾（PTMs）：

- **GLU**：S-glutathionylation  
- **NIT**：S-nitrosylation  
- **PAL**：S-palmitoylation  

每一種 PTM 都被視為一個 **獨立的二元分類任務**（有修飾 / 無修飾），分別訓練一個 CNN 模型並進行評估。  

---

## 1. Dataset

### 1.1 檔案內容

使用的資料檔為 `train_esm2_meta.csv`，每一列代表一個候選位點，欄位包含：

- `ID`：樣本編號  
- `Sequence`：以該位點為中心之 **31 個胺基酸**序列片段  
- `S-glutathionylation`：是否為 GLU 位點（0 / 1）  
- `S-nitrosylation`：是否為 NIT 位點（0 / 1）  
- `S-palmitoylation`：是否為 PAL 位點（0 / 1）

總樣本數約 **89,010 筆**，其中每一種 PTM 的正樣本遠少於負樣本，是一個典型的 **高度不平衡資料集**。  
在訓練過程中，會將資料以 **8:2** 切分為訓練集與測試集，並使用 `stratify` 參數保持正負比例一致。

---

## 2. Model Architecture

本專案對三種 PTM 任務皆使用相同的一維 CNN 架構，差別僅在於標籤不同。模型輸入為長度固定的胺基酸序列（長度 = 31），首先將字元轉換為索引，再送入嵌入層與卷積網路。

### 2.1 序列編碼

1. 收集所有出現在 `Sequence` 中的胺基酸字元，建立字典 `char2idx`  
2. 每一個字元映射為整數索引（從 1 開始，0 保留作 padding）  
3. 將每條序列轉換為長度 31 的索引序列，不足補 0、過長則截斷  

結果為形狀 `(N, 31)` 的整數張量 `X_ids`。

### 2.2 CNN 架構

對每一個任務，建立一個 Binary CNN 模型：

1. **Embedding 層**
   - `input_dim = vocab_size`
   - `output_dim = 32`
   - 將索引序列轉為長度 31、每步 32 維的連續向量表示  

2. **一維卷積與池化**
   - `Conv1D(64, kernel_size=3, padding="same", activation="relu")`
   - `MaxPooling1D(pool_size=2)`
   - `Conv1D(128, kernel_size=3, padding="same", activation="relu")`
   - `GlobalMaxPooling1D()`

   這些卷積濾鏡會在序列上滑動，學習與 PTM 相關的局部序列 pattern（類似 motif 掃描）。

3. **全連接層與 Dropout**
   - `Dense(128, activation="relu")`
   - `Dropout(0.5)`

   用來整合卷積特徵並降低過度擬合。

4. **輸出層**
   - `Dense(1, activation="sigmoid")`

   輸出一個介於 0–1 的機率，代表該序列是否具有該種 PTM。

5. **訓練設定**
   - Loss：`binary_crossentropy`
   - Optimizer：`Adam`
   - Metrics：`accuracy`
   - Epochs：20
   - Batch size：128

---

## 3. Handling Class Imbalance

由於三種 PTM 的正樣本比例明顯偏低，若直接訓練，模型容易學成「全部猜 0」。  
因此在每一個任務中，使用 `sklearn.utils.class_weight.compute_class_weight` 估計 **類別權重（class weight）**，並在訓練時傳入 Keras 的 `class_weight` 參數：

- 正樣本（label = 1）給予較大的權重  
- 負樣本（label = 0）給予較小的權重  

這樣可以放大少數類別在 loss 中的貢獻，強迫模型更重視 PTM 位點的預測。

---

## 4. Training & Evaluation

### 4.1 訓練流程

對於每一個 PTM 標籤（GLU / NIT / PAL），重複以下步驟：

1. 讀入 `Sequence` 與對應的單一 PTM 標籤（0/1）  
2. 以 8:2 切分 train / test，並計算 class weight  
3. 建立一個 CNN 模型並訓練 20 個 epochs  
4. 將訓練好的模型以 Keras 格式儲存：
   - `cnn_GLU.keras`
   - `cnn_NIT.keras`
   - `cnn_PAL.keras`
5. 在測試集上評估：
   - 取輸出機率 `p`，以閾值 0.5 轉為預測標籤  
   - 計算 Accuracy、ROC AUC  
   - 對「正類（PTM=1）」計算 Precision / Recall / F1-score（`average="binary"`）  
   - 產生完整的 `classification_report` 以檢視正、負兩類的表現

---

## 5. Results

### 5.1 Overall Performance Summary

下表彙整三個任務在測試集上的表現。  
注意：Precision、Recall、F1 Score 皆為「正類（有該 PTM）」的指標。

| 任務 (Task) | Accuracy |   AUC  | Precision | Recall | F1 Score |
| :---        | :------: | :----: | :-------: | :----: | :------: |
| **GLU**     | 0.7561   | 0.7466 | 0.1220    | 0.5781 | 0.2015   |
| **NIT**     | 0.7299   | 0.7802 | 0.2673    | 0.6739 | 0.3826   |
| **PAL**     | 0.8775   | 0.7962 | 0.1556    | 0.5227 | 0.2409   |

### 5.2 Per-class Metrics

以下節錄三個任務的 `classification_report`（precision / recall / F1 / support）作為輔助說明：

#### 5.2.1 S-glutathionylation (GLU)

- Accuracy：**0.7561**  
- AUC：**0.7467**  

| Label    | Precision | Recall | F1-score | Support |
| :------- | :-------: | :----: | :------: | ------: |
| negative | 0.97      | 0.77   | 0.86     | 16854   |
| positive | 0.12      | 0.58   | 0.20     |   948   |

- Macro avg F1 ≈ 0.53  
- Weighted avg F1 ≈ 0.82  

#### 5.2.2 S-nitrosylation (NIT)

- Accuracy：**0.7299**  
- AUC：**0.7802**  

| Label    | Precision | Recall | F1-score | Support |
| :------- | :-------: | :----: | :------: | ------: |
| negative | 0.94      | 0.74   | 0.83     | 15588   |
| positive | 0.27      | 0.67   | 0.38     |  2214   |

- Macro avg F1 ≈ 0.60  
- Weighted avg F1 ≈ 0.77  

#### 5.2.3 S-palmitoylation (PAL)

- Accuracy：**0.8775**  
- AUC：**0.7962**  

| Label    | Precision | Recall | F1-score | Support |
| :------- | :-------: | :----: | :------: | ------: |
| negative | 0.98      | 0.89   | 0.93     | 17140   |
| positive | 0.16      | 0.52   | 0.24     |   662   |

- Macro avg F1 ≈ 0.59  
- Weighted avg F1 ≈ 0.91  

---

## 6. Discussion

### 6.1 整體觀察

1. **AUC 介於 ~0.75–0.80 之間**  
   三個任務的 AUC 皆顯示模型對正負樣本具有一定的區分能力，優於隨機猜測。

2. **Accuracy 偏高但受不平衡影響**  
   特別是在 PAL 任務中，Accuracy 接近 0.88，但正樣本比例極低，因此高準確率主要來自對大量負樣本的正確預測，無法單獨代表對 PTM 位點的預測品質。

3. **正類的 Precision 偏低、Recall 中等偏高**  
   - GLU：precision ~0.12，recall ~0.58  
   - NIT：precision ~0.27，recall ~0.67  
   - PAL：precision ~0.16，recall ~0.52  

   顯示模型傾向「多抓一些疑似正例」以提高敏感度（recall），但也付出相當多的假陽性成本。

4. **NIT 任務表現相對最佳**  
   在三種 PTM 中，NIT 擁有較高的正類 F1-score（~0.38）與 AUC（~0.78），代表在這個任務上 CNN 能較有效地學到與 NIT 相關的序列模式。

### 6.2 可能原因與改進方向

- **資料高度不平衡** 是造成 precision 偏低的主要原因之一。即使已使用 class weight，模型仍然較容易產生假陽性。  
- 目前僅使用 **序列本身** 作為輸入，尚未結合結構、演化資訊或 ESM2 這類預訓練蛋白語言模型特徵，可能限制了模型能捕捉的生物學訊號。  
- 未對 decision threshold 進行調整；若將門檻從 0.5 上調，可能提升 precision、降低 recall，適合需要「高可信度預測位點」的情境。

後續可嘗試：

1. 結合預訓練蛋白語言模型（例如 ESM2）的 embedding 作為輸入特徵  
2. 採用更進階的架構（如 multi-branch CNN、BiLSTM、attention 機制等）  
3. 使用 focal loss、oversampling 或合成少數樣本等方式進一步處理不平衡  
4. 針對特定應用調整 decision threshold，取得 precision / recall 的最佳折衷。

---

## 7. How to Run

1. 將 `train_esm2_meta.csv` 與 `train.py` 放在同一資料夾下  
2. 建議建立虛擬環境並安裝所需套件（TensorFlow、scikit-learn、pandas、numpy 等）  
3. 執行：

```bash
python train.py
```

程式會依序訓練 GLU、NIT、PAL 三個任務，訓練完成後：

- 輸出各任務的 `classification_report` 與整體效能比較表  
- 在當前目錄生成三個模型檔案：
  - `cnn_GLU.keras`
  - `cnn_NIT.keras`
  - `cnn_PAL.keras`

這些 `.keras` 檔案可於後續載入，用於對新序列進行 PTM 位點預測。
