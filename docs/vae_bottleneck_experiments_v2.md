# VAEボトルネック可視化：実装タスク共有（UWF / SDXL Custom VAE）

このドキュメントは **「UWF画像に対して SDXL の VAE がボトルネックになっているか」**を定量・定性的に可視化するための **診断（diagnostic）実験**を、実装担当に共有する目的でまとめたものです。

- 下流タスク：UWF画像の **マルチラベル分類（ConvNeXt-Large）**
- 問題意識：SDXLで生成・変換すると細部が壊れ、下流性能が低下。特に **VAE潜在にノイズが入ったときの耐性（拡散過程の耐性）**が疑わしい。
- ゴール：VAEの再構成／潜在ノイズ→再構成が、どの程度 **診断に重要な特徴**を壊しているかを可視化し、**VAE改良（ch=4/8/16、学習損失など）の比較**ができる状態にする。

> 注意  
> - 本ドキュメントでいう「再構成入力での評価」は **最終性能の評価（実運用）ではなく、VAEボトルネック診断**です。  
> - **最終のスコアは必ず real test（実画像）で評価**すること。  
> - ただし、診断のために **val上で「入力変換版（再構成・ノイズ再構成）」を作って評価**するのは一般的なロバスト性/圧縮妥当性検証の一種です。

---

## 0. 既存コード（project.zip）に合わせた前提

プロジェクト構成（抜粋）：

- 再構成（VAE）
  - `src/scripts/reconstruct_from_config.py`
  - config例：`configs/vae/reconstruct/recon_pretrained_val_4ch.yaml`
- 教師分類器（ConvNeXt-Large）
  - 学習：`src/scripts/train_teacher_from_config.py`
  - 評価：`src/scripts/evaluate_teacher_from_config.py`
  - config例：`configs/classifier/train/teacher_convnextl_1024_20labels.yaml`  
           `configs/classifier/eval/eval_teacher_proposed.yaml`
- Dataset：`src/sdxl_custom_vae/datasets/image_dataset.py`（`MultiLabelMedicalDataset`）
- ラベルスキーマ：`src/sdxl_custom_vae/labels/schema.py`（`load_label_schema`）

---

## 1. 実験全体の設計方針（重要）

### 1.1 実験で答える問い
1) **再構成（t=0）で情報が落ちるか？**  
   - VAE圧縮/復元だけで、分類に必要な特徴が消えるなら、VAEは明確にボトルネック。
2) **潜在ノイズ（t>0）でどれだけ壊れるか？**  
   - 拡散過程を模したノイズで、どのt（ノイズ強度）から急激に壊れるかを曲線で可視化。
3) **どのラベルが壊れやすいか？**  
   - UWFでは周辺部・微小所見が特に壊れやすい可能性があるため、ラベル別解析は必須。

### 1.2 “診断評価”と“最終評価”を混同しない
- 診断評価（本ドキュメントの中心）：  
  - **val（または診断用subset）** で、入力を `x / recon(x) / recon_noise(x,t)` に置き換えて評価し、VAE由来の情報落ちを直接観測する。
- 最終評価（augmentation効果の検証）：  
  - trainに合成を混ぜて学習し、**testは実画像**で性能評価する。  
  - ※本ドキュメントは主に診断評価を扱う（必要なら後述の拡張実験も実装）。

---

## 2. 実装してほしい「診断実験」一覧（最小セット）

### E0: 教師分類器のベースライン評価（realのみ）
- 目的：以降の診断の比較基準を確定
- 実行：既存 `evaluate_teacher_from_config.py` を使用
- 出力：val/test の AUROC/mAP など、しきい値（macro-F1最大）を保存

---

### E1: VAE再構成（t=0）の診断評価
- 目的：VAE圧縮/復元だけで下流特徴が落ちるか（VAEボトルネックの一次判定）
- 入力：
  - orig: 実画像 `x`
  - recon: `x̂0 = decode( encode(x) )`
- 評価対象：val（推奨）
- 出力：
  - (a) **GTに対する分類性能**（AUROC/mAP, label別も）
  - (b) **orig vs recon の予測差分**（下記の “一致指標” 参照）
  - (c) **破綻サンプルの可視化**（top-K：side-by-side＋差分）

---

### E2: 潜在ノイズ→再構成（t-sweep）の診断評価（最重要）
- 目的：拡散過程に近いノイズ下で、どの程度 “意味” が壊れるかを **t-curve** で可視化
- 入力：
  - `z = encode(x)`（※評価では原則 `posterior.mode()` で決定論にする。後述）
  - `z_t = α(t) * z + σ(t) * ε`
  - `x̂_t = decode(z_t)`
- tの候補：まずは 6〜10点でOK（例：`[0, 50, 100, 200, 300, 500, 700, 900]` など）
  - 可能なら **SNR等間隔**になるように選ぶ（小ノイズ側を少し細かめ）
- 出力：
  - (a) **t-curve（AUROC/mAP vs t）**（CSVでOK）
  - (b) **予測一致度 vs t**（後述の一致指標：KL/|Δp|/label flip など）
  - (c) **ラベル別崩壊点**（どのラベルが早く落ちるか）

---

### E3: VAE候補比較（ch=4/8/16 × 手法）
- 目的：VAE改善案を “ボトルネック指標” で比較できる状態にする
- 比較軸（まずは最小でOK）：
  - latent channel：`C ∈ {4, 8, 16}`
  - VAE種：`Base-VAE（recon+KL）` vs `Feat-VAE（recon+KL + 特徴一致）`  
- 出力：
  - 各VAEについて E1/E2 のまとめ（JSON + CSV）
  - 1枚の summary 表（例：t=0 の mAP低下、tでの崩壊点、label flip率など）

---

## 3. 診断で必ず出したい指標（実装仕様）

### 3.1 “GTに対する性能”（通常指標）
- macro/micro AUROC
- macro mAP
- label別 AUROC/mAP（必須）
- しきい値固定の bin 指標（macro-F1など）
  - しきい値は **real val** で決めて固定（`evaluate_teacher_from_config.py` と同じ方針）

### 3.2 “orig vs 変換後” の一致指標（VAEボトルネックの可視化に効く）
同じ教師モデル `f` に対して：
- `p_orig = sigmoid(f(x))`
- `p_recon = sigmoid(f(x̂))`

以下を集計（macro + label別）：

1) **平均絶対差**  
   - `mean(|p_recon - p_orig|)`  
   - label別にも出す（どの所見が壊れているかが見える）

2) **KL divergence（任意）**  
   - `KL(p_orig || p_recon)`（数値安定化注意：epsilon）

3) **Label flip rate（閾値固定）**  
   - real valで決めた global threshold `τ` を用いて  
     - `y_hat_orig = p_orig >= τ`  
     - `y_hat_recon = p_recon >= τ`
   - `flip = mean(y_hat_orig != y_hat_recon)`（label別・全体）

4) **Embedding cosine similarity（任意だが強い）**  
   - ConvNeXt の最終embedding `g(x)` を取り、`cos(g(x), g(x̂))` を出す  
   - （可能なら stage別特徴でcosを出すと、どのスケールで崩れているか分かる）

> 実装上の注意：  
> - この “一致指標” は **GT無しでも診断可能**で、VAE劣化の切り分けに強い。  
> - ただし最終的な結論はGTに対する性能（AUROC/mAP）も併記する。

---

## 4. 実装タスク：新規スクリプト（推奨）

### 4.1 追加してほしいスクリプト案
`src/scripts/diagnose_vae_bottleneck_from_config.py`（新規）

- 既存の `evaluate_teacher_from_config.py` を拡張してもOKだが、
  - VAE encode/decode
  - 潜在ノイズ sweep
  - top-K可視化
  を入れると肥大化しやすいので、新規を推奨。

### 4.2 スクリプトの処理フロー（擬似コード）
```python
# 1) load config
# 2) load label_schema -> classes, label_groups, mask
# 3) load dataset (val) with TWO transforms:
#    - teacher_tf: normalize etc (既存 build_teacher_transforms(train=False))
#    - vae_tf: [-1,1] (既存 reconstruct_from_config.py の build_vae_transform)
# 4) load teacher model + checkpoint
# 5) load VAE (pretrained or custom)
# 6) for each batch:
#      x_teacher, y, paths = dataset_teacher_tf
#      x_vae,     _, _     = dataset_vae_tf  (同一順序で取れるようにDataset設計 or 2回読み)
#      z = vae.encode(x_vae).latent_dist.mode() * scaling_factor
#      # (a) t=0
#      recon0 = vae.decode(z / scaling_factor).sample
#      # teacher forward for orig/recon0
#      p_orig  = sigmoid(teacher(x_teacher))
#      p_recon0 = sigmoid(teacher( teacher_preprocess(recon0) ))
#      accumulate metrics
#      # (b) t-sweep
#      for t in timesteps:
#          z_t = alpha[t] * z + sigma[t] * eps
#          recon_t = decode(z_t)
#          p_t = sigmoid(teacher(recon_t_preprocessed))
#          accumulate t-curve metrics
#      track top-K worst samples (by |Δp| or flip count), save images
# 7) write outputs:
#      - summary.json
#      - per_label.csv
#      - t_curve.csv
#      - worst_samples/ (orig|recon|diff)
```

### 4.3 重要：再構成は「posterior.sample」ではなく「posterior.mode」を基本に
既存 `reconstruct_from_config.py` は
```python
latents = posterior.latent_dist.sample()
```
になっているが、**診断実験では mode（平均）を推奨**。

- 理由：sampleだと、VAEのボトルネックなのか “サンプルノイズ” なのかが混ざる。
- 診断の基本：  
  - `z = posterior.mode() * scaling_factor`（決定論）  
  - その上で、別途 `z_t = αz + σε` の **明示的なノイズ**で壊れ方を見る（E2）。

> 例外：  
> - “生成時の揺らぎ” も見たい場合に `sample` 版も追加して良い（ただし main は mode）。

---

## 5. ノイズスケジュール（SDXL模倣）の実装方針

### 方針A（推奨）：diffusers の scheduler を利用
SDXLの学習時と同型の forward process を作れるため、最も自然。

- 例：`DDPMScheduler` あるいは訓練設定に合う scheduler を初期化し、
  - `add_noise(z, noise, timesteps)` を使う
  - `timesteps` を sweep（t値の集合）

> 実装担当へ：使用schedulerは環境依存があるため、configで切替可能にするのが安全。

### 方針B（簡易）：α(t), σ(t) を自前で用意（SNR基準）
- まずは比較の相対評価が目的なので、以下でも十分：
  - `z_t = sqrt(1-β_t) * z + sqrt(β_t) * ε`
  - `β_t` を0→大まで単調増加させる（線形/コサインなど）
- ただしSDXL “完全再現” ではないため、論文では注意書きが必要。

---

## 6. 出力ファイル仕様（実装担当向け）

各 experiment run（VAEごと）で以下を出力：

```
outputs/diagnostics/vae_bottleneck/<exp_name>/
  config_used.yaml
  summary.json
  per_label.csv
  t_curve.csv
  worst_samples/
    <id>_orig.png
    <id>_recon_t0.png
    <id>_recon_tXXX.png   # 必要なら
    <id>_diff.png         # |orig - recon| の可視化（正規化してOK）
```

### summary.json（例）
- `teacher_metrics_real`: real入力のAUROC/mAP
- `teacher_metrics_recon_t0`: recon入力のAUROC/mAP
- `agreement_recon_t0`: mean|Δp|, flip率, cos類
- `t_curve`: tごとの要約（最小限でOK。詳細はcsv）

### per_label.csv（例）
- class名
- AUROC/mAP（real, recon0, recon_t...）
- mean|Δp|
- flip率

### t_curve.csv（例）
- timestep t
- macro_mAP / macro_AUROC
- mean|Δp|
- flip率
- mean_cosine（任意）

---

## 7. 実行例（既存スクリプトとの接続）

### E0: 教師の評価
```bash
python -m src.scripts.evaluate_teacher_from_config \
  --config configs/classifier/eval/eval_teacher_proposed.yaml
```

### E1/E2/E3: VAEボトルネック診断（新規）
```bash
python -m src.scripts.diagnose_vae_bottleneck_from_config \
  --config configs/vae/diagnostic/diag_pretrained_4ch.yaml
```

---

## 8. どのように「VAEがボトルネック」と判断するか（解釈ガイド）

- **t=0（再構成だけ）**で、real入力に対して mAP/AUROC が有意に落ちる  
  → VAE圧縮で既に情報が落ちている可能性が高い
- **t-sweepで急激に崩壊する**（小さなtでflip率増、|Δp|増）  
  → 拡散過程耐性が低い（まさに問題の症状）
- **特定ラベルで顕著に崩壊**  
  → 微小所見/周辺部の保持に失敗している可能性が高い（UWFで重要）

---

## 9. （質問への回答）再構成実験での VAE学習はどうするべきか？

診断実験（E1/E2）を公平にするには、VAE学習の扱いを明確化する。

### 9.1 推奨：2段階のベースライン（Pretrained固定 → UWF fine-tune）
UWFはSDXL事前学習分布と大きく異なるため、**「固定VAEでの診断」だけだと “分布ギャップ” と “拡散過程耐性” が混ざって見える**可能性がある。  
そこで、以下の2段階ベースラインで切り分けるのを推奨。

- (A) **Pretrained SDXL VAE（固定）**で E1/E2 を *1回だけ* 回す（任意だが推奨）
  - 目的：UWFとの分布ギャップ（特に t=0 再構成でどれだけ落ちるか）を把握する “温度計”
- (B) **UWF実データで fine-tune した VAE（元の損失）**を、以降の **主ベースライン**として E1/E2/E3 を回す
  - 目的：分布ギャップを潰した上で、「潜在ノイズ耐性（拡散過程耐性）」の差だけを評価する
- (C) 改善案（特徴一致/ノイズ下特徴一致/分類ロス等）は、(B)と **同一初期値・同一データ・同一ステップ数**で学習し比較する  
  - 例：`Base-VAE（recon+KL）` vs `Feat-VAE（recon+KL + 特徴一致）` を *同条件* で fine-tune して比較する

> もし計算資源やスケジュールの都合で(A)を省く場合でも、(B)→(C) の比較は必須。

### 9.2 カスタムVAEを比較する場合の学習方針（リーク回避）
- 原則：**VAEは train split のみで学習**し、valで診断・早期停止。testは触らない。
  - VAEは教師なしでも、testを使うと “評価が甘くなる” 可能性があるため（論文の健全性）。
- 学習の早期停止指標として、
  - 単純な recon loss だけでなく、**E1/E2の診断指標（特に t=0 と小tの一致指標）**を推奨。

### 9.3 再構成時の推奨設定
- 診断では `posterior.mode()`（決定論）を基本
- dtype/精度：
  - 診断では `fp32` が安全（既存configもfp32）
  - GPUメモリ次第で `fp16` も可（ただし差が出る場合は注意）

---

## 10. 追加（任意）：augmentation効果の最終評価（real testのみ）

診断で有望なVAEが見えた後、最終目的（trainに混ぜたときの性能）を評価する。

- trainで **オンザフライ混合（確率的に x / recon / recon_t を混ぜる）** を推奨  
  - 「静的に追加してデータ数が増える」比較の汚れを避ける
- testは必ず実画像
- まず混合率は2点で十分：`0.25 / 0.5`

---

## Appendix A: 診断用config（例：たたき台）

`configs/vae/diagnostic/diag_pretrained_4ch.yaml`（例）

```yaml
experiment_name: "diag_vae_pretrained_4ch_val"

data:
  root: "../data/uwf/multilabel/MedicalCheckup/splitted"
  split_filename: "default_split.yaml"
  split: "val"
  label_schema_file: "configs/labels/proposed_schema.yaml"
  mean: [0.485, 0.456, 0.406]
  std:  [0.229, 0.224, 0.225]

image:
  center_crop_size: 3072
  image_size: 1024

teacher:
  checkpoint: "outputs/checkpoints/teacher/teacher_convnextl_proposed/best.pt"
  arch: "convnext_large"
  gpu_ids: [0]

vae:
  repo_id: "stabilityai/sdxl-vae"
  dtype: "fp32"
  device: "cuda"
  gpu_ids: [0]
  posterior: "mode"   # "mode" | "sample"

noise:
  scheduler: "ddpm"   # "ddpm" | "custom"
  timesteps: [0, 50, 100, 200, 300, 500, 700, 900]
  seed: 123

threshold:
  mode: "global_from_real_val"  # real valで決めた値を使う
  # or: "search_on_real_val" でこのスクリプトが決めても良い

output:
  root_dir: "outputs/diagnostics/vae_bottleneck"
  save_topk: 64
  save_side_by_side: true
  save_diff: true
```

---

## Appendix B: 実装メモ（落とし穴）

- Datasetの読み込みで、teacher用transformとVAE用transformを両方使う必要あり。
  - 対応案：
    1) 同一pathを2回開く（teacher_tfとvae_tfで別Datasetを作る）
    2) Datasetがpathを返すので、pathベースで再ロードする
- 再構成画像をteacherに入れる前に、teacherの正規化（ImageNet mean/std）を適用すること。
- t-sweepは計算が重いので、まず `num_samples` を少なくして動作確認 → 全valに拡張。
- top-K抽出は “バッチごとにヒープ更新” でOK（全サンプル保持しない）。

