# 商談即提案 / Minutes to Proposal Agent

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Poetry](https://img.shields.io/badge/Poetry-60A5FA?style=flat&logo=poetry&logoColor=white)](https://python-poetry.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)](https://platform.openai.com/)

商談即提案は、商談議事録や RFP を入力すると、提案書のたたき台、概算見積、PoC 計画、次回確認事項までを一気通貫で整理するプリセールス支援 AI エージェントです。

完全自動化だけを目指すのではなく、AI が初稿を出し、人が途中で確認や補足を入れながら前に進められることを重視しています。曖昧な要件は `Ask / Assume` に切り分け、顧客に確認すべきことと、仮定で進められることを分けて扱います。

## デモ動画

[![デモ動画を YouTube で開く](docs/youtube.png)](https://youtu.be/u3jsmUtGNIs)

## システム概要

このシステムは、商談後の初動を速めるための GUI ツールです。議事録や RFP を入力すると、AI エージェントが要件整理を行い、提案準備に必要な成果物をまとめて生成します。

主な流れは次のとおりです。

1. 議事録または RFP をファイルでアップロードする
2. 要件、論点、不明点、前提条件を抽出する
3. `Ask / Assume` を切り分ける
4. 必要に応じて担当者が補足情報を入力する
5. 提案資料、概算見積、WBS ベースの PoC 計画、確認事項を生成する
6. 提案内容に応じた簡易デモアプリと trace を確認する

## 機能概要

- 議事録テキストまたは RFP のファイル入力（`.txt` / `.md`）
- 要件整理と前提条件の抽出
- `Ask / Assume` の自動判定
- 人手による補足情報の追加入力
- ローカルナレッジを参照した提案資料 HTML 生成
- WBS、概算見積、次回ヒアリング質問の生成
- 提案内容に応じた Streamlit の簡易デモアプリ生成
- trace / 実行ログの表示

## このシステムの特徴

- 曖昧な要件をそのまま流さず、`Ask / Assume` で整理できる
- 提案書だけでなく、見積、PoC 計画、確認事項まで連続して出力できる
- GUI で途中の判断や補足を入れられるため、実務フローに載せやすい
- trace により、何を参照してどう判断したかを追いやすい

## 対象範囲

### このリポジトリが主に対象としている用途

- 商談後の初期提案準備の高速化
- 議事録や RFP からの論点整理と不足情報の抽出
- 提案書初稿、概算見積、WBS ベースの PoC 計画のたたき台作成
- 次回打ち合わせで確認すべき事項の整理
- 提案内容に沿った簡易デモイメージの生成

### 現時点で、次の用途を主対象としていません

- 契約確定に使う正式見積の作成
- 本番導入前提の詳細要件定義書や詳細設計書の完全自動生成
- 会議音声ファイルからの文字起こし
- 公開情報の自動調査や外部サイト巡回
- 複数提案案の自動比較
- PowerPoint、Excel、PDF など複数形式への資料出力

## 起動方法

### 前提条件

- Python 3.12
- Poetry

### 1. リポジトリを取得する

```bash
git clone https://github.com/atanaka-dev/minutes-to-proposal minutes_to_proposal
cd minutes_to_proposal
```

### 2. 依存関係をインストールする

```bash
poetry install
```

### 3. 必要に応じて環境変数を設定する

OpenAI API を利用します。`.env.example` を参考に `.env` を作成し、`OPENAI_API_KEY` を設定してください。

```bash
cp .env.example .env
```

補足:

- 抽出 / 生成 / 批評 / 復旧のモデルは `.env` の `OPENAI_*_MODEL` で用途別に切り替えられます
- API キーが未設定の場合はローカル処理にフォールバックします（機能が制限されます）

### 4. アプリを起動する

```bash
poetry run streamlit run app/main.py
```

起動後、表示されたローカル URL をブラウザで開いて利用します。

## CLI 実行（Claude Code / SKILL 向け）

GUI を開かずに同じエージェント処理を実行する場合は、次を使います。

```bash
poetry run python scripts/run_presales_agent.py --input demo_inputs/meeting_note_form_screening.md
```

JSON 形式の実行サマリ（`run_id`, `run_dir`, `artifacts` など）が標準出力されます。

主なオプション:

- `--output-json <path>`: 実行サマリ JSON を保存
- `--extract-model <model>`: 抽出モデル上書き
- `--generate-model <model>`: 生成モデル上書き
- `--planner-model <model>`: 判断モデル上書き

## 使い方

1. `議事録 / RFP ファイル` に `.txt` または `.md` をアップロードする
2. 必要に応じてサイドバーでモデルを選ぶ
3. `実行` を押す
4. 生成された `Ask / Assume`、提案資料、見積、WBS、デモ案、trace（`Log` タブ）を確認する
5. 確認カードを調整して `再提案` することもできる

## 生成される成果物

代表的な出力は次のとおりです。

- `artifacts/<client>/proposal.html`
- `artifacts/<client>/demo_app/app.py`

提案資料には、主に次のような内容が含まれます。

- 課題整理
- 要件仮説
- 制約条件
- `Ask`
- `Assume`
- 参照ナレッジ
- 提案ソリューション
- WBS
- 概算見積
- 主要リスク
- 次回ヒアリング質問
- 簡易デモ方針

## リポジトリ構成

```text
.
├── app/          # Streamlit UI
├── src/
│   ├── agent/    # エージェント実行ループ
│   ├── services/ # 抽出、ナレッジ参照、提案生成、デモ生成
│   └── tools/    # エージェントが使う明示的なツール群
├── templates/    # 提案資料テンプレート、標準 WBS
├── knowledge/    # 単価表、過去案件サマリ、リスクカタログ
├── .claude/skills/minutes-to-proposal/
│   └── SKILLS.md
├── demo_inputs/  # テスト・フィクスチャ用のサンプル入力
└── artifacts/    # 生成結果の出力先
```

## 内部設計ドキュメント

内部の設計方針や処理フローを確認したい場合は、次を参照してください。

- `docs/agent_design_flow.md`

## 安全設計

- 実顧客名、実単価、機密議事録は公開成果物に含めない前提です
- trace には生の議事録全文や PII ではなく、判断理由の要約を残します

## トラブルシューティング

### `poetry install` が失敗する

- Python 3.12 が使われているか確認する（`python3.12 --version` など）。
- Poetry が別バージョンの Python を掴んでいる場合は、`poetry env use python3.12` で環境の Python を合わせてから再度 `poetry install` を試す。

### アプリが起動しない / モジュールが見つからない

- リポジトリのルート（`pyproject.toml` があるディレクトリ）でコマンドを実行しているか確認する。
- `poetry run streamlit run app/main.py` で起動する（グローバルの `streamlit` だけを叩いていないか）。

### `Address already in use`（ポートが使用中）

- 別の Streamlit やプロセスが同じポートを使っている可能性があります。例として別ポートで起動する:

```bash
poetry run streamlit run app/main.py --server.port 8502
```

### OpenAI API まわり

- `.env` は**アプリを起動するカレントディレクトリ**（通常はリポジトリルート）に置き、`OPENAI_API_KEY` が設定されているか確認する。
- 認証エラーやレート制限のメッセージが出る場合は、キーの有効性・利用上限・モデル名（`.env` の `OPENAI_*_MODEL`）を確認する。
- キー未設定時はローカル処理にフォールバックし、**機能が制限**されます（「起動方法」の補足参照）。

### 生成結果が期待と違う / 途中で止まったように見える

- `Log` タブの trace でエラーやフォールバックの有無を確認する。
- 入力ファイルが `.txt` / `.md` であること、サイズや文字コード（UTF-8 推奨）を確認する。

## ライセンス

ライセンスは**未確定**です。利用・改変・再配布の条件は、確定次第この README およびリポジトリルートのライセンス表記で示す予定です。
