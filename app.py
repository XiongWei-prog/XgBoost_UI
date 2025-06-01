# ---------------------------  app.py  ---------------------------
# 依赖：
#   shiny ≥1.4, shinywidgets ≥0.2, plotly ≥5.8, shap ≥0.44,
#   pandas, numpy, joblib, python-dotenv, qianfan
# ----------------------------------------------------------------
from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget
import plotly.express as px, plotly.graph_objects as go
import pandas as pd, numpy as np, shap, joblib, pathlib, os
from dotenv import load_dotenv, find_dotenv
import markdown
import qianfan
import base64
import tempfile
import shap
import akshare as ak

load_dotenv(find_dotenv())
chat_comp = qianfan.ChatCompletion(
    model="ERNIE-Bot",
    ak="FY5WvTZX8dBALfDlToRrRpAO",
    sk="FJBICkqjKTJ2EQE2GIyzpgmEdYdC4Gh1"
)

root = pathlib.Path(__file__).parent
model  = joblib.load(root / "xgboost.pkl")
scaler = joblib.load(root / "scaler.pkl")

# 读取数据
df_train = pd.read_csv(root / "data_for_training.csv").drop(columns=["Unnamed: 0"], errors="ignore")
df_pred  = pd.read_csv(root / "data_2025.csv").drop(columns=["Unnamed: 0", "MA"], errors="ignore")
df_train["Stkcd"] = df_train["Stkcd"].astype(str).str.zfill(6)
df_pred["Stkcd"]  = df_pred["Stkcd"].astype(str).str.zfill(6)
explainer = shap.TreeExplainer(model)
print("SHAP expected value (logit):", explainer.expected_value)

candidate_cols = (list(getattr(explainer, "feature_names", []) or []) or 
                   list(getattr(model, "feature_names_in_", []) or []))
if not candidate_cols:
    candidate_cols = list(df_train.columns.drop(["Stkcd", "year", "MA"]))
FEATURE_COLS = candidate_cols
for c in FEATURE_COLS:
    if c not in df_pred.columns:
        df_pred[c] = 0.0
years_all = ["2025"]

def extract_shap_vector(shap_raw, n_feat):
    arr = np.array(shap_raw).squeeze()
    if arr.ndim == 2:
        if arr.shape[0] == 2 and arr.shape[1] == n_feat:
            return arr[1]
        if arr.shape[1] == 2 and arr.shape[0] == n_feat:
            return arr[:, 1]
        return arr.flatten()[:n_feat]
    if arr.ndim == 1:
        return arr[:n_feat]
    return arr.flatten()[:n_feat]

def get_stock_name_map():
    stock_info = ak.stock_info_a_code_name()
    name_map = dict(zip(stock_info["code"].str.zfill(6), stock_info["name"]))
    return name_map

# UI
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_text("code", "股票代码", placeholder="600519"),
        ui.input_select("year", "年份", {str(y): str(y) for y in years_all}, selected=str(years_all[0])),
        ui.output_ui("warn")
    ),
    ui.tags.style("""
        body {
            background-color: #0F0F0F;
            color: white;
        }
        .card, .sidebar, .form-control, .shiny-input-container {
            background-color: #1A1A1A !important;
            color: white !important;
            border: none !important;
        }
        input[type="text"],
        select {
            background-color: #01B075 !important;
            color: white !important;
            font-weight: normal;
            border: none !important;
        }
        input::placeholder {
            color: white !important;
            opacity: 0.7;
        }
        option {
            background-color: #1A1A1A;
            color: white;
        }
        .table, .table th, .table td {
            color: white !important;
            background-color: #111 !important;
        }
        .shiny-output-error {
            color: #FF4C4C;
        }
        .card-header {
            color: #01B075 !important;
            font-weight: bold;
            font-size: 18px !important;
        }
        .custom-warning {
            background-color: #01B075;
            color:white;
            padding: 8px;
            border-radius: 4px;
            font-weight: bold;
            text-align: center;
            margin-top: 10px;
            animation: fadeIn 0.5s ease-out;
            font-size: 14px;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    """),
    ui.row(
        ui.column(6,
            ui.div(  # ⭐ 新增一个 div 包裹并加样式
                ui.card(
                    ui.card_header("Watchlist — Top10 并购概率"),
                    ui.output_table("watch_tbl")
                ),
                style="height: 500px;"  # ⭐ 这里改高度
            )
        ),
        ui.column(6,
            ui.card(ui.card_header("全局特征重要度 (Top-20)"), output_widget("global_imp_plot"))
        )
    ),
    ui.row(
        ui.column(12,
            ui.card(ui.card_header("SHAP 力图解释"), ui.output_ui("local_shap"))
        ),
        ui.column(12,
            ui.card(ui.card_header("🧠 AI 深度解读"), ui.output_ui("llm_analysis"))
        )
    )
)

# Server

def server(input, output, session):
    stock_name_map = get_stock_name_map()
    def load_annual_report(code, year=2024):
        """
        传入股票代码，找到并返回对应的年报内容（如果存在）
        """
        # 6位代码补全
        code = str(code).zfill(6)
        # 目录遍历匹配
        reports_dir = root  # 假设年报放在 ./annual_reports 文件夹
        pattern = f"{code}_{year}_*.txt"
        for f in reports_dir.glob(pattern):
            with open(f, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()
                # ⭐ 可选：截取前1000字避免 prompt 过长
                return content[:1000] + "..." if len(content) > 1000 else content
        return "未找到年报"

    @reactive.calc
    def df_year():
        # 直接返回 df_pred，因为只有2025年
        return df_pred.copy()

    @reactive.calc
    def watch_df():
        df = df_year().copy()

        # ⭐ 确定模型需要的特征列
        model_features = getattr(model, "feature_names_in_", FEATURE_COLS)

        # ⭐ 缺失列补0（保证模型特征完整性）
        for c in model_features:
            if c not in df.columns:
                df[c] = 0.0

        # ⭐ 只保留模型需要的特征列（顺序对齐）
        df_model = df[model_features].copy()

        # ⭐ 替换 inf 为 NaN
        df_model = df_model.replace([np.inf, -np.inf], np.nan)

        # ⭐ 删除含有 NaN（含 inf）的整行
        df_model = df_model.dropna()

        # ⭐ 同步删除主 DataFrame 中对应行
        df = df.loc[df_model.index].copy()

        # ⭐ 做预测
        x_input = scaler.transform(df_model)
        df["Prob"] = model.predict_proba(x_input)[:, 1]

        # ⭐ 计算前三年 MA
        last3 = df_train[df_train["year"].between(int(input.year())-2, int(input.year()))]
        ma_3yr = last3.groupby("Stkcd")["MA"].sum()
        df["MA_3yr"] = df["Stkcd"].map(ma_3yr).fillna(0).astype(int)

        return df[["Stkcd", "Prob", "MA_3yr"]].sort_values("Prob", ascending=False).head(10).reset_index(drop=True)

    @output
    @render.table
    def watch_tbl():
        df = watch_df().copy()
        df["Name"] = df["Stkcd"].map(stock_name_map).fillna("未知")
        df["Prob%"] = (df["Prob"]*100).round(2).astype(str) + " %"
        return df[["Stkcd","Name","Prob%","MA_3yr"]]

    @reactive.calc
    def current_pred():
        row = df_year()[df_year()["Stkcd"].astype(str) == input.code().strip()]
        if row.empty:
            return None
        x = scaler.transform(row[FEATURE_COLS])
        prob = model.predict_proba(x)[0,1]
        shap_raw = explainer.shap_values(x)
        n_feat = len(FEATURE_COLS)
        shap_vec = extract_shap_vector(shap_raw, n_feat)
        code = input.code().strip()
        name = stock_name_map.get(code, "未知")
        return int(input.year()), code, name, prob, shap_vec


    @output
    @render.ui
    def warn():
        if not input.code().strip():
            return ui.HTML('<div class="custom-warning">请输入股票代码</div>')
        elif current_pred() is None:
            return ui.HTML('<div class="custom-warning">⚠ 该股票在所选年份无数据</div>')
        else:
            return None  # 返回None或空HTML

    @output
    @render.text
    def pred_text():
        res = current_pred()
        if not res:
            return ""
        year, code, prob, _ = res
        tag = "⚠ 高并购概率" if prob>=0.5 else "√ 并购概率低"
        return f"{year} 年预测并购概率：{prob:.2%} → {tag}"

    @output
    @render_widget
    def global_imp_plot():
        # 使用历史数据（df_train）做特征重要度分析
        df_imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(10).reset_index()
        df_imp.columns = ["Feature", "Importance"]

        bar = go.Bar(
            x=df_imp["Importance"],
            y=df_imp["Feature"],
            orientation="h",
            marker=dict(color="#01B075", line=dict(width=0)),
            width=0.4,
            hoverinfo="x+y",
            textposition="auto",
            showlegend=False
        )

        dot_x = df_imp["Importance"] * 1.005
        dot_y = df_imp["Feature"]
        dots = go.Scatter(
            x=dot_x,
            y=dot_y,
            mode="markers+text",
            marker=dict(size=14, color="#01B075"),
            text=[f"{v:.2f}" for v in df_imp["Importance"]],
            textposition="middle right",
            textfont=dict(size=10),
            hoverinfo="skip",
            showlegend=False
        )

        fig = go.Figure(data=[bar, dots])
        fig.update_layout(
            height=458,
            bargap=0.005,
            title=None,
            margin=dict(l=60, r=60, t=30, b=30),
            plot_bgcolor="#0F0F0F",
            paper_bgcolor="#0F0F0F",
            font=dict(color="white"),
            xaxis=dict(title="Importance", showgrid=False, zeroline=False, range=[0, 0.35], color="white"),
            yaxis=dict(autorange="reversed", color="white"),
            coloraxis_showscale=False
        )
        return fig


    @output
    @render.ui
    def local_shap():
        res = current_pred()
        if not res:
            return ui.HTML("<em>暂无数据</em>")

        year, code, name, prob, shap_vec = res  # ⭐ 5个值

        # 获取当前股票的原始数据行
        row = df_year()[df_year()["Stkcd"].astype(str) == code]
        if row.empty:
            return ui.HTML("<em>该股票数据不存在</em>")

        x_row = row[FEATURE_COLS].iloc[0]  # 提取特征值（Series）
        shap_values = np.array(shap_vec)   # 转成 numpy 数组
        base_value = explainer.expected_value[1]  # 取正类基础值

        # ⭐ 生成 SHAP 力图（不做概率还原处理）
        force_plot = shap.plots.force(
            base_value,
            shap_values,
            x_row,
            matplotlib=False,
            show=False,
            plot_cmap=["#01B075", "#FB483B"],
            text_rotation=0,
            link="logit"
        )

        # ⭐ 写入临时 HTML 文件，嵌入到 Shiny 界面
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
            shap.save_html(tmp.name, force_plot)
            tmp.flush()
            with open(tmp.name, "r", encoding="utf-8") as f:
                html = f.read()

        return ui.HTML(html)

    @output
    @render.ui
    def llm_analysis():
        res = current_pred()
        if not res:
            return ui.HTML("<em>暂无预测结果</em>")
        year, code, name, prob, shap_vec = res
        contrib5 = pd.Series(shap_vec, index=FEATURE_COLS).abs().sort_values(ascending=False).head(5)
        top5 = "; ".join(f"{f}={v:.4f}" for f, v in contrib5.items())
        glob5s = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(5)
        glob_str = "; ".join(f"{f}={v:.4f}" for f, v in glob5s.items())

        # ⭐ 加载年报内容
        annual_report = load_annual_report(code)

        prompt = (
            f"你是一名并购分析专家，结合你可访问的公开网络资料（如行业研究报告、新闻、财务数据等），"
            f"以及以下年报节选，评估股票 {code}（{name}）在 {year} 年的并购可能性。\n"
            f"并购概率: {prob:.2%}\n"
            f"股票名称: {name}\n"
            f"SHAP前5: {top5}\n"
            f"全局前5: {glob_str}\n"
            f"以下是该股票的 2024 年年报部分内容：\n"
            f"{annual_report}\n"
            f"请结合行业背景、宏观经济环境、以及历史案例，使用 Markdown 格式输出深度分析报告，第一行请写出并购预测是并购还是不并购，并写出概率。"
        )
        resp = chat_comp.do(messages=[{"role": "user", "content": prompt}], top_p=0.8, temperature=0.4)
        markdown_text = resp.get("result", "")
        html = markdown.markdown(markdown_text)
        return ui.HTML(html)

app = App(app_ui, server)