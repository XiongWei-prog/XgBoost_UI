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
# 百度千帆 SDK
import qianfan

# ═════ 1 资产加载 & SDK 初始化 ════════════════════════════════════
load_dotenv(find_dotenv())
# 硬编码或环境变量读取 AK/SK
chat_comp = qianfan.ChatCompletion(
    model="ERNIE-Bot",
    ak="FY5WvTZX8dBALfDlToRrRpAO",
    sk="FJBICkqjKTJ2EQE2GIyzpgmEdYdC4Gh1"
)

# 加载模型（改为 XGBoost）
root = pathlib.Path(__file__).parent
model  = joblib.load(root / "xgboost.pkl")
scaler = joblib.load(root / "scaler.pkl")

# 读取并准备数据
df_full = pd.read_csv(root / "newest_data.csv").drop(columns=["Unnamed: 0"], errors="ignore")
explainer = shap.TreeExplainer(model)
print("SHAP expected value (logit):", explainer.expected_value)
# 对齐特征列
candidate_cols = (list(getattr(explainer, "feature_names", []) or [])
                  or list(getattr(model, "feature_names_in_", []) or []))
if not candidate_cols:
    candidate_cols = list(df_full.columns.drop(["MA","Stkcd","year"]))
FEATURE_COLS = candidate_cols
for c in FEATURE_COLS:
    if c not in df_full.columns:
        df_full[c] = 0.0
years_all = sorted(df_full["year"].unique(), reverse=True)

# 提取 SHAP 向量的帮助函数
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

# ═════ 2 UI 布局 ════════════════════════════════════════════════
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_text("code", "股票代码", placeholder="600519"),
        ui.input_select("year", "年份", {str(y): str(y) for y in years_all}, selected=str(years_all[0])),
        ui.output_text_verbatim("warn")
    ),
    ui.row(
        ui.column(6,
            ui.card(ui.card_header("Watchlist — Top10 并购概率"), ui.output_table("watch_tbl"))
        ),
        ui.column(6,
            ui.card(ui.card_header("全局特征重要度 (Top-20)"), output_widget("global_imp_plot"))
        )
    ),
    ui.row(
        ui.column(6,
            ui.card(ui.card_header("并购概率 & SHAP 解释"), ui.output_text("pred_text"), output_widget("local_shap"))
        ),
        ui.column(6,
            ui.card(ui.card_header("🧠 AI 深度解读"), ui.output_text("llm_analysis"))
        )
    )
)

# ═════ 3 Server 逻辑 ═══════════════════════════════════════════════
def server(input, output, session):
    
    @reactive.calc
    def df_year():
        return df_full[df_full["year"] == int(input.year())].copy()

    @reactive.calc
    def watch_df():
        df = df_year().copy()
        df["Prob"] = model.predict_proba(scaler.transform(df[FEATURE_COLS]))[:, 1]
        last3 = df_full[df_full["year"].between(int(input.year())-2, int(input.year()))]
        df["MA_3yr"] = (last3.groupby("Stkcd")["MA"].sum().reindex(df["Stkcd"]).fillna(0).astype(int).values)
        return df[["Stkcd","Prob","MA_3yr"]].sort_values("Prob", ascending=False).head(10).reset_index(drop=True)

    @output
    @render.table
    def watch_tbl():
        df = watch_df().copy()
        df["Prob%"] = (df["Prob"]*100).round(2).astype(str) + " %"
        return df[["Stkcd","Prob%","MA_3yr"]]
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
        return int(input.year()), input.code().strip(), prob, shap_vec

    @output
    @render.text
    def warn():
        if not input.code().strip():
            return "请输入股票代码"
        if current_pred() is None:
            return "⚠ 该股票在所选年份无数据"
        return ""

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
    def local_shap():
        res = current_pred()
        if not res:
            return None
        _, _, _, shap_vec = res
        contrib = pd.Series(shap_vec, index=FEATURE_COLS).abs().sort_values(ascending=False).head(15)
        fig = px.bar(x=contrib.values*np.sign(shap_vec[:len(contrib)]), y=contrib.index, orientation="h",
                     color=contrib.values*np.sign(shap_vec[:len(contrib)]), color_continuous_scale="RdBu",
                     labels={"x":"对 logit 的贡献","y":"特征"}, title=f"SHAP Top-{len(contrib)}")
        fig.update_layout(coloraxis_showscale=False, height=430, yaxis_categoryorder="total ascending")
        return fig

    @output
    @render_widget
    def global_imp_plot():
        global_imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).reset_index().rename(columns={"index":"Feature",0:"Importance"})
        fig = px.bar(global_imp.head(20), x="Importance", y="Feature", orientation="h", title="全局特征重要度 (Top-20)")
        fig.update_layout(height=430, yaxis_categoryorder="total ascending")
        return fig

    @output
    @render.text
    def llm_analysis():
        res = current_pred()
        if not res:
            return ""
        year, code, prob, shap_vec = res
        contrib5 = pd.Series(shap_vec, index=FEATURE_COLS).abs().sort_values(ascending=False).head(5)
        top5 = "; ".join(f"{f}={v:.4f}" for f,v in contrib5.items())
        glob5s = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(5)
        glob_str = "; ".join(f"{f}={v:.4f}" for f,v in glob5s.items())
        prompt = (f"你是一名并购分析专家，评估股票{code}在{year}年的并购可能性。" +
                  f"并购概率:{prob:.2%}；SHAP前5:{top5}；全局前5:{glob_str}。" +
                  "请输出简明洞察和建议。")
        resp = chat_comp.do(messages=[{"role":"user","content":prompt}], top_p=0.8, temperature=0.4)
        return resp.get("result", "")

# ═════ 4 实例化 & 运行 ════════════════════════════════════════════
app = App(app_ui, server)





