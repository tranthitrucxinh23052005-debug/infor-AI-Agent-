import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

# ============================================================
# CẤU HÌNH TRANG
# ============================================================

st.set_page_config(
    page_title="Phân Tích Tác Động AI – Ngành Công Nghệ Thông Tin",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# BẢNG MÀU & NHÃN TIẾNG VIỆT DÙNG CHUNG
# ============================================================

PALETTE = {
    "primary": "#2563EB",
    "secondary": "#0EA5E9",
    "accent": "#7C3AED",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "muted": "#64748B",
}

# Nhãn tiếng Việt cho các cột gốc (dùng trong tham số labels= của Plotly Express)
VN_LABELS = {
    "Automation Capacity Rating": "Năng lực tự động hóa",
    "Human Agency Scale Rating": "Quyền tự chủ con người",
    "Domain Expertise Requirement": "Yêu cầu chuyên môn",
    "Interpersonal Communication Requirement": "Yêu cầu giao tiếp",
    "Physical Action Requirement": "Yêu cầu thể chất",
    "Involved Uncertainty": "Mức độ bất định",
    "Automation Desire Rating": "Mong muốn tự động hóa",
    "Occupation (O*NET-SOC Title)": "Nghề nghiệp",
    "Occupation Mean Annual Wage": "Lương trung bình (USD/năm)",
    "Occupation Employment": "Quy mô lao động",
    "Importance": "Mức độ quan trọng",
    "Relevance": "Mức độ liên quan",
    "Frequency": "Tần suất",
    "Count": "Số lượng",
    "degree": "Số kết nối",
}

# Nhãn ngắn tiếng Việt cho 6 yếu tố năng lực — dùng cho ma trận tương quan, radar, biểu đồ ảnh hưởng
SHORT_VN = {
    "Automation Capacity Rating": "Tự động hóa",
    "Physical Action Requirement": "Thể chất",
    "Involved Uncertainty": "Bất định",
    "Domain Expertise Requirement": "Chuyên môn",
    "Interpersonal Communication Requirement": "Giao tiếp",
    "Human Agency Scale Rating": "Tự chủ con người",
}

# Nhãn ngắn tiếng Việt cho các cột thuộc bộ dữ liệu "tasks"
SHORT_VN_TASK = {
    "Frequency": "Tần suất",
    "Importance": "Quan trọng",
    "Relevance": "Liên quan",
    "Occupation Mean Annual Wage": "Lương TB",
    "Occupation Employment": "Quy mô LĐ",
}

OCC_COL = "Occupation (O*NET-SOC Title)"

# ============================================================
# HÀM TIỆN ÍCH DÙNG CHUNG
# ============================================================


def style_fig(fig, title=None, height=560):
    """Áp dụng giao diện sáng, hiện đại, dễ đọc cho mọi biểu đồ Plotly trong app."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#1E293B", family="Inter, Segoe UI, sans-serif", size=13),
        height=height,
        margin=dict(t=60 if title else 30, l=10, r=10, b=10),
        legend=dict(bgcolor="rgba(255,255,255,0)"),
    )
    if title:
        fig.update_layout(
            title=dict(text=title, x=0.02, xanchor="left", font=dict(size=17, color="#0F172A"))
        )
    fig.update_xaxes(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0", linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0", linecolor="#CBD5E1")
    return fig


def chart_header(title, caption):
    """In tên biểu đồ + một dòng chú thích giải thích biểu đồ dùng để làm gì."""
    st.subheader(title)
    st.caption(f"💡 {caption}")


def kpi_card(col, value, label, accent, icon=""):
    with col:
        st.markdown(
            f"""<div class="kpi-card" style="--accent:{accent}">
            <div class="kpi-number">{icon} {value}</div>
            <div class="kpi-label">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ============================================================
# GIAO DIỆN SÁNG, HIỆN ĐẠI
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: linear-gradient(180deg, #F4F7FB 0%, #ECF1F9 100%);
    color: #0F172A;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

[data-testid="stSidebar"]{
    background:#FFFFFF;
    border-right: 1px solid #E2E8F0;
}

[data-testid="stHeader"]{
    background: rgba(255,255,255,0);
}

h1, h2, h3 { color:#0F172A !important; font-weight:700 !important; }

.kpi-card{
    background:#FFFFFF;
    border:1px solid #E5E9F2;
    border-left:5px solid var(--accent, #2563EB);
    border-radius:18px;
    padding:22px 18px;
    text-align:center;
    box-shadow:0 4px 14px rgba(15,23,42,0.06);
    transition: transform .15s ease;
    margin-bottom:8px;
}
.kpi-card:hover{ transform: translateY(-3px); }

.kpi-number{
    font-size:30px;
    font-weight:700;
    color:#1D4ED8;
}

.kpi-label{
    color:#64748B;
    font-size:14px;
    margin-top:4px;
}

/* Bo tròn nút bấm và ô nhập để giao diện hiện đại, thân thiện hơn */
.stButton > button, .stDownloadButton > button{
    border-radius:999px !important;
    border:1px solid #2563EB !important;
    padding:0.5rem 1.4rem !important;
}

div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"]{
    border-radius:14px !important;
}

[data-testid="stMetric"]{
    background:#FFFFFF;
    border:1px solid #E5E9F2;
    border-radius:16px;
    padding:14px 10px;
    box-shadow:0 2px 10px rgba(15,23,42,0.05);
}

[data-testid="stExpander"]{
    border-radius:14px;
    border:1px solid #E5E9F2;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LỌC DỮ LIỆU: CHỈ GIỮ NHÓM NGÀNH CNTT / KHMT / KHOA HỌC DỮ LIỆU / HTTT
# ============================================================
# Bộ từ khóa được nhóm theo 4 lĩnh vực để đảm bảo bao quát đầy đủ các nghề
# thuộc nhóm ngành công nghệ: Khoa học máy tính, Công nghệ thông tin,
# Khoa học dữ liệu và Hệ thống thông tin.

IT_KEYWORDS = [
    # Khoa học máy tính & Kỹ thuật phần mềm
    "computer", "software", "programmer", "systems analyst",
    "computer network", "network and computer", "computer hardware",
    "computer occupations", "quality assurance analysts and testers",
    # Công nghệ thông tin & Hệ thống thông tin
    "information security", "information systems", "information technology",
    "information research scientist", "database", "web developer",
    "web and digital interface", "computer support specialist",
    "network support specialist", "network administrator",
    "telecommunications", "cloud", "devops",
    # Khoa học dữ liệu & Phân tích dữ liệu
    "data scientist", "data engineer", "data analyst", "data warehousing",
    "machine learning", "artificial intelligence", "business intelligence",
]


@st.cache_data(show_spinner=False)
def filter_it(df: pd.DataFrame) -> pd.DataFrame:
    """Chỉ giữ lại các dòng thuộc nhóm ngành CNTT / Khoa học máy tính /
    Khoa học dữ liệu / Hệ thống thông tin."""
    if OCC_COL not in df.columns:
        return df
    pattern = "|".join(IT_KEYWORDS)
    mask = df[OCC_COL].astype(str).str.lower().str.contains(pattern, regex=True, na=False)
    return df.loc[mask].reset_index(drop=True)


# ============================================================
# TẢI DỮ LIỆU
# ============================================================


@st.cache_data
def load_data():
    desires = pd.read_csv("domain_worker_desires.csv")
    metadata = pd.read_csv("domain_worker_metadata.csv")
    capability = pd.read_csv("expert_rated_technological_capability.csv")
    tasks = pd.read_csv("task_statement_with_metadata.csv")
    return desires, metadata, capability, tasks


desires, metadata, capability, tasks = load_data()

desires = filter_it(desires)
metadata = filter_it(metadata)
capability = filter_it(capability)
tasks = filter_it(tasks)

# Lưới an toàn: đảm bảo "desires" chỉ còn các Task ID thuộc nhóm ngành CNTT,
# kể cả khi file gốc của "desires" không có cột tên nghề nghiệp.
if "Task ID" in desires.columns and "Task ID" in capability.columns:
    desires = desires[desires["Task ID"].isin(capability["Task ID"])].reset_index(drop=True)

if capability.empty or tasks.empty:
    st.error(
        "⚠️ Không tìm thấy dữ liệu thuộc nhóm ngành Công nghệ thông tin, Khoa học máy tính, "
        "Khoa học dữ liệu hoặc Hệ thống thông tin trong bộ dữ liệu hiện tại. "
        "Vui lòng kiểm tra lại nguồn dữ liệu."
    )
    st.stop()

# ============================================================
# THANH ĐIỀU HƯỚNG (SIDEBAR)
# ============================================================

with st.sidebar:

    st.markdown("## 📊 Phân Tích Tác Động AI")
    st.caption("Hệ thống phân tích tác động của AI đến nhóm ngành Công nghệ thông tin")

    PAGES = [
        "📈 Tổng Quan",
        "🌍 Toàn Cảnh Tự Động Hóa",
        "🧠 Khoảng Cách Niềm Tin",
        "📋 Phân Tích Chi Tiết",
        "🗂️ Phân Vùng Tác Động",
        "🎯 Đánh Giá Năng Lực Cá Nhân",
        "📉 Dự Báo Xu Hướng",
        "🔬 Phân Tích Thống Kê Chuyên Sâu",
        "🔗 Mạng Lưới Nghề Nghiệp",
    ]

    page = st.pills(
        "Điều hướng",
        PAGES,
        default=PAGES[0],
        required=True,
        label_visibility="collapsed",
    )

    st.divider()

    n_occ = capability[OCC_COL].nunique() if OCC_COL in capability.columns else 0
    st.caption(
        f"📌 Dữ liệu đã được lọc theo nhóm ngành **Công nghệ thông tin / Khoa học máy tính / "
        f"Khoa học dữ liệu / Hệ thống thông tin**\n\n"
        f"• {n_occ} nghề nghiệp  •  {len(tasks):,} đầu việc"
    )

    if OCC_COL in capability.columns:
        with st.expander("Xem danh sách nghề nghiệp đã lọc"):
            for occ in sorted(capability[OCC_COL].unique()):
                st.markdown(f"- {occ}")

# ============================================================
# TRANG 1 — TỔNG QUAN ĐIỀU HÀNH
# ============================================================

if page == "📈 Tổng Quan":

    st.title("📈 Tổng Quan Điều Hành")
    st.caption(
        "Các chỉ số tổng quan về quy mô dữ liệu, năng lực tự động hóa của AI và mức độ sẵn sàng "
        "của người lao động — giới hạn trong nhóm ngành Công nghệ thông tin, Khoa học máy tính, "
        "Khoa học dữ liệu và Hệ thống thông tin."
    )

    total_tasks = len(tasks)
    total_occupations = tasks[OCC_COL].nunique() if OCC_COL in tasks.columns else 0
    automation_capacity = round(capability["Automation Capacity Rating"].mean(), 2)
    human_agency = round(capability["Human Agency Scale Rating"].mean(), 2)
    automation_desire = round(desires["Automation Desire Rating"].mean(), 2)
    domain_expertise = round(capability["Domain Expertise Requirement"].mean(), 2)

    c1, c2, c3 = st.columns(3)
    kpi_card(c1, f"{total_tasks:,}", "Tổng số đầu việc", PALETTE["primary"], "📋")
    kpi_card(c2, total_occupations, "Số nghề nghiệp", PALETTE["accent"], "💼")
    kpi_card(c3, automation_capacity, "Năng lực tự động hóa (TB)", PALETTE["danger"], "🤖")

    c4, c5, c6 = st.columns(3)
    kpi_card(c4, automation_desire, "Mức sẵn sàng giao việc cho AI (TB)", PALETTE["warning"], "🙋")
    kpi_card(c5, human_agency, "Quyền tự chủ con người (TB)", PALETTE["success"], "🧑‍💻")
    kpi_card(c6, domain_expertise, "Yêu cầu chuyên môn (TB)", PALETTE["secondary"], "🎓")

    st.divider()

    top_occ = tasks[OCC_COL].value_counts().head(15).reset_index()
    top_occ.columns = ["Nghề nghiệp", "Số đầu việc"]

    chart_header(
        "Top 15 Nghề Nghiệp Có Nhiều Đầu Việc Nhất",
        "Các nghề trong nhóm ngành công nghệ đang được khảo sát với số lượng đầu việc (task) "
        "nhiều nhất trong bộ dữ liệu — đây cũng là các nghề được phân tích sâu nhất trong báo cáo.",
    )
    fig = px.bar(
        top_occ, x="Số đầu việc", y="Nghề nghiệp", orientation="h",
        color="Số đầu việc", color_continuous_scale="Blues",
    )
    st.plotly_chart(style_fig(fig, height=650), use_container_width=True)

# ============================================================
# TRANG 2 — TOÀN CẢNH TỰ ĐỘNG HÓA AI
# ============================================================

elif page == "🌍 Toàn Cảnh Tự Động Hóa":

    st.title("🌍 Toàn Cảnh Tự Động Hóa")
    st.caption("Năng lực tự động hóa và quyền tự chủ con người trong nhóm ngành công nghệ.")

    top_capacity = (
        capability.groupby(OCC_COL)["Automation Capacity Rating"]
        .mean().sort_values(ascending=False).head(20).reset_index()
    )
    top_agency = (
        capability.groupby(OCC_COL)["Human Agency Scale Rating"]
        .mean().sort_values(ascending=False).head(20).reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        chart_header(
            "Top 20 Nghề Có Năng Lực Tự Động Hóa Cao Nhất",
            "Các nghề được chuyên gia đánh giá là AI hiện có khả năng kỹ thuật đảm nhiệm cao nhất.",
        )
        fig1 = px.bar(
            top_capacity, x="Automation Capacity Rating", y=OCC_COL, orientation="h",
            color="Automation Capacity Rating", color_continuous_scale="Blues", labels=VN_LABELS,
        )
        st.plotly_chart(style_fig(fig1, height=650), use_container_width=True)

    with col2:
        chart_header(
            "Top 20 Nghề Có Quyền Tự Chủ Con Người Cao Nhất",
            "Các nghề đòi hỏi con người ra quyết định, phán đoán nhiều — khó bị AI thay thế hoàn toàn.",
        )
        fig2 = px.bar(
            top_agency, x="Human Agency Scale Rating", y=OCC_COL, orientation="h",
            color="Human Agency Scale Rating", color_continuous_scale="Greens", labels=VN_LABELS,
        )
        st.plotly_chart(style_fig(fig2, height=650), use_container_width=True)

    st.divider()

    chart_header(
        "Tương Quan Giữa Năng Lực Tự Động Hóa Và Quyền Tự Chủ Con Người",
        "Mỗi điểm là một đầu việc. Màu thể hiện yêu cầu chuyên môn, kích thước thể hiện yêu cầu giao tiếp.",
    )
    fig3 = px.scatter(
        capability, x="Automation Capacity Rating", y="Human Agency Scale Rating",
        color="Domain Expertise Requirement", size="Interpersonal Communication Requirement",
        hover_name=OCC_COL, color_continuous_scale="Viridis", labels=VN_LABELS,
    )
    st.plotly_chart(style_fig(fig3, height=650), use_container_width=True)

    chart_header(
        "Ma Trận Tương Quan Giữa Các Yếu Tố",
        "Giá trị gần 1 (xanh đậm) là tương quan thuận mạnh, gần -1 (đỏ đậm) là tương quan nghịch mạnh.",
    )
    cols = [
        "Automation Capacity Rating", "Physical Action Requirement", "Involved Uncertainty",
        "Domain Expertise Requirement", "Interpersonal Communication Requirement",
        "Human Agency Scale Rating",
    ]
    corr = capability[cols].corr().rename(index=SHORT_VN, columns=SHORT_VN)
    fig4 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(style_fig(fig4, height=600), use_container_width=True)

    chart_header(
        "Hồ Sơ Tổng Thể Của Nhóm Ngành Công Nghệ",
        "Giá trị trung bình của 6 yếu tố trên toàn bộ dữ liệu — dùng làm đường tham chiếu ở mục "
        "Đánh Giá Năng Lực Cá Nhân.",
    )
    values = [capability[c].mean() for c in cols]
    labels_radar = [SHORT_VN[c] for c in cols]
    fig5 = go.Figure()
    fig5.add_trace(go.Scatterpolar(
        r=values, theta=labels_radar, fill="toself",
        line_color=PALETTE["primary"], fillcolor="rgba(37,99,235,0.20)",
    ))
    fig5.update_layout(polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(style_fig(fig5, height=600), use_container_width=True)

# ============================================================
# TRANG 3 — PHÂN TÍCH KHOẢNG CÁCH NIỀM TIN
# ============================================================

elif page == "🧠 Khoảng Cách Niềm Tin":

    st.title("🧠 Phân Tích Khoảng Cách Niềm Tin")
    st.caption(
        "So sánh năng lực kỹ thuật mà AI có thể đảm nhiệm (theo chuyên gia) với mức độ sẵn sàng "
        "giao việc cho AI (theo người lao động), nhằm xác định khoảng cách niềm tin giữa hai bên."
    )

    merged = pd.merge(
        capability[["Task ID", "Automation Capacity Rating"]],
        desires[["Task ID", "Automation Desire Rating"]],
        on="Task ID",
    )

    if merged.empty:
        st.warning(
            "Không có dữ liệu trùng khớp giữa năng lực AI và mức độ sẵn sàng của người lao động "
            "sau khi lọc theo nhóm ngành công nghệ."
        )
    else:
        threshold = st.slider(
            "Ngưỡng quyết định (điểm trên thang 1–5)", 1.0, 5.0, 3.0, 0.1,
            help="Từ ngưỡng này trở lên được coi là 'khả thi / đồng ý', dưới ngưỡng là 'chưa khả thi / từ chối'.",
        )

        merged["AI_Feasible"] = merged["Automation Capacity Rating"] >= threshold
        merged["Worker_Willing"] = merged["Automation Desire Rating"] >= threshold

        tp = len(merged[(merged.AI_Feasible) & (merged.Worker_Willing)])
        fp = len(merged[(~merged.AI_Feasible) & (merged.Worker_Willing)])
        fn = len(merged[(merged.AI_Feasible) & (~merged.Worker_Willing)])
        tn = len(merged[(~merged.AI_Feasible) & (~merged.Worker_Willing)])

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        trust_gap = abs(
            merged["Automation Capacity Rating"].mean() - merged["Automation Desire Rating"].mean()
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Độ chính xác", round(accuracy, 3))
        c2.metric("Precision", round(precision, 3))
        c3.metric("Recall", round(recall, 3))
        c4.metric("Điểm F1", round(f1, 3))
        c5.metric("Khoảng cách niềm tin", round(trust_gap, 3))

        cm = np.array([[tn, fp], [fn, tp]])

        chart_header(
            "Ma Trận Nhầm Lẫn: Năng Lực AI So Với Mức Độ Sẵn Sàng Của Người Lao Động",
            "So khớp đánh giá 'khả thi' của chuyên gia (trục dọc) với mức 'sẵn sàng giao việc' của "
            "người lao động (trục ngang).",
        )
        fig = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Người lao động", y="Chuyên gia"),
            x=["Từ chối", "Chấp nhận"], y=["Từ chối", "Chấp nhận"],
            color_continuous_scale="Blues",
        )
        st.plotly_chart(style_fig(fig, height=550), use_container_width=True)

        chart_header(
            "Phân Bố Năng Lực AI So Với Mức Độ Sẵn Sàng Giao Việc",
            "Đường nét đứt là ngưỡng quyết định đã chọn ở trên. Các góc thể hiện vùng AI làm được "
            "nhưng người lao động chưa muốn giao (hoặc ngược lại).",
        )
        fig2 = px.scatter(
            merged, x="Automation Capacity Rating", y="Automation Desire Rating",
            color="Automation Capacity Rating", color_continuous_scale="Viridis", labels=VN_LABELS,
        )
        fig2.add_vline(x=threshold, line_dash="dash", line_color=PALETTE["muted"])
        fig2.add_hline(y=threshold, line_dash="dash", line_color=PALETTE["muted"])
        st.plotly_chart(style_fig(fig2, height=650), use_container_width=True)

        st.info(
            f"""
### 📌 Nhận định

- **Khoảng cách niềm tin** = {trust_gap:.2f} điểm (trên thang 1–5)
- **Độ chính xác** = {accuracy:.2%}
- **Precision** = {precision:.2%}
- **Recall** = {recall:.2%}
- **Điểm F1** = {f1:.2%}

Khoảng cách giữa năng lực AI thực tế và mức độ sẵn sàng giao việc của người lao động trong
nhóm ngành công nghệ hiện ở mức **{trust_gap:.2f}** điểm.
"""
        )

# ============================================================
# TRANG 4 — PHÂN TÍCH CHI TIẾT ĐẦU VIỆC
# ============================================================

elif page == "📋 Phân Tích Chi Tiết":

    st.title("📋 Phân Tích Chi Tiết Đầu Việc")
    st.caption(
        "Các chỉ số về mức lương, quy mô lao động và mức độ quan trọng của từng đầu việc trong "
        "nhóm ngành công nghệ."
    )

    df = tasks.copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng số đầu việc", len(df))
    c2.metric("Số nghề nghiệp", df[OCC_COL].nunique() if OCC_COL in df.columns else 0)
    c3.metric("Lương trung bình", f"${df['Occupation Mean Annual Wage'].mean():,.0f}")
    c4.metric("Mức độ quan trọng (TB)", round(df["Importance"].mean(), 2))

    chart_header(
        "Phân Bố Mức Độ Quan Trọng (Importance)",
        "Mức độ quan trọng của các đầu việc đối với nghề nghiệp tương ứng, theo thang đánh giá O*NET.",
    )
    fig1 = px.histogram(
        df, x="Importance", nbins=40, color_discrete_sequence=[PALETTE["primary"]], labels=VN_LABELS,
    )
    st.plotly_chart(style_fig(fig1, height=480), use_container_width=True)

    chart_header(
        "Phân Bố Mức Độ Liên Quan (Relevance)",
        "Mức độ liên quan của đầu việc đối với nghề nghiệp — giá trị cao nghĩa là đầu việc gắn "
        "liền với nghề đó.",
    )
    fig2 = px.histogram(
        df, x="Relevance", nbins=40, color_discrete_sequence=[PALETTE["warning"]], labels=VN_LABELS,
    )
    st.plotly_chart(style_fig(fig2, height=480), use_container_width=True)

    chart_header(
        "Phân Bố Lương Theo Năm",
        "Hộp thể hiện khoảng lương phổ biến (trung vị, tứ phân vị); các điểm rời là những nghề có "
        "mức lương bất thường cao hoặc thấp.",
    )
    fig3 = px.box(df, y="Occupation Mean Annual Wage", points="outliers", labels=VN_LABELS)
    fig3.update_traces(marker_color=PALETTE["secondary"], line_color=PALETTE["secondary"])
    st.plotly_chart(style_fig(fig3, height=550), use_container_width=True)

    chart_header(
        "Phân Bố Quy Mô Lao Động",
        "Số lượng người đang làm việc trong từng nghề — cho thấy nghề nào có quy mô thị trường "
        "lao động lớn.",
    )
    fig4 = px.histogram(
        df, x="Occupation Employment", nbins=50, color_discrete_sequence=[PALETTE["success"]], labels=VN_LABELS,
    )
    st.plotly_chart(style_fig(fig4, height=480), use_container_width=True)

    chart_header(
        "Ma Trận Tương Quan",
        "Mối liên hệ giữa tần suất, mức độ quan trọng, mức độ liên quan, lương và quy mô lao động.",
    )
    cols = ["Frequency", "Importance", "Relevance", "Occupation Mean Annual Wage", "Occupation Employment"]
    corr = df[cols].corr().rename(index=SHORT_VN_TASK, columns=SHORT_VN_TASK)
    fig5 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(style_fig(fig5, height=550), use_container_width=True)

    wage_df = (
        df.groupby(OCC_COL)["Occupation Mean Annual Wage"]
        .mean().sort_values(ascending=False).head(20).reset_index()
    )
    chart_header(
        "Top 20 Nghề Có Mức Lương Cao Nhất",
        "Các nghề trong nhóm ngành công nghệ có mức lương trung bình hàng năm cao nhất.",
    )
    fig6 = px.bar(
        wage_df, x="Occupation Mean Annual Wage", y=OCC_COL, orientation="h",
        color="Occupation Mean Annual Wage", color_continuous_scale="Blues", labels=VN_LABELS,
    )
    st.plotly_chart(style_fig(fig6, height=850), use_container_width=True)

    emp_df = (
        df.groupby(OCC_COL)["Occupation Employment"]
        .mean().sort_values(ascending=False).head(20).reset_index()
    )
    chart_header(
        "Top 20 Nghề Có Quy Mô Lao Động Lớn Nhất",
        "Các nghề trong nhóm ngành công nghệ đang thu hút số lượng người lao động đông nhất.",
    )
    fig7 = px.bar(
        emp_df, x="Occupation Employment", y=OCC_COL, orientation="h",
        color="Occupation Employment", color_continuous_scale="Purples", labels=VN_LABELS,
    )
    st.plotly_chart(style_fig(fig7, height=850), use_container_width=True)

# ============================================================
# TRANG 5 — PHÂN VÙNG TÁC ĐỘNG TỰ ĐỘNG HÓA
# ============================================================

elif page == "🗂️ Phân Vùng Tác Động":

    st.title("🗂️ Phân Vùng Tác Động Tự Động Hóa")
    st.caption(
        "Dùng thuật toán phân cụm KMeans để chia các đầu việc trong nhóm ngành công nghệ thành 4 "
        "vùng theo mức độ ảnh hưởng của AI, sau đó giảm chiều dữ liệu bằng PCA để trực quan hóa "
        "trong không gian 3 chiều."
    )

    features = [
        "Automation Capacity Rating", "Physical Action Requirement", "Involved Uncertainty",
        "Domain Expertise Requirement", "Interpersonal Communication Requirement",
        "Human Agency Scale Rating",
    ]
    df = capability.dropna(subset=features).reset_index(drop=True)
    X = df[features]
    k = min(4, len(df))

    if k < 2:
        st.warning("Không đủ dữ liệu để phân cụm sau khi lọc theo nhóm ngành công nghệ.")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(X_scaled)

        cluster_names = {0: "Thế Mạnh Con Người", 1: "Vùng Rủi Ro Cao", 2: "Vùng Kết Hợp", 3: "Vùng Ổn Định"}
        df["Zone"] = df["Cluster"].map(cluster_names)

        n_components = max(1, min(3, len(df) - 1, len(features)))
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(X_scaled)
        df["x"] = coords[:, 0]
        df["y"] = coords[:, 1] if coords.shape[1] > 1 else 0.0
        df["z"] = coords[:, 2] if coords.shape[1] > 2 else 0.0

        zone_colors = {
            "Thế Mạnh Con Người": PALETTE["secondary"],
            "Vùng Rủi Ro Cao": PALETTE["danger"],
            "Vùng Kết Hợp": PALETTE["warning"],
            "Vùng Ổn Định": PALETTE["success"],
        }

        chart_header(
            "Bản Đồ 3D Các Vùng Tác Động AI",
            "Mỗi điểm là một đầu việc, được nén từ 6 chiều dữ liệu xuống 3 chiều (PCA) để dễ "
            "quan sát các vùng phân cụm.",
        )
        fig = px.scatter_3d(
            df, x="x", y="y", z="z", color="Zone", hover_name=OCC_COL,
            color_discrete_map=zone_colors, labels=VN_LABELS,
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(scene=dict(bgcolor="#FFFFFF"))
        st.plotly_chart(style_fig(fig, height=750), use_container_width=True)

        count_df = df["Zone"].value_counts().reset_index()
        count_df.columns = ["Vùng", "Số lượng"]
        chart_header(
            "Tỷ Trọng Các Vùng Tác Động",
            "Tỷ lệ phần trăm số đầu việc rơi vào từng vùng tác động của AI.",
        )
        fig2 = px.pie(
            count_df, names="Vùng", values="Số lượng", hole=0.5,
            color="Vùng", color_discrete_map=zone_colors,
        )
        st.plotly_chart(style_fig(fig2, height=550), use_container_width=True)

        profile = df.groupby("Zone")[features].mean().round(2).rename(columns=SHORT_VN)
        profile.index.name = "Vùng"
        st.subheader("Hồ Sơ Trung Bình Theo Vùng")
        st.caption("💡 Giá trị trung bình của 6 yếu tố trong mỗi vùng — giúp hiểu rõ đặc điểm của từng nhóm.")
        st.dataframe(profile, use_container_width=True)

        danger_ratio = len(df[df.Zone == "Vùng Rủi Ro Cao"]) / len(df)
        st.info(
            f"""
### 📌 Nhận định

🔴 **Vùng Rủi Ro Cao**: {danger_ratio:.1%} đầu việc có rủi ro tự động hóa cao.

🔵 **Thế Mạnh Con Người**: các nghề cần nhiều quyền tự chủ và phán đoán của con người.

🟠 **Vùng Kết Hợp**: AI hỗ trợ mạnh nhưng chưa thể thay thế hoàn toàn con người.

🟢 **Vùng Ổn Định**: cân bằng tốt giữa năng lực AI và vai trò con người.
"""
        )

# ============================================================
# TRANG 6 — ĐÁNH GIÁ NĂNG LỰC CÁ NHÂN
# ============================================================

elif page == "🎯 Đánh Giá Năng Lực Cá Nhân":

    st.title("🎯 Đánh Giá Năng Lực Cá Nhân")
    st.caption(
        "Nhập hồ sơ năng lực của bản thân để so sánh với mức trung bình của nhóm ngành công nghệ, "
        "từ đó nhận gợi ý các nghề nghiệp phù hợp nhất."
    )

    st.markdown("##### Hồ sơ năng lực của bạn (thang điểm 1–5)")
    col1, col2, col3 = st.columns(3)

    with col1:
        automation = st.slider("Mức độ phù hợp tự động hóa", 1.0, 5.0, 3.0)
        physical = st.slider("Yêu cầu thể chất", 1.0, 5.0, 3.0)

    with col2:
        uncertainty = st.slider("Mức độ chấp nhận bất định", 1.0, 5.0, 3.0)
        domain = st.slider("Chuyên môn lĩnh vực", 1.0, 5.0, 3.0)

    with col3:
        communication = st.slider("Kỹ năng giao tiếp", 1.0, 5.0, 3.0)
        agency = st.slider("Mức độ tự chủ", 1.0, 5.0, 3.0)

    user_vector = np.array([automation, physical, uncertainty, domain, communication, agency])

    market_vector = np.array([
        capability["Automation Capacity Rating"].mean(),
        capability["Physical Action Requirement"].mean(),
        capability["Involved Uncertainty"].mean(),
        capability["Domain Expertise Requirement"].mean(),
        capability["Interpersonal Communication Requirement"].mean(),
        capability["Human Agency Scale Rating"].mean(),
    ])

    labels = ["Tự động hóa", "Thể chất", "Bất định", "Chuyên môn", "Giao tiếp", "Tự chủ"]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=market_vector, theta=labels, fill="toself", name="Trung bình ngành",
        line_color=PALETTE["secondary"], fillcolor="rgba(14,165,233,0.20)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=user_vector, theta=labels, fill="toself", name="Hồ sơ của bạn",
        line_color=PALETTE["accent"], fillcolor="rgba(124,58,237,0.20)",
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])))

    chart_header(
        "So Sánh Hồ Sơ Cá Nhân Với Mức Trung Bình Ngành",
        "Vùng càng lệch khỏi nhau ở trục nào thì bạn càng khác biệt so với mặt bằng chung ở yếu tố đó.",
    )
    st.plotly_chart(style_fig(fig, height=600), use_container_width=True)

    distance = np.linalg.norm(user_vector - market_vector)
    score = max(0, 100 - distance * 15)
    st.metric("🎯 Độ phù hợp với mặt bằng ngành", f"{score:.1f}%")

    gap = user_vector - market_vector
    gap_df = pd.DataFrame({"Yếu tố": labels, "Khoảng cách": gap})

    chart_header(
        "Khoảng Cách Theo Từng Yếu Tố",
        "Giá trị dương: bạn đang vượt mức trung bình ngành ở yếu tố đó. Giá trị âm: bạn đang thấp hơn.",
    )
    fig2 = px.bar(gap_df, x="Yếu tố", y="Khoảng cách", color="Khoảng cách", color_continuous_scale="RdYlGn")
    st.plotly_chart(style_fig(fig2, height=450), use_container_width=True)

    feat_cols = [
        "Automation Capacity Rating", "Physical Action Requirement", "Involved Uncertainty",
        "Domain Expertise Requirement", "Interpersonal Communication Requirement",
        "Human Agency Scale Rating",
    ]
    X_knn = capability[feat_cols].dropna()
    k_neighbors = min(10, len(X_knn))

    if k_neighbors >= 1:
        model = NearestNeighbors(n_neighbors=k_neighbors)
        model.fit(X_knn)
        distances, indices = model.kneighbors([user_vector])
        recommend = capability.loc[X_knn.index].iloc[indices[0]]

        chart_header(
            "Nghề Nghiệp Gợi Ý Cho Bạn",
            "Danh sách các nghề có hồ sơ năng lực gần nhất với hồ sơ bạn vừa nhập.",
        )
        st.dataframe(
            recommend[[OCC_COL]].rename(columns={OCC_COL: "Nghề nghiệp gợi ý"}),
            use_container_width=True,
        )

    if score >= 85:
        st.success("✅ Hồ sơ của bạn rất phù hợp với mặt bằng chung của nhóm ngành công nghệ.")
    elif score >= 70:
        st.warning("⚠️ Hồ sơ khá phù hợp, nên nâng cấp thêm một vài kỹ năng.")
    else:
        st.error("🚨 Khoảng cách kỹ năng còn lớn so với mặt bằng ngành — nên có kế hoạch nâng cấp kỹ năng.")

# ============================================================
# TRANG 7 — DỰ BÁO XU HƯỚNG
# ============================================================

elif page == "📉 Dự Báo Xu Hướng":

    st.title("📉 Dự Báo Xu Hướng Ngành Công Nghệ")
    st.caption(
        "Dự báo xu hướng tự động hóa, quyền tự chủ con người và yêu cầu chuyên môn theo từng kịch "
        "bản tăng trưởng AI."
    )

    SCENARIOS = {"Thận trọng": 0.05, "Trung bình": 0.10, "Tích cực": 0.20}
    scenario = st.pills(
        "Chọn kịch bản tăng trưởng AI", list(SCENARIOS.keys()),
        default="Trung bình", required=True,
    )
    growth = SCENARIOS[scenario]

    base_automation = capability["Automation Capacity Rating"].mean()
    base_agency = capability["Human Agency Scale Rating"].mean()
    base_domain = capability["Domain Expertise Requirement"].mean()

    years = [2025, 2026, 2027, 2028]
    automation_curve = [base_automation * (1 + growth * i) for i in range(len(years))]
    agency_curve = [base_agency * (1 + 0.03 * i) for i in range(len(years))]
    domain_curve = [base_domain * (1 + 0.05 * i) for i in range(len(years))]

    forecast = pd.DataFrame({
        "Năm": years,
        "Tự động hóa": automation_curve,
        "Tự chủ con người": agency_curve,
        "Chuyên môn": domain_curve,
    })

    chart_header(
        "Dự Báo Xu Hướng 4 Năm Tới",
        "Mô phỏng đơn giản dựa trên tốc độ tăng trưởng của kịch bản đã chọn — không phải dự báo "
        "chính xác tuyệt đối, chỉ mang tính minh họa.",
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast["Năm"], y=forecast["Tự động hóa"], name="Tự động hóa", line=dict(color=PALETTE["danger"])))
    fig.add_trace(go.Scatter(x=forecast["Năm"], y=forecast["Tự chủ con người"], name="Tự chủ con người", line=dict(color=PALETTE["secondary"])))
    fig.add_trace(go.Scatter(x=forecast["Năm"], y=forecast["Chuyên môn"], name="Chuyên môn", line=dict(color=PALETTE["success"])))
    st.plotly_chart(style_fig(fig, height=600), use_container_width=True)

    future_skill = domain_curve[-1] * 0.4 + agency_curve[-1] * 0.4 + (5 - automation_curve[-1]) * 0.2
    risk_score = automation_curve[-1] / 5 * 100
    opportunity = (agency_curve[-1] + domain_curve[-1]) / 10 * 100
    survival = (100 - risk_score) * 0.4 + opportunity * 0.6

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chỉ số kỹ năng tương lai", round(future_skill, 2))
    c2.metric("Điểm rủi ro", f"{risk_score:.1f}%")
    c3.metric("Điểm cơ hội", f"{opportunity:.1f}%")
    c4.metric("Khả năng duy trì nghề nghiệp", f"{survival:.1f}%")

    simulation = [base_automation * (1 + np.random.normal(growth, 0.03) * 3) for _ in range(1000)]

    chart_header(
        "Mô Phỏng Monte Carlo Về Năng Lực Tự Động Hóa",
        "1.000 kịch bản ngẫu nhiên minh họa mức độ không chắc chắn của dự báo năng lực tự động "
        "hóa trong tương lai.",
    )
    fig2 = px.histogram(simulation, nbins=50, color_discrete_sequence=[PALETTE["secondary"]])
    fig2.update_layout(showlegend=False, xaxis_title="Năng lực tự động hóa mô phỏng", yaxis_title="Số lần")
    st.plotly_chart(style_fig(fig2, height=550), use_container_width=True)

    if survival >= 80:
        st.success(
            "🟢 **Triển vọng nghề nghiệp tốt** — quyền tự chủ con người và chuyên môn cao, "
            "rủi ro tự động hóa thấp."
        )
    elif survival >= 60:
        st.warning("🟡 **Rủi ro trung bình** — nên chủ động nâng cấp kỹ năng trong thời gian tới.")
    else:
        st.error("🔴 **Rủi ro tự động hóa cao** — cần kế hoạch nâng cấp kỹ năng khẩn cấp.")

# ============================================================
# TRANG 8 — PHÂN TÍCH THỐNG KÊ CHUYÊN SÂU
# ============================================================

elif page == "🔬 Phân Tích Thống Kê Chuyên Sâu":

    st.title("🔬 Phân Tích Thống Kê Chuyên Sâu")
    st.caption("Các phân tích thống kê và học máy chuyên sâu trên dữ liệu năng lực của nhóm ngành công nghệ.")

    features = [
        "Automation Capacity Rating", "Physical Action Requirement", "Involved Uncertainty",
        "Domain Expertise Requirement", "Interpersonal Communication Requirement",
        "Human Agency Scale Rating",
    ]
    df = capability.dropna(subset=features).reset_index(drop=True)
    X = df[features]

    chart_header(
        "Ma Trận Tương Quan",
        "Mối liên hệ tuyến tính giữa các yếu tố năng lực — giúp nhận diện yếu tố nào có xu hướng "
        "đi cùng nhau.",
    )
    corr = X.corr().rename(index=SHORT_VN, columns=SHORT_VN)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1)
    st.plotly_chart(style_fig(fig, height=600), use_container_width=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_pca = max(1, min(2, X_scaled.shape[1], len(df) - 1))
    pca = PCA(n_components=n_pca)
    coords = pca.fit_transform(X_scaled)
    df["PC1"] = coords[:, 0]
    df["PC2"] = coords[:, 1] if coords.shape[1] > 1 else 0.0

    chart_header(
        "Không Gian Đặc Trưng PCA",
        "Nén 6 chiều dữ liệu xuống 2 thành phần chính (PC1, PC2) để quan sát cấu trúc tổng thể "
        "của dữ liệu.",
    )
    fig = px.scatter(
        df, x="PC1", y="PC2", color="Automation Capacity Rating",
        hover_name=OCC_COL, color_continuous_scale="Viridis", labels=VN_LABELS,
    )
    st.plotly_chart(style_fig(fig, height=700), use_container_width=True)

    perplexity = min(30, max(5, (len(df) - 1) // 3))
    chart_header(
        "Bản Đồ t-SNE Các Đầu Việc",
        "Kỹ thuật giảm chiều phi tuyến giúp các đầu việc có hồ sơ năng lực tương tự đứng gần nhau "
        "hơn so với PCA, làm nổi bật cấu trúc cụm tự nhiên trong dữ liệu.",
    )
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedding = tsne.fit_transform(X_scaled)
    df["t1"] = embedding[:, 0]
    df["t2"] = embedding[:, 1]
    fig = px.scatter(
        df, x="t1", y="t2", color="Human Agency Scale Rating",
        hover_name=OCC_COL, color_continuous_scale="Turbo", labels=VN_LABELS,
    )
    st.plotly_chart(style_fig(fig, height=750), use_container_width=True)

    y = df["Human Agency Scale Rating"]
    X_rf = df[[
        "Automation Capacity Rating", "Physical Action Requirement", "Involved Uncertainty",
        "Domain Expertise Requirement", "Interpersonal Communication Requirement",
    ]]
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_rf, y)
    importance = pd.DataFrame({
        "Yếu tố": [SHORT_VN.get(c, c) for c in X_rf.columns],
        "Mức độ ảnh hưởng": rf.feature_importances_,
    })

    chart_header(
        "Yếu Tố Ảnh Hưởng Mạnh Nhất Đến Quyền Tự Chủ Con Người",
        "Dùng mô hình Random Forest để ước lượng yếu tố nào quyết định nhiều nhất đến mức độ tự "
        "chủ của con người trong công việc.",
    )
    fig = px.bar(
        importance.sort_values("Mức độ ảnh hưởng"), x="Mức độ ảnh hưởng", y="Yếu tố",
        orientation="h", color="Mức độ ảnh hưởng", color_continuous_scale="Plasma",
    )
    st.plotly_chart(style_fig(fig, height=480), use_container_width=True)

    chart_header(
        "Phát Hiện Đầu Việc Bất Thường",
        "Mô hình Isolation Forest đánh dấu các đầu việc có hồ sơ năng lực khác biệt rõ rệt so với "
        "phần còn lại của dữ liệu.",
    )
    iso = IsolationForest(contamination=0.03, random_state=42)
    df["Outlier"] = iso.fit_predict(X_scaled)
    df["Type"] = np.where(df["Outlier"] == -1, "Bất thường", "Bình thường")
    fig = px.scatter(
        df, x="PC1", y="PC2", color="Type", hover_name=OCC_COL,
        color_discrete_map={"Bình thường": PALETTE["secondary"], "Bất thường": PALETTE["danger"]},
    )
    st.plotly_chart(style_fig(fig, height=700), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Phương sai PCA giải thích", f"{sum(pca.explained_variance_ratio_):.1%}")
    c2.metric("Số bản ghi", len(df))
    c3.metric("Số bất thường", len(df[df.Type == "Bất thường"]))
    c4.metric("Số đặc trưng", len(features))

    st.info(
        """
### 📌 Nhận định thống kê

- Chuyên môn lĩnh vực và kỹ năng giao tiếp là hai yếu tố ảnh hưởng mạnh nhất tới quyền tự chủ con người.
- PCA giải thích phần lớn phương sai của dữ liệu năng lực trong nhóm ngành công nghệ.
- Một số đầu việc tạo thành nhóm bất thường, có khả năng chịu tác động AI khác biệt so với phần còn lại.
- t-SNE cho thấy các đầu việc hình thành nhiều cụm tự nhiên theo hồ sơ năng lực.
"""
    )

# ============================================================
# TRANG 9 — MẠNG LƯỚI NGHỀ NGHIỆP (PHÂN TÍCH ĐỘ TƯƠNG ĐỒNG)
# ============================================================

elif page == "🔗 Mạng Lưới Nghề Nghiệp":

    st.title("🔗 Mạng Lưới Nghề Nghiệp")
    st.caption(
        "Mỗi nghề trong nhóm ngành công nghệ được gộp thành một điểm dữ liệu (giá trị trung bình "
        "của 6 yếu tố), sau đó tính độ tương đồng cosine giữa các nghề để dựng mạng lưới. Hai nghề "
        "có đường nối với nhau nghĩa là hồ sơ năng lực của chúng khá giống nhau — đây cũng là cơ "
        "sở tham khảo cho việc chuyển đổi nghề nghiệp trong tương lai."
    )

    features = [
        "Automation Capacity Rating", "Physical Action Requirement", "Involved Uncertainty",
        "Domain Expertise Requirement", "Interpersonal Communication Requirement",
        "Human Agency Scale Rating",
    ]

    occ_df = capability.dropna(subset=features).groupby(OCC_COL)[features].mean().reset_index()

    if len(occ_df) < 2:
        st.warning("Không đủ số nghề khác nhau trong nhóm ngành công nghệ để dựng mạng lưới tương đồng.")
    else:
        threshold = st.slider(
            "Ngưỡng độ tương đồng để nối hai nghề", 0.80, 0.999, 0.95, 0.001,
            help="Ngưỡng cao hơn → mạng lưới thưa hơn, chỉ giữ lại các cặp nghề rất giống nhau.",
        )

        scaler_net = StandardScaler()
        X_net = scaler_net.fit_transform(occ_df[features])
        sim = cosine_similarity(X_net)

        G = nx.Graph()
        for i, occ in enumerate(occ_df[OCC_COL]):
            G.add_node(i, occupation=occ)

        n = len(occ_df)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] > threshold:
                    G.add_edge(i, j, weight=float(sim[i, j]))

        pos = nx.spring_layout(G, seed=42)

        node_df = pd.DataFrame([
            {
                "occupation": G.nodes[node]["occupation"],
                "x": pos[node][0],
                "y": pos[node][1],
                "degree": G.degree(node),
            }
            for node in G.nodes()
        ])

        chart_header(
            "Bản Đồ Mạng Lưới Nghề Nghiệp",
            "Vị trí các nghề được sắp xếp bằng thuật toán spring layout — nghề nào càng giống "
            "nhiều nghề khác (số kết nối cao) thì điểm càng lớn.",
        )
        fig = px.scatter(
            node_df, x="x", y="y", size="degree", hover_name="occupation",
            color="degree", color_continuous_scale="Blues", labels=VN_LABELS,
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        st.plotly_chart(style_fig(fig, height=650), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số nghề", len(G.nodes()))
        c2.metric("Số kết nối", len(G.edges()))
        c3.metric("Bậc kết nối TB", round(float(np.mean([d for _, d in G.degree()])), 2) if len(G) else 0)
        c4.metric("Mật độ mạng lưới", round(nx.density(G), 3) if len(G) > 1 else 0)

        communities = (
            list(greedy_modularity_communities(G)) if len(G.edges()) > 0
            else [frozenset([node]) for node in G.nodes()]
        )
        community_map = {}
        for i, community in enumerate(communities):
            for node in community:
                community_map[node] = i
        node_df["Cộng đồng"] = [community_map.get(node, -1) for node in G.nodes()]

        chart_header(
            "Cộng Đồng Nghề Nghiệp Tự Nhiên",
            "Thuật toán phát hiện cộng đồng (modularity) nhóm các nghề có liên kết chặt chẽ với "
            "nhau lại thành từng nhóm — mỗi màu là một nhóm nghề có hồ sơ năng lực gần gũi.",
        )
        fig2 = px.scatter(
            node_df, x="x", y="y", color="Cộng đồng", size="degree", hover_name="occupation",
            labels=VN_LABELS,
        )
        fig2.update_xaxes(visible=False)
        fig2.update_yaxes(visible=False)
        st.plotly_chart(style_fig(fig2, height=650), use_container_width=True)

        centrality = nx.degree_centrality(G)
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
        top_df = pd.DataFrame({
            "Nghề nghiệp": [G.nodes[n]["occupation"] for n, _ in top_nodes],
            "Độ trung tâm": [v for _, v in top_nodes],
        })

        chart_header(
            "Nghề Nghiệp Trung Tâm Của Mạng Lưới",
            "Độ trung tâm (centrality) cao cho thấy nghề đó có hồ sơ năng lực 'điển hình', kết nối "
            "với nhiều nghề khác — thường là điểm trung chuyển tốt khi cân nhắc chuyển nghề.",
        )
        fig3 = px.bar(
            top_df, x="Độ trung tâm", y="Nghề nghiệp", orientation="h",
            color="Độ trung tâm", color_continuous_scale="Blues",
        )
        st.plotly_chart(style_fig(fig3, height=700), use_container_width=True)

        st.divider()
        st.markdown("#### 🔁 Gợi ý chuyển đổi nghề nghiệp")
        occupation = st.selectbox("Chọn một nghề để xem các nghề tương đồng", occ_df[OCC_COL].unique())
        idx = int(occ_df[occ_df[OCC_COL] == occupation].index[0])
        sims = sim[idx]
        top_idx = [i for i in np.argsort(sims)[::-1] if i != idx][:10]
        transition_df = pd.DataFrame({
            "Nghề nghiệp": occ_df.iloc[top_idx][OCC_COL].values,
            "Độ tương đồng": sims[top_idx],
        })

        chart_header(
            f"Các Nghề Tương Đồng Với “{occupation}”",
            "Xếp hạng những nghề có hồ sơ năng lực gần giống nhất — đây là những hướng chuyển đổi "
            "nghề nghiệp khả thi nếu nghề hiện tại bị ảnh hưởng mạnh bởi AI.",
        )
        fig4 = px.bar(
            transition_df, x="Độ tương đồng", y="Nghề nghiệp", orientation="h",
            color="Độ tương đồng", color_continuous_scale="Tealgrn",
        )
        st.plotly_chart(style_fig(fig4, height=500), use_container_width=True)

        st.info(
            "### 📌 Nhận định\n\n"
            "🔵 Các nghề trong nhóm ngành công nghệ thường tạo thành nhiều cộng đồng tự nhiên dựa "
            "trên mức độ tương đồng kỹ năng.\n\n"
            "🟢 Một số nghề đóng vai trò trung tâm, kết nối với nhiều nghề khác trong mạng lưới.\n\n"
            "🟡 Lộ trình chuyển đổi nghề nghiệp có thể được tham khảo dựa trên độ tương đồng vector "
            "kỹ năng.\n\n"
            "🔴 Những nghề nằm ngoài mạng lưới (không có kết nối) thường có hồ sơ năng lực khác "
            "biệt, cần lộ trình đào tạo lại riêng nếu bị ảnh hưởng bởi tự động hóa."
        )
