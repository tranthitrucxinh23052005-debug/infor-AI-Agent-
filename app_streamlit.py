import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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

SHORT_VN = {
    "Automation Capacity Rating": "Tự động hóa",
    "Physical Action Requirement": "Thể chất",
    "Involved Uncertainty": "Bất định",
    "Domain Expertise Requirement": "Chuyên môn",
    "Interpersonal Communication Requirement": "Giao tiếp",
    "Human Agency Scale Rating": "Tự chủ con người",
}

SHORT_VN_TASK = {
    "Frequency": "Tần suất",
    "Importance": "Quan trọng",
    "Relevance": "Liên quan",
    "Occupation Mean Annual Wage": "Lương TB",
    "Occupation Employment": "Quy mô LĐ",
}

OCC_COL = "Occupation (O*NET-SOC Title)"
CAP_FEATURES = [
    "Automation Capacity Rating", "Physical Action Requirement", "Involved Uncertainty",
    "Domain Expertise Requirement", "Interpersonal Communication Requirement",
    "Human Agency Scale Rating",
]

# ============================================================
# HÀM TIỆN ÍCH DÙNG CHUNG
# ============================================================


def style_fig(fig, title=None, height=560):
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


def get_capability_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Trả về bản dữ liệu chỉ gồm 6 cột năng lực dùng chung cho nhiều trang,
    tránh lặp lại logic dropna ở từng nơi."""
    cols = [c for c in CAP_FEATURES if c in df.columns]
    return df.dropna(subset=cols).reset_index(drop=True) if dropna else df[cols]


def corr_chart(df: pd.DataFrame, cols: list, rename_map: dict, height: int = 560):
    """Vẽ ma trận tương quan dùng chung — tránh lặp code ở nhiều trang."""
    corr = df[cols].corr().rename(index=rename_map, columns=rename_map)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    return style_fig(fig, height=height)


def heuristic_note(text: str):
    """Nhắc người dùng rằng một chỉ số là công thức minh họa, không phải kết quả
    suy luận thống kê chặt chẽ — áp dụng nhất quán ở các trang có điểm số tự chế."""
    st.caption(f"⚠️ *Chỉ số minh họa (heuristic):* {text}")


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
[data-testid="stSidebar"]{ background:#FFFFFF; border-right: 1px solid #E2E8F0; }
[data-testid="stHeader"]{ background: rgba(255,255,255,0); }
h1, h2, h3 { color:#0F172A !important; font-weight:700 !important; }

.kpi-card{
    background:#FFFFFF; border:1px solid #E5E9F2; border-left:5px solid var(--accent, #2563EB);
    border-radius:18px; padding:22px 18px; text-align:center;
    box-shadow:0 4px 14px rgba(15,23,42,0.06); transition: transform .15s ease; margin-bottom:8px;
}
.kpi-card:hover{ transform: translateY(-3px); }
.kpi-number{ font-size:30px; font-weight:700; color:#1D4ED8; }
.kpi-label{ color:#64748B; font-size:14px; margin-top:4px; }

.stButton > button, .stDownloadButton > button{
    border-radius:999px !important; border:1px solid #2563EB !important; padding:0.5rem 1.4rem !important;
}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"]{
    border-radius:14px !important;
}
[data-testid="stMetric"]{
    background:#FFFFFF; border:1px solid #E5E9F2; border-radius:16px; padding:14px 10px;
    box-shadow:0 2px 10px rgba(15,23,42,0.05);
}
[data-testid="stExpander"]{ border-radius:14px; border:1px solid #E5E9F2; }
.insight-box{
    background:#F8FAFC; border:1px solid #E2E8F0; border-left:4px solid #2563EB;
    border-radius:12px; padding:16px 20px; margin:10px 0 18px 0; font-size:14.5px; color:#1E293B;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LỌC DỮ LIỆU: CHỈ GIỮ NHÓM NGÀNH CNTT / KHMT / KHOA HỌC DỮ LIỆU / HTTT
# ============================================================
# Ưu tiên 1: lọc theo mã O*NET-SOC nếu cột mã nghề có sẵn trong dữ liệu — đây là
# cách lọc chính xác nhất vì không phụ thuộc vào cách viết tên nghề.
# Nhóm mã SOC liên quan tới CNTT/KHMT/KHDL/HTTT đều nằm trong họ "15-12xx"
# (Computer Occupations) theo phân loại O*NET-SOC 2019/2020.
SOC_CODE_COL_CANDIDATES = ["O*NET-SOC Code", "Occupation Code", "SOC Code"]
SOC_PREFIX = "15-12"

# Ưu tiên 2 (dự phòng khi không có mã SOC): lọc theo từ khóa trong tên nghề,
# nhưng dùng RANH GIỚI TỪ (\b...\b) để tránh khớp nhầm các nghề không liên quan
# (ví dụ: "computer" không được khớp vào "Computer Numerically Controlled Tool
# Programmers" — nghề thuộc cơ khí, không thuộc CNTT).
IT_KEYWORDS = [
    r"computer (and information|network|systems|hardware|support)",
    r"software developer", r"software quality assurance", r"programmer",
    r"systems analyst", r"computer occupations", r"quality assurance analysts and testers",
    r"information security", r"information systems", r"information technology",
    r"information research scientist", r"database administrator", r"database architect",
    r"web developer", r"web and digital interface", r"computer support specialist",
    r"network support specialist", r"network administrator", r"network architect",
    r"cloud (engineer|architect)", r"devops",
    r"data scientist", r"data engineer", r"data warehousing",
    r"\bdata analyst", r"machine learning", r"\bartificial intelligence\b",
    r"business intelligence",
]
IT_PATTERN = "|".join(rf"(?:{kw})" for kw in IT_KEYWORDS)


@st.cache_data(show_spinner=False)
def filter_it(df: pd.DataFrame) -> pd.DataFrame:
    """Chỉ giữ lại các dòng thuộc nhóm ngành CNTT / Khoa học máy tính /
    Khoa học dữ liệu / Hệ thống thông tin. Ưu tiên lọc theo mã SOC nếu có,
    nếu không thì dùng từ khóa có ranh giới từ trên tên nghề."""
    soc_col = next((c for c in SOC_CODE_COL_CANDIDATES if c in df.columns), None)
    if soc_col is not None:
        mask = df[soc_col].astype(str).str.startswith(SOC_PREFIX)
        if mask.any():
            return df.loc[mask].reset_index(drop=True)

    if OCC_COL not in df.columns:
        return df
    mask = df[OCC_COL].astype(str).str.lower().str.contains(IT_PATTERN, regex=True, na=False)
    return df.loc[mask].reset_index(drop=True)


# ============================================================
# TẢI DỮ LIỆU
# ============================================================


@st.cache_data
def load_data():
    desires = pd.read_csv("data/domain_worker_desires.csv")
    metadata = pd.read_csv("data/domain_worker_metadata.csv")
    capability = pd.read_csv("data/expert_rated_technological_capability.csv")
    tasks = pd.read_csv("data/task_statement_with_metadata.csv")
    return desires, metadata, capability, tasks


desires, metadata, capability, tasks = load_data()

desires = filter_it(desires)
metadata = filter_it(metadata)
capability = filter_it(capability)
tasks = filter_it(tasks)

# Lưới an toàn: đồng bộ "desires" theo Task ID của "capability" — chỉ áp dụng
# khi cả hai có cột Task ID, tránh lỗi KeyError nếu schema thay đổi.
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
# Đã gộp 10 trang ban đầu xuống 7 trang để giảm trùng lặp nội dung
# (nhiều ma trận tương quan / biểu đồ giống nhau ở các trang khác nhau).

with st.sidebar:
    st.markdown("## 📊 Phân Tích Tác Động AI")
    st.caption("Hệ thống phân tích tác động của AI đến nhóm ngành Công nghệ thông tin")

    PAGES = [
        "📈 Tổng Quan",
        "🌍 Toàn Cảnh Tự Động Hóa",
        "🧠 Khoảng Cách Niềm Tin",
        "📋 Phân Tích Chi Tiết Đầu Việc",
        "🗂️ Phân Vùng & Thống Kê Chuyên Sâu",
        "🔗 Mạng Lưới Nghề Nghiệp",
        "🎯 Đánh Giá & Lộ Trình Cá Nhân",
    ]

    page = st.pills("Điều hướng", PAGES, default=PAGES[0], required=True, label_visibility="collapsed")

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
    automation_desire = round(desires["Automation Desire Rating"].mean(), 2) if not desires.empty else float("nan")
    domain_expertise = round(capability["Domain Expertise Requirement"].mean(), 2)

    c1, c2, c3 = st.columns(3)
    kpi_card(c1, f"{total_tasks:,}", "Tổng số đầu việc", PALETTE["primary"], "📋")
    kpi_card(c2, total_occupations, "Số nghề nghiệp", PALETTE["accent"], "💼")
    kpi_card(c3, automation_capacity, "Năng lực tự động hóa (TB)", PALETTE["danger"], "🤖")

    c4, c5, c6 = st.columns(3)
    kpi_card(c4, automation_desire, "Mức sẵn sàng giao việc cho AI (TB)", PALETTE["warning"], "🙋")
    kpi_card(c5, human_agency, "Quyền tự chủ con người (TB)", PALETTE["success"], "🧑‍💻")
    kpi_card(c6, domain_expertise, "Yêu cầu chuyên môn (TB)", PALETTE["secondary"], "🎓")

    # Tóm tắt điều hành bằng văn bản — tính động theo dữ liệu hiện tại, không
    # phải câu chữ cố định, để người ra quyết định nắm ngay thông điệp chính.
    gap = automation_capacity - human_agency
    gap_desc = (
        "năng lực tự động hóa kỹ thuật đang cao hơn quyền tự chủ con người"
        if gap > 0 else "quyền tự chủ con người vẫn chiếm ưu thế so với năng lực tự động hóa kỹ thuật"
    )
    st.markdown(
        f"""<div class="insight-box">
        📝 <b>Tóm tắt:</b> Trên {total_occupations} nghề và {total_tasks:,} đầu việc thuộc nhóm
        ngành công nghệ, {gap_desc} (chênh lệch {abs(gap):.2f} điểm trên thang 1–5).
        Mức sẵn sàng giao việc cho AI của người lao động hiện ở mức {automation_desire:.2f}/5,
        {'cao hơn' if automation_desire > automation_capacity else 'thấp hơn'} năng lực kỹ thuật
        thực tế của AI — cho thấy {'người lao động cởi mở hơn năng lực AI hiện có' if automation_desire > automation_capacity else 'còn khoảng cách niềm tin cần thu hẹp'}.
        </div>""",
        unsafe_allow_html=True,
    )

    st.divider()

    top_occ = tasks[OCC_COL].value_counts().head(15).reset_index()
    top_occ.columns = ["Nghề nghiệp", "Số đầu việc"]

    chart_header(
        "Top 15 Nghề Nghiệp Có Nhiều Đầu Việc Nhất",
        "Các nghề trong nhóm ngành công nghệ đang được khảo sát với số lượng đầu việc (task) "
        "nhiều nhất trong bộ dữ liệu — đây cũng là các nghề được phân tích sâu nhất trong báo cáo.",
    )
    fig = px.bar(top_occ, x="Số đầu việc", y="Nghề nghiệp", orientation="h",
                 color="Số đầu việc", color_continuous_scale="Blues")
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
        fig1 = px.bar(top_capacity, x="Automation Capacity Rating", y=OCC_COL, orientation="h",
                      color="Automation Capacity Rating", color_continuous_scale="Blues", labels=VN_LABELS)
        st.plotly_chart(style_fig(fig1, height=650), use_container_width=True)

    with col2:
        chart_header(
            "Top 20 Nghề Có Quyền Tự Chủ Con Người Cao Nhất",
            "Các nghề đòi hỏi con người ra quyết định, phán đoán nhiều — khó bị AI thay thế hoàn toàn.",
        )
        fig2 = px.bar(top_agency, x="Human Agency Scale Rating", y=OCC_COL, orientation="h",
                      color="Human Agency Scale Rating", color_continuous_scale="Greens", labels=VN_LABELS)
        st.plotly_chart(style_fig(fig2, height=650), use_container_width=True)

    st.divider()

    chart_header(
        "Tương Quan Giữa Năng Lực Tự Động Hóa Và Quyền Tự Chủ Con Người",
        "Mỗi điểm là một đầu việc. Màu thể hiện yêu cầu chuyên môn, kích thước thể hiện yêu cầu giao tiếp.",
    )
    fig3 = px.scatter(capability, x="Automation Capacity Rating", y="Human Agency Scale Rating",
                       color="Domain Expertise Requirement", size="Interpersonal Communication Requirement",
                       hover_name=OCC_COL, color_continuous_scale="Viridis", labels=VN_LABELS)
    st.plotly_chart(style_fig(fig3, height=650), use_container_width=True)

    chart_header(
        "Ma Trận Tương Quan Giữa Các Yếu Tố",
        "Giá trị gần 1 (xanh đậm) là tương quan thuận mạnh, gần -1 (đỏ đậm) là tương quan nghịch mạnh.",
    )
    st.plotly_chart(corr_chart(capability, CAP_FEATURES, SHORT_VN, height=600), use_container_width=True)

    chart_header(
        "Hồ Sơ Tổng Thể Của Nhóm Ngành Công Nghệ",
        "Giá trị trung bình của 6 yếu tố trên toàn bộ dữ liệu — dùng làm đường tham chiếu ở mục "
        "Đánh Giá Năng Lực Cá Nhân.",
    )
    values = [capability[c].mean() for c in CAP_FEATURES]
    labels_radar = [SHORT_VN[c] for c in CAP_FEATURES]
    fig5 = go.Figure()
    fig5.add_trace(go.Scatterpolar(r=values, theta=labels_radar, fill="toself",
                                    line_color=PALETTE["primary"], fillcolor="rgba(37,99,235,0.20)"))
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

    if "Task ID" not in capability.columns or "Task ID" not in desires.columns:
        st.warning(
            "Không thể thực hiện phân tích này vì dữ liệu thiếu cột 'Task ID' để ghép năng lực AI "
            "với mong muốn của người lao động."
        )
    else:
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
            total = tp + tn + fp + fn

            accuracy = (tp + tn) / total if total else float("nan")
            precision = tp / (tp + fp) if (tp + fp) else float("nan")
            recall = tp / (tp + fn) if (tp + fn) else float("nan")
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float("nan")
            trust_gap = abs(merged["Automation Capacity Rating"].mean() - merged["Automation Desire Rating"].mean())

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Độ chính xác (Accuracy)", f"{accuracy:.3f}" if accuracy == accuracy else "—")
            c2.metric("Precision", f"{precision:.3f}" if precision == precision else "—")
            c3.metric("Recall", f"{recall:.3f}" if recall == recall else "—")
            c4.metric("Điểm F1", f"{f1:.3f}" if f1 == f1 else "—")
            c5.metric("Khoảng cách niềm tin", round(trust_gap, 3))
            st.caption(
                "ℹ️ Accuracy/Precision/Recall/F1 ở đây coi 'năng lực AI ≥ ngưỡng' là nhãn tham chiếu "
                "(không phải nhãn thật tuyệt đối) để đo mức đồng thuận giữa đánh giá chuyên gia và "
                "mong muốn người lao động — không phải độ chính xác của một mô hình dự báo."
            )

            cm = np.array([[tn, fp], [fn, tp]])
            chart_header(
                "Ma Trận Đồng Thuận: Năng Lực AI So Với Mức Độ Sẵn Sàng Của Người Lao Động",
                "So khớp đánh giá 'khả thi' của chuyên gia (trục dọc) với mức 'sẵn sàng giao việc' của "
                "người lao động (trục ngang).",
            )
            fig = px.imshow(cm, text_auto=True, labels=dict(x="Người lao động", y="Chuyên gia"),
                             x=["Từ chối", "Chấp nhận"], y=["Từ chối", "Chấp nhận"], color_continuous_scale="Blues")
            st.plotly_chart(style_fig(fig, height=550), use_container_width=True)

            chart_header(
                "Phân Bố Năng Lực AI So Với Mức Độ Sẵn Sàng Giao Việc",
                "Đường nét đứt là ngưỡng quyết định đã chọn ở trên. Các góc thể hiện vùng AI làm được "
                "nhưng người lao động chưa muốn giao (hoặc ngược lại).",
            )
            fig2 = px.scatter(merged, x="Automation Capacity Rating", y="Automation Desire Rating",
                               color="Automation Capacity Rating", color_continuous_scale="Viridis", labels=VN_LABELS)
            fig2.add_vline(x=threshold, line_dash="dash", line_color=PALETTE["muted"])
            fig2.add_hline(y=threshold, line_dash="dash", line_color=PALETTE["muted"])
            st.plotly_chart(style_fig(fig2, height=650), use_container_width=True)

            st.markdown(
                f"""<div class="insight-box">
                📌 <b>Nhận định:</b> Khoảng cách niềm tin hiện ở mức <b>{trust_gap:.2f}</b> điểm
                (thang 1–5). Với ngưỡng {threshold:.1f}, tỉ lệ đồng thuận giữa chuyên gia và người
                lao động là <b>{accuracy:.1%}</b>; trong các trường hợp chuyên gia cho là khả thi,
                có <b>{recall:.1%}</b> được người lao động cũng đồng ý giao cho AI.
                </div>""",
                unsafe_allow_html=True,
            )

# ============================================================
# TRANG 4 — PHÂN TÍCH CHI TIẾT ĐẦU VIỆC
# ============================================================

elif page == "📋 Phân Tích Chi Tiết Đầu Việc":

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

    tab1, tab2 = st.tabs(["📊 Phân bố", "💰 Lương & Quy mô lao động"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            chart_header("Phân Bố Mức Độ Quan Trọng (Importance)",
                          "Mức độ quan trọng của các đầu việc đối với nghề nghiệp tương ứng, theo thang O*NET.")
            fig1 = px.histogram(df, x="Importance", nbins=40, color_discrete_sequence=[PALETTE["primary"]], labels=VN_LABELS)
            st.plotly_chart(style_fig(fig1, height=440), use_container_width=True)
        with col2:
            chart_header("Phân Bố Mức Độ Liên Quan (Relevance)",
                          "Mức độ liên quan của đầu việc đối với nghề nghiệp — giá trị cao nghĩa là đầu việc gắn liền với nghề đó.")
            fig2 = px.histogram(df, x="Relevance", nbins=40, color_discrete_sequence=[PALETTE["warning"]], labels=VN_LABELS)
            st.plotly_chart(style_fig(fig2, height=440), use_container_width=True)

        chart_header(
            "Ma Trận Tương Quan",
            "Mối liên hệ giữa tần suất, mức độ quan trọng, mức độ liên quan, lương và quy mô lao động.",
        )
        cols5 = ["Frequency", "Importance", "Relevance", "Occupation Mean Annual Wage", "Occupation Employment"]
        st.plotly_chart(corr_chart(df, cols5, SHORT_VN_TASK, height=520), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            chart_header("Phân Bố Lương Theo Năm",
                          "Hộp thể hiện khoảng lương phổ biến (trung vị, tứ phân vị); các điểm rời là nghề có mức lương bất thường.")
            fig3 = px.box(df, y="Occupation Mean Annual Wage", points="outliers", labels=VN_LABELS)
            fig3.update_traces(marker_color=PALETTE["secondary"], line_color=PALETTE["secondary"])
            st.plotly_chart(style_fig(fig3, height=480), use_container_width=True)
        with col2:
            chart_header("Phân Bố Quy Mô Lao Động",
                          "Số lượng người đang làm việc trong từng nghề — cho thấy nghề nào có quy mô thị trường lao động lớn.")
            fig4 = px.histogram(df, x="Occupation Employment", nbins=50, color_discrete_sequence=[PALETTE["success"]], labels=VN_LABELS)
            st.plotly_chart(style_fig(fig4, height=480), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            wage_df = df.groupby(OCC_COL)["Occupation Mean Annual Wage"].mean().sort_values(ascending=False).head(15).reset_index()
            chart_header("Top 15 Nghề Có Mức Lương Cao Nhất", "Các nghề có mức lương trung bình hàng năm cao nhất trong nhóm ngành.")
            fig6 = px.bar(wage_df, x="Occupation Mean Annual Wage", y=OCC_COL, orientation="h",
                          color="Occupation Mean Annual Wage", color_continuous_scale="Blues", labels=VN_LABELS)
            st.plotly_chart(style_fig(fig6, height=600), use_container_width=True)
        with col4:
            emp_df = df.groupby(OCC_COL)["Occupation Employment"].mean().sort_values(ascending=False).head(15).reset_index()
            chart_header("Top 15 Nghề Có Quy Mô Lao Động Lớn Nhất", "Các nghề thu hút số lượng người lao động đông nhất.")
            fig7 = px.bar(emp_df, x="Occupation Employment", y=OCC_COL, orientation="h",
                          color="Occupation Employment", color_continuous_scale="Purples", labels=VN_LABELS)
            st.plotly_chart(style_fig(fig7, height=600), use_container_width=True)

# ============================================================
# TRANG 5 — PHÂN VÙNG TÁC ĐỘNG & THỐNG KÊ CHUYÊN SÂU
# ============================================================

elif page == "🗂️ Phân Vùng & Thống Kê Chuyên Sâu":

    st.title("🗂️ Phân Vùng Tác Động & Thống Kê Chuyên Sâu")
    st.caption(
        "Phân cụm các đầu việc theo mức độ ảnh hưởng của AI (KMeans), sau đó áp dụng các kỹ thuật "
        "thống kê/học máy bổ sung (PCA, t-SNE, Random Forest, Isolation Forest) để hiểu sâu hơn cấu "
        "trúc dữ liệu của nhóm ngành công nghệ."
    )

    df = get_capability_features(capability)
    k = min(4, len(df))

    if k < 2:
        st.warning("Không đủ dữ liệu để phân cụm sau khi lọc theo nhóm ngành công nghệ.")
        st.stop()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CAP_FEATURES])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # --- Gán tên vùng dựa trên ĐẶC ĐIỂM THỰC của từng cụm, không theo số thứ tự
    # ngẫu nhiên của KMeans (đây là điểm sửa lỗi quan trọng so với bản gốc) ---
    cluster_profile = df.groupby("Cluster")[["Automation Capacity Rating", "Human Agency Scale Rating"]].mean()
    cluster_profile["automation_rank"] = cluster_profile["Automation Capacity Rating"].rank(ascending=False)
    cluster_profile["agency_rank"] = cluster_profile["Human Agency Scale Rating"].rank(ascending=False)

    def name_cluster(row):
        if row["automation_rank"] == 1:
            return "Vùng Rủi Ro Cao"          # năng lực tự động hóa cao nhất
        if row["agency_rank"] == 1:
            return "Thế Mạnh Con Người"        # quyền tự chủ con người cao nhất
        if row["automation_rank"] <= cluster_profile["automation_rank"].median():
            return "Vùng Kết Hợp"
        return "Vùng Ổn Định"

    cluster_profile["Zone"] = cluster_profile.apply(name_cluster, axis=1)
    df["Zone"] = df["Cluster"].map(cluster_profile["Zone"])

    n_components = max(1, min(3, len(df) - 1, len(CAP_FEATURES)))
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

    tab1, tab2 = st.tabs(["🗂️ Phân vùng tác động", "🔬 Thống kê chuyên sâu"])

    with tab1:
        chart_header(
            "Bản Đồ 3D Các Vùng Tác Động AI",
            "Mỗi điểm là một đầu việc, được nén từ 6 chiều dữ liệu xuống 3 chiều (PCA). Tên vùng "
            "được gán theo đặc điểm thực tế của từng cụm (mức tự động hóa & quyền tự chủ trung "
            "bình), không theo số thứ tự cố định.",
        )
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="Zone", hover_name=OCC_COL,
                             color_discrete_map=zone_colors, labels=VN_LABELS)
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(scene=dict(bgcolor="#FFFFFF"))
        st.plotly_chart(style_fig(fig, height=700), use_container_width=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            count_df = df["Zone"].value_counts().reset_index()
            count_df.columns = ["Vùng", "Số lượng"]
            chart_header("Tỷ Trọng Các Vùng Tác Động", "Tỷ lệ % số đầu việc rơi vào từng vùng tác động của AI.")
            fig2 = px.pie(count_df, names="Vùng", values="Số lượng", hole=0.5, color="Vùng", color_discrete_map=zone_colors)
            st.plotly_chart(style_fig(fig2, height=480), use_container_width=True)
        with col2:
            profile = df.groupby("Zone")[CAP_FEATURES].mean().round(2).rename(columns=SHORT_VN)
            profile.index.name = "Vùng"
            st.subheader("Hồ Sơ Trung Bình Theo Vùng")
            st.caption("💡 Giá trị trung bình của 6 yếu tố trong mỗi vùng.")
            st.dataframe(profile, use_container_width=True)

        danger_ratio = len(df[df.Zone == "Vùng Rủi Ro Cao"]) / len(df)
        st.markdown(
            f"""<div class="insight-box">
            📌 <b>Nhận định:</b> <b>{danger_ratio:.1%}</b> đầu việc rơi vào <b>Vùng Rủi Ro Cao</b>
            (năng lực tự động hóa kỹ thuật cao nhất trong các nhóm). Các nghề trong <b>Thế Mạnh Con
            Người</b> nổi bật ở quyền tự chủ và phán đoán; <b>Vùng Kết Hợp</b> phản ánh nhóm AI hỗ
            trợ mạnh nhưng chưa thay thế hoàn toàn; <b>Vùng Ổn Định</b> cân bằng giữa hai yếu tố.
            </div>""",
            unsafe_allow_html=True,
        )

    with tab2:
        chart_header("Ma Trận Tương Quan", "Mối liên hệ tuyến tính giữa các yếu tố năng lực.")
        st.plotly_chart(corr_chart(df, CAP_FEATURES, SHORT_VN, height=560), use_container_width=True)

        chart_header(
            "Không Gian Đặc Trưng PCA",
            f"Nén 6 chiều dữ liệu xuống 2 thành phần chính, giải thích "
            f"{sum(pca.explained_variance_ratio_[:2]):.1%} phương sai của dữ liệu gốc.",
        )
        fig_pca = px.scatter(df, x="x", y="y", color="Automation Capacity Rating", hover_name=OCC_COL,
                              color_continuous_scale="Viridis", labels=VN_LABELS)
        st.plotly_chart(style_fig(fig_pca, height=620), use_container_width=True)

        perplexity = min(30, max(5, (len(df) - 1) // 3))
        if perplexity < len(df):
            chart_header(
                "Bản Đồ t-SNE Các Đầu Việc",
                "Kỹ thuật giảm chiều phi tuyến giúp các đầu việc có hồ sơ năng lực tương tự đứng "
                "gần nhau hơn so với PCA, làm nổi bật cấu trúc cụm tự nhiên trong dữ liệu.",
            )
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            embedding = tsne.fit_transform(X_scaled)
            df["t1"], df["t2"] = embedding[:, 0], embedding[:, 1]
            fig_tsne = px.scatter(df, x="t1", y="t2", color="Human Agency Scale Rating", hover_name=OCC_COL,
                                   color_continuous_scale="Turbo", labels=VN_LABELS)
            st.plotly_chart(style_fig(fig_tsne, height=650), use_container_width=True)
        else:
            st.info("Không đủ số lượng đầu việc để tính t-SNE ổn định sau khi lọc dữ liệu.")

        # --- Random Forest với train/test split để R² phản ánh đúng khả năng
        # tổng quát hóa, thay vì chỉ báo cáo điểm trên tập huấn luyện (lỗi cũ) ---
        y = df["Human Agency Scale Rating"]
        X_rf = df[[c for c in CAP_FEATURES if c != "Human Agency Scale Rating"]]
        if len(df) >= 10:
            X_tr, X_te, y_tr, y_te = train_test_split(X_rf, y, test_size=0.25, random_state=42)
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(X_tr, y_tr)
            r2_test = r2_score(y_te, rf.predict(X_te))
            st.caption(f"📐 R² trên tập kiểm tra (giữ lại 25% dữ liệu): **{r2_test:.2f}** — phản ánh khả năng tổng quát hóa thực tế của mô hình.")
        else:
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(X_rf, y)
            st.caption("⚠️ Mẫu quá nhỏ để chia train/test — mô hình dưới đây chỉ mang tính minh họa xu hướng.")

        importance = pd.DataFrame({
            "Yếu tố": [SHORT_VN.get(c, c) for c in X_rf.columns],
            "Mức độ ảnh hưởng": rf.feature_importances_,
        })
        chart_header(
            "Yếu Tố Ảnh Hưởng Mạnh Nhất Đến Quyền Tự Chủ Con Người",
            "Mô hình Random Forest ước lượng yếu tố nào quyết định nhiều nhất đến mức độ tự chủ của con người trong công việc.",
        )
        fig_imp = px.bar(importance.sort_values("Mức độ ảnh hưởng"), x="Mức độ ảnh hưởng", y="Yếu tố",
                          orientation="h", color="Mức độ ảnh hưởng", color_continuous_scale="Plasma")
        st.plotly_chart(style_fig(fig_imp, height=420), use_container_width=True)

        contamination = st.slider("Tỉ lệ bất thường kỳ vọng (Isolation Forest)", 0.01, 0.10, 0.03, 0.01)
        iso = IsolationForest(contamination=contamination, random_state=42)
        df["Outlier"] = iso.fit_predict(X_scaled)
        df["Type"] = np.where(df["Outlier"] == -1, "Bất thường", "Bình thường")
        chart_header(
            "Phát Hiện Đầu Việc Bất Thường",
            "Mô hình Isolation Forest đánh dấu các đầu việc có hồ sơ năng lực khác biệt rõ rệt so với phần còn lại.",
        )
        fig_out = px.scatter(df, x="x", y="y", color="Type", hover_name=OCC_COL,
                              color_discrete_map={"Bình thường": PALETTE["secondary"], "Bất thường": PALETTE["danger"]})
        st.plotly_chart(style_fig(fig_out, height=620), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Phương sai PCA giải thích", f"{sum(pca.explained_variance_ratio_):.1%}")
        c2.metric("Số bản ghi", len(df))
        c3.metric("Số bất thường", len(df[df.Type == "Bất thường"]))
        c4.metric("Số đặc trưng", len(CAP_FEATURES))

# ============================================================
# TRANG 6 — MẠNG LƯỚI NGHỀ NGHIỆP
# ============================================================

elif page == "🔗 Mạng Lưới Nghề Nghiệp":

    st.title("🔗 Mạng Lưới Nghề Nghiệp")
    st.caption(
        "Mỗi nghề trong nhóm ngành công nghệ được gộp thành một điểm dữ liệu (giá trị trung bình "
        "của 6 yếu tố), sau đó tính độ tương đồng cosine giữa các nghề để dựng mạng lưới. Hai nghề "
        "có đường nối với nhau nghĩa là hồ sơ năng lực của chúng khá giống nhau — đây cũng là cơ "
        "sở tham khảo cho việc chuyển đổi nghề nghiệp trong tương lai."
    )

    occ_df = capability.dropna(subset=CAP_FEATURES).groupby(OCC_COL)[CAP_FEATURES].mean().reset_index()

    if len(occ_df) < 2:
        st.warning("Không đủ số nghề khác nhau trong nhóm ngành công nghệ để dựng mạng lưới tương đồng.")
    else:
        threshold = st.slider(
            "Ngưỡng độ tương đồng để nối hai nghề", 0.80, 0.999, 0.95, 0.001,
            help="Ngưỡng cao hơn → mạng lưới thưa hơn, chỉ giữ lại các cặp nghề rất giống nhau.",
        )

        scaler_net = StandardScaler()
        X_net = scaler_net.fit_transform(occ_df[CAP_FEATURES])
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
            {"occupation": G.nodes[node]["occupation"], "x": pos[node][0], "y": pos[node][1], "degree": G.degree(node)}
            for node in G.nodes()
        ])

        chart_header(
            "Bản Đồ Mạng Lưới Nghề Nghiệp",
            "Vị trí các nghề được sắp xếp bằng spring layout — nghề nào càng giống nhiều nghề khác (số kết nối cao) thì điểm càng lớn.",
        )
        fig = px.scatter(node_df, x="x", y="y", size="degree", hover_name="occupation",
                          color="degree", color_continuous_scale="Blues", labels=VN_LABELS)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        st.plotly_chart(style_fig(fig, height=600), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số nghề", len(G.nodes()))
        c2.metric("Số kết nối", len(G.edges()))
        c3.metric("Bậc kết nối TB", round(float(np.mean([d for _, d in G.degree()])), 2) if len(G) else 0)
        c4.metric("Mật độ mạng lưới", round(nx.density(G), 3) if len(G) > 1 else 0)

        communities = (
            list(greedy_modularity_communities(G)) if len(G.edges()) > 0
            else [frozenset([node]) for node in G.nodes()]
        )
        community_map = {node: i for i, community in enumerate(communities) for node in community}
        node_df["Cộng đồng"] = [community_map.get(node, -1) for node in G.nodes()]

        chart_header(
            "Cộng Đồng Nghề Nghiệp Tự Nhiên",
            "Thuật toán phát hiện cộng đồng (modularity) nhóm các nghề có liên kết chặt chẽ với nhau — mỗi màu là một nhóm nghề có hồ sơ năng lực gần gũi.",
        )
        fig2 = px.scatter(node_df, x="x", y="y", color="Cộng đồng", size="degree", hover_name="occupation", labels=VN_LABELS)
        fig2.update_xaxes(visible=False)
        fig2.update_yaxes(visible=False)
        st.plotly_chart(style_fig(fig2, height=600), use_container_width=True)

        if len(G.edges()) > 0:
            centrality = nx.degree_centrality(G)
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:15]
            top_df = pd.DataFrame({
                "Nghề nghiệp": [G.nodes[n]["occupation"] for n, _ in top_nodes],
                "Độ trung tâm": [v for _, v in top_nodes],
            })
            chart_header(
                "Nghề Nghiệp Trung Tâm Của Mạng Lưới",
                "Độ trung tâm cao cho thấy nghề đó có hồ sơ năng lực 'điển hình', kết nối với nhiều nghề khác — thường là điểm trung chuyển tốt khi cân nhắc chuyển nghề.",
            )
            fig3 = px.bar(top_df, x="Độ trung tâm", y="Nghề nghiệp", orientation="h",
                          color="Độ trung tâm", color_continuous_scale="Blues")
            st.plotly_chart(style_fig(fig3, height=550), use_container_width=True)
        else:
            st.info("Chưa có cặp nghề nào vượt ngưỡng tương đồng đã chọn — hãy giảm ngưỡng ở thanh trượt phía trên.")

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
            "Xếp hạng những nghề có hồ sơ năng lực gần giống nhất — hướng chuyển đổi nghề nghiệp khả thi nếu nghề hiện tại bị ảnh hưởng mạnh bởi AI.",
        )
        fig4 = px.bar(transition_df, x="Độ tương đồng", y="Nghề nghiệp", orientation="h",
                      color="Độ tương đồng", color_continuous_scale="Tealgrn")
        st.plotly_chart(style_fig(fig4, height=480), use_container_width=True)

# ============================================================
# TRANG 7 — ĐÁNH GIÁ NĂNG LỰC CÁ NHÂN & LỘ TRÌNH HỌC TẬP
# ============================================================

elif page == "🎯 Đánh Giá & Lộ Trình Cá Nhân":

    st.title("🎯 Đánh Giá Năng Lực Cá Nhân & Lộ Trình Phát Triển")
    st.caption(
        "Nhập hồ sơ năng lực của bản thân để so sánh với mức trung bình ngành, nhận gợi ý nghề "
        "nghiệp phù hợp và lộ trình phát triển kỹ năng cá nhân hóa."
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
    market_vector = np.array([capability[c].mean() for c in CAP_FEATURES])
    labels6 = [SHORT_VN[c] for c in CAP_FEATURES]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=market_vector, theta=labels6, fill="toself", name="Trung bình ngành",
                                   line_color=PALETTE["secondary"], fillcolor="rgba(14,165,233,0.20)"))
    fig.add_trace(go.Scatterpolar(r=user_vector, theta=labels6, fill="toself", name="Hồ sơ của bạn",
                                   line_color=PALETTE["accent"], fillcolor="rgba(124,58,237,0.20)"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])))
    chart_header("So Sánh Hồ Sơ Cá Nhân Với Mức Trung Bình Ngành",
                  "Vùng càng lệch khỏi nhau ở trục nào thì bạn càng khác biệt so với mặt bằng chung ở yếu tố đó.")
    st.plotly_chart(style_fig(fig, height=560), use_container_width=True)

    distance = np.linalg.norm(user_vector - market_vector)
    score = max(0, 100 - distance * 15)
    heuristic_note(
        "độ phù hợp = 100 − (khoảng cách Euclid × 15), chỉ dùng để xếp hạng tương đối, "
        "không phải xác suất hay điểm chuẩn hóa thống kê."
    )
    st.metric("🎯 Độ phù hợp với mặt bằng ngành", f"{score:.1f}%")

    gap = user_vector - market_vector
    gap_df = pd.DataFrame({"Yếu tố": labels6, "Khoảng cách": gap})
    chart_header("Khoảng Cách Theo Từng Yếu Tố",
                  "Giá trị dương: bạn đang vượt mức trung bình ngành. Giá trị âm: bạn đang thấp hơn.")
    fig2 = px.bar(gap_df, x="Yếu tố", y="Khoảng cách", color="Khoảng cách", color_continuous_scale="RdYlGn")
    st.plotly_chart(style_fig(fig2, height=420), use_container_width=True)

    X_knn = capability[CAP_FEATURES].dropna()
    k_neighbors = min(10, len(X_knn))
    if k_neighbors >= 1:
        model = NearestNeighbors(n_neighbors=k_neighbors)
        model.fit(X_knn)
        distances, indices = model.kneighbors([user_vector])
        recommend = capability.loc[X_knn.index].iloc[indices[0]]
        chart_header("Nghề Nghiệp Gợi Ý Cho Bạn", "Các nghề có hồ sơ năng lực gần nhất với hồ sơ bạn vừa nhập.")
        st.dataframe(recommend[[OCC_COL]].rename(columns={OCC_COL: "Nghề nghiệp gợi ý"}), use_container_width=True)

    if score >= 85:
        st.success("✅ Hồ sơ của bạn rất phù hợp với mặt bằng chung của nhóm ngành công nghệ.")
    elif score >= 70:
        st.warning("⚠️ Hồ sơ khá phù hợp, nên nâng cấp thêm một vài kỹ năng.")
    else:
        st.error("🚨 Khoảng cách kỹ năng còn lớn so với mặt bằng ngành — nên có kế hoạch nâng cấp kỹ năng.")

    st.divider()
    st.markdown("### 🗺️ Lộ trình phát triển theo nghề mục tiêu")

    occ_stats = capability.dropna(subset=CAP_FEATURES).groupby(OCC_COL)[CAP_FEATURES].mean()
    wage_emp = tasks.groupby(OCC_COL)[["Occupation Mean Annual Wage", "Occupation Employment"]].mean()
    occ_stats = occ_stats.join(wage_emp, how="inner").dropna()

    if len(occ_stats) < 5:
        st.warning("Không đủ số nghề trong nhóm ngành công nghệ để xây dựng lộ trình gợi ý đáng tin cậy.")
    else:
        scaler_r = StandardScaler()
        X_occ_scaled = scaler_r.fit_transform(occ_stats[CAP_FEATURES])

        if len(occ_stats) >= 10:
            X_tr, X_te, y_tr, y_te = train_test_split(
                occ_stats[CAP_FEATURES], occ_stats["Occupation Mean Annual Wage"], test_size=0.25, random_state=42
            )
            rf_wage = RandomForestRegressor(n_estimators=300, random_state=42)
            rf_wage.fit(X_tr, y_tr)
            r2_test = r2_score(y_te, rf_wage.predict(X_te))
            r2_display = f"{r2_test:.2f}"
            r2_note = "R² đo trên 25% dữ liệu giữ lại để kiểm tra (không dùng để huấn luyện)."
        else:
            rf_wage = RandomForestRegressor(n_estimators=300, random_state=42)
            rf_wage.fit(occ_stats[CAP_FEATURES], occ_stats["Occupation Mean Annual Wage"])
            r2_display = "—"
            r2_note = "Mẫu dưới 10 nghề nên không đủ để chia train/test; mô hình bên dưới chỉ mang tính minh họa xu hướng, không có chỉ số độ khớp đáng tin cậy."

        importance_wage = pd.DataFrame({
            "Yếu tố": [SHORT_VN.get(c, c) for c in CAP_FEATURES],
            "Mức độ ảnh hưởng đến lương": rf_wage.feature_importances_,
        }).sort_values("Mức độ ảnh hưởng đến lương", ascending=False)

        c1, c2, c3 = st.columns(3)
        c1.metric("Số nghề dùng để huấn luyện", len(occ_stats))
        c2.metric("R² trên tập kiểm tra", r2_display)
        c3.metric("Số yếu tố đầu vào", len(CAP_FEATURES))
        st.caption(f"ℹ️ {r2_note}")
        st.caption(
            "⚠️ Mẫu huấn luyện chỉ gồm các nghề trong nhóm ngành công nghệ đã lọc, nên kết quả mang "
            "tính minh họa xu hướng, không phải con số dự báo tuyệt đối."
        )

        chart_header("Yếu Tố Ảnh Hưởng Mạnh Nhất Đến Mức Lương",
                      "Yếu tố năng lực nào tác động nhiều nhất đến lương trung bình của nghề, theo Random Forest.")
        fig_imp = px.bar(importance_wage, x="Mức độ ảnh hưởng đến lương", y="Yếu tố", orientation="h",
                          color="Mức độ ảnh hưởng đến lương", color_continuous_scale="Plasma")
        st.plotly_chart(style_fig(fig_imp, height=380), use_container_width=True)

        target_priority = st.radio(
            "Mục tiêu ưu tiên", ["Tăng lương", "Giảm rủi ro tự động hóa", "Phát triển toàn diện"], horizontal=True,
        )

        user_scaled = scaler_r.transform([user_vector])
        similarity = cosine_similarity(user_scaled, X_occ_scaled)[0]
        wage_arr = occ_stats["Occupation Mean Annual Wage"].values
        risk_arr = occ_stats["Automation Capacity Rating"].values

        def _normalize(v):
            v = np.asarray(v, dtype=float)
            rng = v.max() - v.min()
            return (v - v.min()) / rng if rng > 0 else np.ones_like(v) * 0.5

        sim_norm, wage_norm, risk_norm = _normalize(similarity), _normalize(wage_arr), _normalize(risk_arr)

        if target_priority == "Tăng lương":
            comp_score = 0.5 * sim_norm + 0.5 * wage_norm
        elif target_priority == "Giảm rủi ro tự động hóa":
            comp_score = 0.5 * sim_norm + 0.5 * (1 - risk_norm)
        else:
            comp_score = (sim_norm + wage_norm + (1 - risk_norm)) / 3
        heuristic_note("độ phù hợp nghề = trung bình có trọng số giữa độ tương đồng hồ sơ, mức lương và (1 − rủi ro tự động hóa), chuẩn hóa 0–1 — dùng để xếp hạng tương đối giữa các nghề trong nhóm dữ liệu hiện tại.")

        result = occ_stats.copy()
        result["Độ phù hợp (%)"] = (comp_score * 100).round(1)
        result["Lương trung bình"] = result["Occupation Mean Annual Wage"].round(0)
        result["Rủi ro tự động hóa (%)"] = (result["Automation Capacity Rating"] / 5 * 100).round(1)
        result = result.sort_values("Độ phù hợp (%)", ascending=False)
        top5 = result.head(5)

        chart_header("Top 5 Nghề Nghiệp Được Đề Xuất",
                      "Xếp hạng dựa trên độ tương đồng hồ sơ năng lực, kết hợp với mục tiêu ưu tiên bạn đã chọn.")
        st.dataframe(
            top5[["Lương trung bình", "Rủi ro tự động hóa (%)", "Độ phù hợp (%)"]].rename_axis("Nghề nghiệp").reset_index(),
            use_container_width=True,
        )

        st.markdown("#### 🗺️ Lộ trình phát triển kỹ năng")
        candidate_list = top5.index.tolist() + [o for o in result.index if o not in top5.index][:15]
        target_occ = st.selectbox("Chọn nghề mục tiêu để xây lộ trình", candidate_list)

        target_vector = occ_stats.loc[target_occ, CAP_FEATURES].values
        gap_series = pd.Series(target_vector - user_vector, index=CAP_FEATURES)

        DEVELOPABLE = [
            "Domain Expertise Requirement", "Interpersonal Communication Requirement",
            "Human Agency Scale Rating", "Involved Uncertainty",
        ]
        ACTIVITY_HINTS = {
            "Domain Expertise Requirement": "Học chuyên sâu kiến thức ngành: chứng chỉ chuyên môn, dự án thực tế, tài liệu kỹ thuật chuyên ngành.",
            "Interpersonal Communication Requirement": "Rèn kỹ năng giao tiếp: thuyết trình, viết báo cáo kỹ thuật, làm việc nhóm liên phòng ban.",
            "Human Agency Scale Rating": "Rèn khả năng ra quyết định độc lập: chủ động đề xuất giải pháp, quản lý dự án nhỏ, chịu trách nhiệm kết quả.",
            "Involved Uncertainty": "Rèn khả năng thích nghi với sự mơ hồ: làm việc trong môi trường thay đổi liên tục, thử nghiệm — đánh giá — cải tiến.",
        }
        importance_map = dict(zip(CAP_FEATURES, rf_wage.feature_importances_))

        roadmap_rows = []
        for f in DEVELOPABLE:
            g = gap_series[f]
            if g <= 0.15:
                continue
            priority_score = g * importance_map[f]
            duration = "4–6 tuần" if g < 0.5 else ("2–3 tháng" if g < 1.2 else "4–6 tháng")
            roadmap_rows.append({
                "Yếu tố cần phát triển": SHORT_VN.get(f, f),
                "Khoảng cách hiện tại": round(g, 2),
                "Mức độ ưu tiên": round(priority_score, 3),
                "Thời gian đề xuất": duration,
                "Gợi ý hoạt động": ACTIVITY_HINTS[f],
            })

        roadmap_df = pd.DataFrame(roadmap_rows)
        if not roadmap_df.empty:
            roadmap_df = roadmap_df.sort_values("Mức độ ưu tiên", ascending=False).reset_index(drop=True)
            roadmap_df.insert(0, "Bước", range(1, len(roadmap_df) + 1))

        if roadmap_df.empty:
            st.success(f"✅ Hồ sơ hiện tại của bạn đã tương đối phù hợp với “{target_occ}”. Không có khoảng cách kỹ năng đáng kể cần ưu tiên phát triển.")
        else:
            chart_header(f"Lộ Trình Phát Triển Để Hướng Tới “{target_occ}”",
                          "Các yếu tố được sắp xếp theo mức độ ưu tiên (khoảng cách kỹ năng × mức độ ảnh hưởng đến lương).")
            st.dataframe(
                roadmap_df[["Bước", "Yếu tố cần phát triển", "Khoảng cách hiện tại", "Thời gian đề xuất", "Gợi ý hoạt động"]],
                use_container_width=True, hide_index=True,
            )
            fig_gap = px.bar(roadmap_df, x="Khoảng cách hiện tại", y="Yếu tố cần phát triển", orientation="h",
                              color="Mức độ ưu tiên", color_continuous_scale="OrRd")
            st.plotly_chart(style_fig(fig_gap, "Mức Độ Ưu Tiên Theo Khoảng Cách Kỹ Năng", height=340), use_container_width=True)

        pred_wage_now = rf_wage.predict([user_vector])[0]
        pred_wage_target = occ_stats.loc[target_occ, "Occupation Mean Annual Wage"]
        wage_uplift = pred_wage_target - pred_wage_now
        risk_target = occ_stats.loc[target_occ, "Automation Capacity Rating"] / 5 * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Lương ước tính với hồ sơ hiện tại", f"${pred_wage_now:,.0f}")
        c2.metric(f"Lương trung bình của “{target_occ}”", f"${pred_wage_target:,.0f}", delta=f"${wage_uplift:,.0f}")
        c3.metric("Rủi ro tự động hóa của nghề mục tiêu", f"{risk_target:.1f}%")

        st.info(
            "### 📌 Lưu ý\n\n"
            "Lộ trình trên được xây dựng từ phân tích dữ liệu thống kê (khoảng cách hồ sơ năng lực "
            "kết hợp mức độ ảnh hưởng của từng yếu tố đến lương), mang tính định hướng. "
            "Đây không phải lời khuyên nghề nghiệp chuyên sâu — nên kết hợp thêm tư vấn từ chuyên "
            "gia hướng nghiệp hoặc người quản lý trực tiếp trước khi ra quyết định."
        )
