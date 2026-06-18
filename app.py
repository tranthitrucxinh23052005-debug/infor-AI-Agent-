import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Agent Dashboard – CS 2025",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════
# CUSTOM CSS – Navy/Blue/White bo tròn
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #131B2E;
    }
    [data-testid="stSidebar"] * {
        color: #F1F5F9 !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        gap: 4px;
    }
    [data-testid="stSidebar"] .stRadio label {
        padding: 10px 14px;
        border-radius: 14px;
        transition: all 0.2s;
        margin-bottom: 2px;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] .stRadio [data-selected="true"] label {
        background: #3B82F6 !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
    }
    
    /* Cards */
    .kpi-card {
        background: white;
        border-radius: 20px;
        padding: 20px 24px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        text-align: center;
        transition: transform 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }
    .kpi-icon { font-size: 28px; margin-bottom: 6px; }
    .kpi-value { font-family: 'Space Grotesk', sans-serif; font-size: 28px; font-weight: 700; }
    .kpi-label { font-size: 12px; color: #64748B; margin-top: 4px; }
    .kpi-sub { font-size: 10px; color: #94A3B8; margin-top: 2px; }
    
    /* Insight boxes */
    .insight-box {
        background: white;
        border-radius: 16px;
        padding: 16px 18px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        display: flex;
        gap: 12px;
        align-items: flex-start;
        margin-bottom: 10px;
        font-size: 13px;
        color: #475569;
        line-height: 1.5;
    }
    .insight-bar {
        min-width: 4px;
        border-radius: 99px;
        align-self: stretch;
    }
    
    /* Chart card */
    .chart-card {
        background: white;
        border-radius: 24px;
        padding: 24px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    }
    
    /* Section header */
    .section-header h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
    }
    .section-header p {
        font-size: 14px;
        color: #64748B;
        margin-top: 2px;
        margin-left: 40px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #F1F5F9;
        border-radius: 16px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px !important;
        padding: 8px 18px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #0F172A !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Metric card */
    .metric-mini {
        background: white;
        border-radius: 20px;
        padding: 18px;
        border: 1px solid #E2E8F0;
        text-align: center;
        box-shadow: 0 4px 12px rgba(15,23,42,0.06);
    }
    .metric-mini .value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 24px;
        font-weight: 700;
    }
    .metric-mini .label {
        font-size: 11px;
        color: #64748B;
        margin-top: 2px;
    }
    
    /* Guide box */
    .guide-box {
        background: white;
        border-radius: 24px;
        padding: 20px 24px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        margin-top: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════

# FIGURE 1: AI Landscape
automation_capacity = [
    ("IS Managers", 2.85), ("Mgmt Analysts", 2.92), ("CS Researchers", 3.05),
    ("Sys Engineers", 3.12), ("IT PM", 3.18), ("InfoSec", 3.25),
    ("QC Mgrs", 3.30), ("Net/SysAdmin", 3.42), ("Sys Analysts", 3.48),
    ("Support Spec.", 3.55), ("Programmers", 3.68), ("QA/Testers", 3.75),
    ("BI Analysts", 3.82), ("DBA", 3.88), ("Data Entry", 3.95),
    ("Web Admins", 4.02), ("Web Devs", 4.15),
]

automation_desire = [
    ("Web Devs", 4.0), ("DBA", 3.8), ("Data Entry", 4.2), ("QA/Testers", 3.6),
    ("BI Analysts", 3.5), ("Net/SysAdmin", 3.3), ("Programmers", 3.5),
    ("Web Admins", 3.4), ("Support Spec.", 3.0), ("Sys Analysts", 2.8),
    ("IS Managers", 2.5), ("CS Researchers", 2.2),
]

scatter_expert_desire = [
    ("Web Devs", 4.15, 4.0, "#10B981"), ("DBA", 3.88, 3.8, "#10B981"),
    ("Data Entry", 3.95, 4.2, "#10B981"), ("QA/Testers", 3.75, 3.6, "#10B981"),
    ("BI Analysts", 3.82, 3.5, "#10B981"), ("Programmers", 3.68, 3.5, "#10B981"),
    ("Web Admins", 4.02, 3.4, "#3B82F6"), ("Support Spec.", 3.55, 3.0, "#3B82F6"),
    ("Net/SysAdmin", 3.42, 3.3, "#F59E0B"), ("Sys Analysts", 3.48, 2.8, "#EF4444"),
    ("IT PM", 3.18, 2.5, "#EF4444"), ("InfoSec", 3.25, 2.8, "#EF4444"),
    ("IS Managers", 2.85, 2.5, "#EF4444"), ("CS Researchers", 3.05, 2.2, "#EF4444"),
    ("Sys Engineers", 3.12, 2.6, "#EF4444"), ("Mgmt Analysts", 2.92, 2.8, "#F59E0B"),
    ("QC Mgrs", 3.30, 3.0, "#F59E0B"),
]

# FIGURE 2: Trust Gap
confusion_3x3 = [
    ("Thấp (1-2)", "Thấp (1-2)", 45, 7.1), ("Thấp (1-2)", "TB (3)", 28, 4.4),
    ("Thấp (1-2)", "Cao (4-5)", 12, 1.9), ("TB (3)", "Thấp (1-2)", 38, 6.0),
    ("TB (3)", "TB (3)", 95, 15.0), ("TB (3)", "Cao (4-5)", 72, 11.4),
    ("Cao (4-5)", "Thấp (1-2)", 22, 3.5), ("Cao (4-5)", "TB (3)", 127, 20.1),
    ("Cao (4-5)", "Cao (4-5)", 193, 30.5),
]

binary_matrix = [
    ("AI KHÔNG thể × Worker KHÔNG muốn", 206, 32.6, "#F1F5F9"),
    ("AI KHÔNG thể × Worker Muốn", 84, 13.3, "#3B82F6"),
    ("AI CÓ thể × Worker KHÔNG muốn", 193, 30.5, "#F59E0B"),
    ("AI CÓ thể × Worker Muốn", 149, 23.6, "#10B981"),
]

class_metrics = [
    ("Accuracy", 0.562, "#10B981"), ("Precision", 0.639, "#3B82F6"),
    ("Recall", 0.170, "#F59E0B"), ("F1-Score", 0.170, "#8B5CF6"),
]

# FIGURE 3: Motivation
reasons_auto = [
    ("Tiết kiệm thời gian", 72.3), ("Mở rộng quy mô (Scale)", 58.1),
    ("Giảm lỗi người dùng", 51.7), ("Tác vụ lặp lại", 48.9),
    ("Giảm căng thẳng", 35.2), ("Tác vụ quá khó", 22.8),
]

reasons_human = [
    ("Kiến thức chuyên sâu", 78.5), ("Kiểm soát trực tiếp", 65.3),
    ("Vấn đề đạo đức", 58.7), ("Giám sát chất lượng", 55.2),
    ("Sự đồng cảm (Empathy)", 48.1), ("Môi trường thay đổi", 42.6),
    ("Hành động vật lý", 31.4),
]

task_chars_level = [
    ("Hành động Vật lý", 3.2, 2.5, 1.8),
    ("Giao tiếp Người", 3.8, 2.9, 2.1),
    ("Độ không chắc chắn", 3.5, 3.0, 2.3),
    ("Chuyên môn sâu", 4.2, 3.5, 2.8),
]

# FIGURE 4: LLM Usage
llm_heatmap = [
    ("Tra cứu thông tin", 45, 30, 15, 10),
    ("Giao tiếp / Email", 42, 28, 18, 12),
    ("Chỉnh sửa văn bản", 38, 32, 18, 12),
    ("Tạo ý tưởng", 35, 28, 22, 15),
    ("Lập trình (Code)", 33, 30, 20, 17),
    ("Xử lý dữ liệu", 28, 25, 25, 22),
    ("Phân tích dữ liệu", 22, 28, 28, 22),
    ("Ra quyết định", 15, 20, 30, 35),
    ("Thiết kế hệ thống", 8, 15, 25, 52),
]

llm_familiarity = [
    ("Dùng thường xuyên", 52, "#10B981"),
    ("Có kinh nghiệm", 28, "#3B82F6"),
    ("Biết nhưng ít dùng", 15, "#F59E0B"),
    ("Chưa biết", 5, "#EF4444"),
]

ai_attitudes = [
    ("Công việc tẻ nhạt", 42, 28, 15, 10, 5),
    ("Công việc quan trọng", 15, 22, 28, 20, 15),
    ("Công việc hằng ngày", 35, 30, 18, 12, 5),
    ("Nhận thức tác hại", 12, 18, 30, 25, 15),
]

# FIGURE 5: Task Characteristics
scatter_phys_inter = [
    ("Data Entry", 1.2, 1.5, 4.5), ("Web Devs", 1.0, 1.3, 4.2),
    ("DBA", 1.1, 2.0, 3.9), ("Programmers", 1.5, 2.2, 3.8),
    ("QA/Testers", 1.3, 1.8, 4.0), ("Net/SysAdmin", 2.0, 3.0, 3.2),
    ("Support Spec.", 2.5, 3.5, 2.8), ("IT PM", 1.8, 3.8, 3.0),
    ("IS Managers", 2.2, 4.0, 2.6), ("InfoSec", 3.0, 2.5, 3.1),
    ("Sys Analysts", 1.5, 2.8, 3.5), ("CS Researchers", 2.8, 4.2, 2.5),
    ("Sys Engineers", 3.5, 3.8, 2.3), ("BI Analysts", 1.2, 2.5, 3.7),
    ("Web Admins", 1.0, 1.8, 4.1),
]

task_heatmap = [
    ("Web Devs", 4.15, 1.0, 2.1, 3.2, 1.3, 2.5),
    ("DBA", 3.88, 1.1, 2.3, 3.5, 2.0, 2.8),
    ("Data Entry", 3.95, 1.2, 1.5, 1.8, 1.5, 2.0),
    ("QA/Testers", 3.75, 1.3, 2.8, 3.8, 1.8, 3.0),
    ("BI Analysts", 3.82, 1.2, 2.5, 3.6, 2.5, 2.8),
    ("Programmers", 3.68, 1.5, 2.8, 4.0, 2.2, 3.2),
    ("Web Admins", 4.02, 1.0, 2.0, 3.0, 1.8, 2.5),
    ("Net/SysAdmin", 3.42, 2.0, 3.0, 3.8, 3.0, 3.5),
    ("Support Spec.", 3.55, 2.5, 3.2, 3.0, 3.5, 3.2),
    ("Sys Analysts", 3.48, 1.5, 3.5, 4.2, 2.8, 3.5),
    ("InfoSec", 3.25, 3.0, 3.5, 4.5, 2.5, 4.0),
    ("IT PM", 3.18, 1.8, 3.8, 3.5, 3.8, 3.8),
    ("QC Mgrs", 3.30, 2.2, 3.2, 4.0, 3.2, 3.5),
    ("Sys Engineers", 3.12, 2.8, 3.8, 4.5, 4.0, 4.2),
    ("CS Researchers", 3.05, 2.8, 4.0, 4.8, 4.2, 4.5),
    ("Mgmt Analysts", 2.92, 2.0, 3.5, 3.8, 3.5, 3.8),
    ("IS Managers", 2.85, 2.2, 3.8, 4.0, 4.0, 4.2),
]

# FIGURE 6: Future AI
radar_data = [
    ("Code Generation", 4.2, 4.8), ("Bug Detection", 3.8, 4.5),
    ("System Design", 2.6, 3.8), ("Security Analysis", 3.1, 4.2),
    ("Data Processing", 4.0, 4.7), ("DevOps Auto", 3.5, 4.5),
    ("Testing Auto", 3.9, 4.6), ("NLP/Docs", 3.7, 4.5),
]

gap_data_list = [
    ("System Design", 1.2), ("Security Analysis", 1.1), ("DevOps Auto", 1.0),
    ("NLP/Docs", 0.8), ("Testing Auto", 0.7), ("Data Processing", 0.7),
    ("Bug Detection", 0.7), ("Code Generation", 0.6),
]

knowledge_reqs = [
    ("Tư duy hệ thống & kiến trúc", 5.0, 5.0, 0.5),
    ("Domain knowledge chuyên sâu", 4.8, 4.5, 0.5),
    ("Kiểm tra & validate output AI", 4.5, 5.0, 1.0),
    ("Giao tiếp & quản lý", 4.0, 4.5, 0.8),
    ("Học liên tục & thích nghi", 3.8, 5.0, 0.5),
    ("Prompt engineering", 3.5, 4.8, 1.5),
    ("Quản lý rủi ro & đạo đức AI", 3.0, 4.8, 0.3),
]

# ══════════════════════════════════════════════════════════════════════
# MATPLOTLIB STYLE
# ══════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.edgecolor': '#E2E8F0',
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.color': '#E2E8F0',
    'xtick.color': '#64748B',
    'ytick.color': '#64748B',
    'axes.labelcolor': '#64748B',
    'text.color': '#0F172A',
})

COLORS = {
    'blue': '#3B82F6', 'green': '#10B981', 'amber': '#F59E0B',
    'red': '#EF4444', 'purple': '#8B5CF6', 'cyan': '#06B6D4',
    'navy': '#131B2E', 'light_blue': '#60A5FA', 'pale_blue': '#93C5FD',
    'gray': '#94A3B8', 'light_gray': '#CBD5E1', 'bg': '#F1F5F9',
}

# ══════════════════════════════════════════════════════════════════════
# HELPER: KPI card HTML
# ══════════════════════════════════════════════════════════════════════
def kpi_card(icon, label, value, subtitle="", color="#3B82F6"):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-label">{label}</div>
        {f'<div class="kpi-sub">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def insight_box(content, accent="#3B82F6"):
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-bar" style="background:{accent}"></div>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)

def metric_mini(value, label, color="#3B82F6"):
    st.markdown(f"""
    <div class="metric-mini">
        <div class="value" style="color:{color}">{value}</div>
        <div class="label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:8px 0 20px 8px;">
        <span style="font-size:28px;">🤖</span>
        <div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:16px;font-weight:700;color:white;">AI Agent Dashboard</div>
            <div style="font-size:11px;color:#94A3B8;">Computer Science 2025</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "",
        [
            "📈 Executive Summary",
            "🌍 1. AI Landscape Overview",
            "🧠 2. Trust Gap",
            "❤️ 3. Motivation Analysis",
            "💬 4. LLM Usage",
            "⚡ 5. Task Characteristics",
            "🚀 6. Future AI Demand",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:#64748B;padding:0 8px;">
        Dữ liệu: O*NET 2025<br>
        Đối tượng: 632 tác vụ KHMT<br>
        Phân tích: Expert vs Worker
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SECTION 0: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════
if page == "📈 Executive Summary":
    st.markdown('<div class="section-header"><h2>📈 EXECUTIVE SUMMARY</h2><p>Tổng quan thực trạng AI trong ngành Khoa học Máy tính năm 2025</p></div>', unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("🤖", "LLM Adoption", "89%", "+5% so với 2024", "#3B82F6")
    with c2: kpi_card("⚡", "Daily Usage", "65.5%", "Sử dụng hằng ngày", "#10B981")
    with c3: kpi_card("🔒", "Trust Gap", "30.5%", "Tỉ lệ kháng cự", "#F59E0B")
    with c4: kpi_card("📉", "F1 Score", "0.170", "Expert vs Worker", "#8B5CF6")

    st.markdown("### 💡 6 Insight Tổng Quan")
    insights = [
        ("🚀 Insight 1", "AI được sử dụng rộng rãi nhưng tập trung chủ yếu vào các công việc đơn giản, lặp đi lặp lại.", "#3B82F6"),
        ("🔒 Insight 2", "Khoảng cách niềm tin (Trust Gap) giữa con người và AI vẫn còn rất lớn, đặc biệt ở khâu quyết định.", "#EF4444"),
        ("👨‍💻 Insight 3", "Con người vẫn chiếm ưu thế tuyệt đối trong các công việc đòi hỏi chuyên môn sâu và tư duy chiến lược.", "#10B981"),
        ("⚙️ Insight 4", "AI hiện tại thuần túy đóng vai trò là một công cụ giúp tăng năng suất (Co-pilot), chưa thể tự chủ.", "#8B5CF6"),
        ("📈 Insight 5", "Nhu cầu tương lai chuyển dịch cực mạnh sang Security Analysis và System Design.", "#F59E0B"),
        ("🏗️ Insight 6", "Kỹ sư tương lai sẽ phải tập trung vào kiến trúc hệ thống (Architecture) và kiểm định AI (Validation).", "#06B6D4"),
    ]
    cols = st.columns(2)
    for i, (title, text, color) in enumerate(insights):
        with cols[i % 2]:
            insight_box(f"<strong>{title}</strong><br>{text}", color)

    st.markdown("""
    <div class="guide-box">
        <h4>📋 Hướng dẫn sử dụng</h4>
        <p style="font-size:13px;color:#64748B;">Sử dụng thanh điều hướng bên trái để khám phá từng Dashboard chi tiết. 
        Mỗi mục bao gồm biểu đồ tương tác, dữ liệu chi tiết và nhận xét chuyên sâu từ phân tích dữ liệu O*NET 2025.</p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SECTION 1: AI LANDSCAPE OVERVIEW
# ══════════════════════════════════════════════════════════════════════
elif page == "🌍 1. AI Landscape Overview":
    st.markdown('<div class="section-header"><h2>🌍 DASHBOARD 1: AI Landscape Overview</h2><p>Đánh giá AI ảnh hưởng đến nghề nghiệp nào nhiều nhất trong ngành KHMT</p></div>', unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("📊", "Tác vụ KHMT được đánh giá", "632", "", "#3B82F6")
    with c2: kpi_card("🤖", "Năng lực AI trung bình", "3.45/5", "", "#10B981")
    with c3: kpi_card("⚡", "Mong muốn tự động hoá", "3.22/5", "", "#F59E0B")
    with c4: kpi_card("💬", "Người KHMT đã dùng LLM", "89%", "", "#8B5CF6")

    tab1, tab2, tab3 = st.tabs(["Năng lực tự động hoá", "Mong muốn tự động hoá", "Scatter: Expert vs Desire"])

    with tab1:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Năng Lực Tự Động Hoá Của AI Theo Nghề")

            names = [x[0] for x in automation_capacity]
            scores = [x[1] for x in automation_capacity]
            colors_bar = ['#3B82F6' if s >= 3.8 else '#60A5FA' if s >= 3.3 else '#93C5FD' for s in scores]

            fig, ax = plt.subplots(figsize=(9, 8))
            y_pos = range(len(names))
            bars = ax.barh(y_pos, scores, color=colors_bar, height=0.55, zorder=3)
            ax.axvline(3.45, color='#EAB308', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.text(3.45, len(names) - 0.3, 'TB: 3.45', fontsize=9, color='#EAB308', fontweight='bold', ha='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=10)
            ax.set_xlim(0, 5)
            ax.invert_yaxis()
            ax.set_xlabel('Điểm (1–5)', fontsize=11)
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("##### 📌 Nhận xét")
            insight_box("<strong>Web Developers</strong> có khả năng tự động hóa cao nhất (4.15/5) trong toàn bộ ngành KHMT.", "#3B82F6")
            insight_box("Nhóm làm việc với <strong>Database và BI</strong> (Business Intelligence) đứng top đầu khả năng thay thế.", "#10B981")
            insight_box("Nhóm <strong>Quản lý (IS Managers, Mgmt Analysts)</strong> có nguy cơ bị thay thế thấp hơn hẳn do yêu cầu tương tác con người cao.", "#F59E0B")

    with tab2:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Phân Phối Mong Muốn Tự Động Hoá")

            names = [x[0] for x in automation_desire]
            values = [x[1] for x in automation_desire]
            colors_bar = ['#10B981' if v >= 3.5 else '#F59E0B' if v >= 2.8 else '#EF4444' for v in values]

            fig, ax = plt.subplots(figsize=(9, 7))
            ax.barh(names, values, color=colors_bar, height=0.5, zorder=3)
            ax.set_xlim(0, 5)
            ax.set_xlabel('Median Mong Muốn (1–5)', fontsize=11)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("##### 📌 Nhận xét")
            insight_box("Người lao động <strong>RẤT MUỐN</strong> tự động hóa các công việc lặp lại, tốn thời gian — đặc biệt nhóm Data Entry & Web Dev.", "#10B981")
            insight_box("<strong>KHÔNG MUỐN</strong> AI can thiệp hay tham gia vào các quyết định mang tính chiến lược hoặc liên quan trực tiếp đến con người.", "#EF4444")

    with tab3:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Khả Năng AI vs Nhu Cầu Người Lao Động")

            fig, ax = plt.subplots(figsize=(9, 7))
            for name, x, y, color in scatter_expert_desire:
                ax.scatter(x, y, c=color, s=100, edgecolors='white', linewidth=2, zorder=5)
                ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 8),
                           fontsize=7, ha='center', color='#475569')

            ax.axhline(3.2, color='#CBD5E1', linestyle='--', linewidth=1, alpha=0.7)
            ax.axvline(3.45, color='#CBD5E1', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_xlabel('Năng lực AI (Expert)', fontsize=11)
            ax.set_ylabel('Mong muốn tự động hoá', fontsize=11)
            ax.set_xlim(2.5, 4.5)
            ax.set_ylim(2, 4.5)
            ax.grid(alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            legend_elements = [
                mpatches.Patch(color='#10B981', label='SẴN SÀNG'),
                mpatches.Patch(color='#F59E0B', label='KỲ VỌNG'),
                mpatches.Patch(color='#3B82F6', label='KHÁNG CỰ'),
                mpatches.Patch(color='#EF4444', label='VÙNG CON NGƯỜI'),
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("##### 📌 Nhận xét")
            insight_box("Tồn tại những nghề nghiệp AI có thể làm rất tốt nhưng con người lại <strong>kháng cự</strong>, không muốn giao việc (Trust Gap).", "#F59E0B")
            insight_box('Nhóm <strong>"SẴN SÀNG"</strong> (xanh lá) là các nghề kỹ thuật triển khai: Web Dev, DBA, Data Entry.', "#10B981")
            insight_box("<strong>Kết luận:</strong> AI đang làm thay đổi mạnh mẽ nhất ở các nghề kỹ thuật triển khai trực tiếp (Coders, Testers).", "#EF4444")

# ══════════════════════════════════════════════════════════════════════
# SECTION 2: TRUST GAP
# ══════════════════════════════════════════════════════════════════════
elif page == "🧠 2. Trust Gap":
    st.markdown('<div class="section-header"><h2>🧠 DASHBOARD 2: Expert vs Worker (Trust Gap)</h2><p>Phân tích sự sai lệch giữa đánh giá của chuyên gia và kỳ vọng của người lao động</p></div>', unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_mini("30.5%", "Trust Gap (Kháng cự)", "#EF4444")
    with c2: metric_mini("0.170", "F1 Score", "#8B5CF6")
    with c3: metric_mini("0.639", "Precision", "#3B82F6")
    with c4: metric_mini("0.562", "Accuracy", "#10B981")

    tab1, tab2, tab3 = st.tabs(["Ma trận 3×3", "Ma trận nhị phân 2×2", "Classification Metrics"])

    with tab1:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Ma Trận 3×3: Thấp / Trung Bình / Cao")

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('off')

            rows = ["Thấp (1-2)", "Trung bình (3)", "Cao (4-5)"]
            cols = ["Thấp (1-2)", "Trung bình (3)", "Cao (4-5)"]
            color_map = {
                ("Thấp (1-2)", "Thấp (1-2)"): "#DBEAFE", ("Thấp (1-2)", "Trung bình (3)"): "#EFF6FF",
                ("Thấp (1-2)", "Cao (4-5)"): "#FEF3C7", ("Trung bình (3)", "Thấp (1-2)"): "#EFF6FF",
                ("Trung bình (3)", "Trung bình (3)"): "#BBF7D0", ("Trung bình (3)", "Cao (4-5)"): "#FDE68A",
                ("Cao (4-5)", "Thấp (1-2)"): "#FECACA", ("Cao (4-5)", "Trung bình (3)"): "#FDE68A",
                ("Cao (4-5)", "Cao (4-5)"): "#86EFAC",
            }

            table_data = {}
            for e, w, cnt, pct in confusion_3x3:
                table_data[(e, w)] = (cnt, pct)

            # Header
            ax.text(0.5, 3.6, "Mức độ mong muốn của Worker →", fontsize=11, fontweight='bold', ha='center', va='center')
            for ci, col in enumerate(cols):
                ax.text(ci + 1.5, 3.1, col, fontsize=9, fontweight='bold', ha='center', va='center')

            for ri, row in enumerate(rows):
                ax.text(0.1, 2.5 - ri, row, fontsize=9, fontweight='bold', ha='right', va='center')

            for ri, row in enumerate(rows):
                for ci, col in enumerate(cols):
                    cnt, pct = table_data.get((row, col), (0, 0))
                    x, y = ci + 1.5, 2.5 - ri
                    bg = color_map.get((row, col), "#F1F5F9")
                    rect = FancyBboxPatch((x - 0.4, y - 0.35), 0.8, 0.7,
                                          boxstyle="round,pad=0.1", facecolor=bg,
                                          edgecolor="#10B981" if ri == ci else "#E2E8F0",
                                          linewidth=2 if ri == ci else 0.5)
                    ax.add_patch(rect)
                    ax.text(x, y + 0.1, str(cnt), fontsize=18, fontweight='bold', ha='center', va='center', color='#0F172A')
                    ax.text(x, y - 0.18, f"{pct}%", fontsize=10, ha='center', va='center', color='#64748B')

            ax.set_xlim(0.5, 4.5)
            ax.set_ylim(0, 4)
            ax.text(2.5, 0.2, "🟢 Ô viền xanh = đường chéo (đồng thuận)", fontsize=10, ha='center', color='#64748B')

            ax.text(0, 2.5, "Expert ↓", fontsize=9, fontweight='bold', ha='right', rotation=90, va='center')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("##### 📌 Nhận xét")
            insight_box("Tồn tại <strong>sai lệch lớn</strong> giữa đánh giá năng lực AI của chuyên gia và mong muốn thực tế của người lao động.", "#3B82F6")
            insight_box("Ô đồng thuận lớn nhất nằm ở góc <strong>Cao–Cao (30.5%)</strong>, cho thấy một phần lớn đã sẵn sàng.", "#10B981")
            insight_box("Tuy nhiên, vùng <strong>ngoài đường chéo</strong> vẫn chiếm phần lớn — sự bất đồng quan điểm rất rõ rệt.", "#EF4444")

    with tab2:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Ma Trận 2×2 Nhị Phân (Ngưỡng ≥ 4)")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.axis('off')

            labels_2x2 = [
                ("AI KHÔNG thể\nWorker KHÔNG muốn", 0, 0, "#F1F5F9"),
                ("AI KHÔNG thể\nWorker Muốn", 1, 0, "#3B82F6"),
                ("AI CÓ thể\nWorker KHÔNG muốn", 0, 1, "#F59E0B"),
                ("AI CÓ thể\nWorker Muốn", 1, 1, "#10B981"),
            ]

            for label, cx, cy, color in labels_2x2:
                item = binary_matrix[[i for i, b in enumerate(binary_matrix) if b[3] == color][0]]
                x, y = cx * 1.2 + 0.6, cy * 1.2 + 0.6
                rect = FancyBboxPatch((x - 0.45, y - 0.45), 0.9, 0.9,
                                      boxstyle="round,pad=0.1", facecolor=color + "25",
                                      edgecolor=color, linewidth=2)
                ax.add_patch(rect)
                ax.text(x, y + 0.2, str(item[1]), fontsize=24, fontweight='bold', ha='center', va='center', color='#0F172A')
                ax.text(x, y - 0.05, f"{item[2]}%", fontsize=14, fontweight='bold', ha='center', va='center', color='#0F172A')
                ax.text(x, y - 0.28, label, fontsize=8, ha='center', va='top', color='#64748B')

            ax.set_xlim(0, 2.4)
            ax.set_ylim(0, 2.4)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("""
            <div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;font-size:11px;">
                <span>🟢 Đồng thuận: Muốn & Có thể</span>
                <span>⬜ Đồng thuận: Không & Không thể</span>
                <span>🟠 Kháng cự: Có thể nhưng không muốn</span>
                <span>🔵 Khoảng trống: Muốn nhưng AI chưa thể</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("##### 📌 Nhận xét")
            insight_box("<strong>30.5%</strong> nhiệm vụ: AI có đủ năng lực làm được → nhưng con người không muốn giao phó. Đây chính là Trust Gap.", "#F59E0B")
            insight_box("<strong>13.3%</strong> nhiệm vụ: Con người muốn tự động hoá nhưng AI hiện chưa đủ năng lực → Khoảng trống công nghệ.", "#3B82F6")

    with tab3:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Chỉ Số Classification (Expert = Ground Truth, Worker = Prediction)")

            names = [x[0] for x in class_metrics]
            values = [x[1] for x in class_metrics]
            colors_bar = [x[2] for x in class_metrics]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(names, values, color=colors_bar, height=0.5, zorder=3)
            for i, (name, val, _) in enumerate(class_metrics):
                ax.text(val + 0.02, i, f"{val:.3f}", fontsize=14, fontweight='bold', va='center', color='#0F172A')
            ax.axvline(0.5, color='#CBD5E1', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Giá trị', fontsize=11)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("##### 📌 Nhận xét")
            insight_box("<strong>F1 Score = 0.170</strong> — cực thấp. Phản ánh sự bất đồng sâu sắc giữa năng lực AI và mong muốn con người.", "#8B5CF6")
            insight_box("<strong>Niềm tin</strong> mới là rào cản lớn nhất của việc áp dụng AI vào thực tế, chứ không phải năng lực công nghệ.", "#EF4444")
            insight_box("Precision khá cao (0.639) cho thấy khi con người đồng ý thì AI thường đúng là có khả năng.", "#3B82F6")

# ══════════════════════════════════════════════════════════════════════
# SECTION 3: MOTIVATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════
elif page == "❤️ 3. Motivation Analysis":
    st.markdown('<div class="section-header"><h2>❤️ DASHBOARD 3: Motivation Analysis</h2><p>Phân tích lý do muốn / không muốn tự động hoá trong ngành Khoa học Máy tính</p></div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Lý do MUỐN tự động hoá", "Lý do CẦN con người", "Đặc trưng tác vụ theo mức độ"])

    with tab1:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Lý Do Worker MUỐN Tự Động Hoá")
            st.caption("Trong nhóm Automation Desire ≥ 4")

            data_sorted = sorted(reasons_auto, key=lambda x: x[1])
            names = [x[0] for x in data_sorted]
            values = [x[1] for x in data_sorted]
            colors_bar = ['#10B981' if v >= 50 else '#3B82F6' for v in values]

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(names, values, color=colors_bar, height=0.5, zorder=3)
            for i, v in enumerate(values):
                ax.text(v + 0.5, i, f"{v:.1f}%", fontsize=11, fontweight='bold', va='center', color='#0F172A')
            ax.set_xlim(0, 85)
            ax.set_xlabel('%', fontsize=11)
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("##### 📌 Nhận xét")
            insight_box("""<strong>Top 3 lý do:</strong><br>
            🥇 Tiết kiệm thời gian (72.3%)<br>
            🥈 Mở rộng quy mô – Scale (58.1%)<br>
            🥉 Giảm lỗi người dùng (51.7%)""", "#10B981")
            insight_box("Động lực chủ yếu đến từ <strong>hiệu quả vận hành</strong> chứ không phải thay thế con người.", "#3B82F6")

    with tab2:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Lý Do Worker CẦN Yếu Tố Con Người")
            st.caption("Trong nhóm Automation Desire ≤ 2 – Human Agency")

            data_sorted = sorted(reasons_human, key=lambda x: x[1])
            names = [x[0] for x in data_sorted]
            values = [x[1] for x in data_sorted]
            colors_bar = ['#EF4444' if v >= 55 else '#F59E0B' for v in values]

            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.barh(names, values, color=colors_bar, height=0.5, zorder=3)
            for i, v in enumerate(values):
                ax.text(v + 0.5, i, f"{v:.1f}%", fontsize=11, fontweight='bold', va='center', color='#0F172A')
            ax.set_xlim(0, 90)
            ax.set_xlabel('%', fontsize=11)
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("##### 📌 Nhận xét")
            insight_box("""<strong>Top 3 lý do cần con người:</strong><br>
            🥇 Kiến thức chuyên sâu (78.5%)<br>
            🥈 Kiểm soát trực tiếp (65.3%)<br>
            🥉 Human factor — Đạo đức (58.7%)""", "#EF4444")
            insight_box("Người lao động muốn AI <strong>hỗ trợ</strong> chứ không phải <strong>thay thế</strong> hoàn toàn.", "#F59E0B")

    with tab3:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Đặc Trưng Tác Vụ Theo Mức Độ Mong Muốn Tự Động Hoá")

            names = [x[0] for x in task_chars_level]
            low_vals = [x[1] for x in task_chars_level]
            med_vals = [x[2] for x in task_chars_level]
            high_vals = [x[3] for x in task_chars_level]

            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(names))
            w = 0.22
            ax.bar(x - w, low_vals, w, label='Thấp (1-2)', color='#EF4444', zorder=3)
            ax.bar(x, med_vals, w, label='Trung bình (3)', color='#EAB308', zorder=3)
            ax.bar(x + w, high_vals, w, label='Cao (4-5)', color='#10B981', zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=9)
            ax.set_ylabel('Điểm TB (1–5)', fontsize=11)
            ax.set_ylim(0, 5.5)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(axis='y', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("##### 📌 Nhận xét")
            insight_box("Nhóm <strong>muốn tự động hoá cao</strong> (xanh) có yêu cầu vật lý và giao tiếp thấp nhất.", "#10B981")
            insight_box("Nhóm <strong>không muốn tự động hoá</strong> (đỏ) có yêu cầu chuyên môn sâu và tương tác người rất cao.", "#EF4444")
            insight_box("<strong>Kết luận:</strong> Tính chất công việc quyết định trực tiếp mức độ chấp nhận tự động hoá.", "#F59E0B")

# ══════════════════════════════════════════════════════════════════════
# SECTION 4: LLM USAGE
# ══════════════════════════════════════════════════════════════════════
elif page == "💬 4. LLM Usage":
    st.markdown('<div class="section-header"><h2>💬 DASHBOARD 4: LLM Usage Analysis</h2><p>Tần suất sử dụng LLM / AI Agent trong công việc ngành KHMT</p></div>', unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("🤖", "Đã sử dụng LLM", "89%", "", "#3B82F6")
    with c2: kpi_card("📧", "Dùng hằng ngày cho Email", "45%", "", "#10B981")
    with c3: kpi_card("🏗️", "Không dùng cho System Design", "52%", "", "#EF4444")

    tab1, tab2, tab3 = st.tabs(["Heatmap tần suất", "Mức độ quen thuộc", "Thái độ với AI"])

    with tab1:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Tần Suất Sử Dụng LLM Theo Loại Tác Vụ")

            freq_colors = {"daily": "#10B981", "weekly": "#3B82F6", "monthly": "#F59E0B", "never": "#94A3B8"}
            freq_keys = ["daily", "weekly", "monthly", "never"]
            freq_labels = ["Hằng ngày", "Hằng tuần", "Hằng tháng", "Không dùng"]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.axis('off')

            tasks = [h[0] for h in llm_heatmap]

            # Header
            ax.text(-0.1, len(tasks) + 0.5, "Tác vụ", fontsize=9, fontweight='bold', ha='left', va='center', color='#64748B')
            for ci, (key, label) in enumerate(zip(freq_keys, freq_labels)):
                ax.text(ci + 1.2, len(tasks) + 0.5, label, fontsize=9, fontweight='bold', ha='center', va='center', color=freq_colors[key])

            for ri, (task, d, w, m, n) in enumerate(llm_heatmap):
                y = len(tasks) - 0.5 - ri
                ax.text(-0.1, y, task, fontsize=9, ha='left', va='center', color='#0F172A')

                vals = [d, w, m, n]
                for ci, (key, val) in enumerate(zip(freq_keys, vals)):
                    x = ci + 1.2
                    alpha = min(val / 50, 1.0)
                    r = int(int(freq_colors[key][1:3], 16) * (1 - alpha * 0.5))
                    g = int(int(freq_colors[key][3:5], 16) * (1 - alpha * 0.5))
                    b = int(int(freq_colors[key][5:7], 16) * (1 - alpha * 0.5))
                    bg = f"#{r:02x}{g:02x}{b:02x}"
                    text_color = '#0F172A' if val > 30 else '#64748B'

                    rect = FancyBboxPatch((x - 0.38, y - 0.35), 0.76, 0.7,
                                          boxstyle="round,pad=0.06", facecolor=bg,
                                          edgecolor='none')
                    ax.add_patch(rect)
                    ax.text(x, y, f"{val}%", fontsize=10, fontweight='bold', ha='center', va='center', color=text_color)

            ax.set_xlim(-0.5, 4.5)
            ax.set_ylim(0, len(tasks) + 1)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("##### 📌 Nhận xét")
            insight_box("<strong>Sử dụng nhiều nhất:</strong> Tra cứu thông tin, Giao tiếp/Email, Chỉnh sửa văn bản — tác vụ đơn giản, lặp lại.", "#10B981")
            insight_box("<strong>Ít sử dụng nhất:</strong> Thiết kế hệ thống, Ra quyết định — tác vụ đòi hỏi tư duy chiến lược.", "#EF4444")
            insight_box("<strong>Kết luận:</strong> LLM hiện tại đóng vai trò trợ lý cá nhân cho các tác vụ routine.", "#3B82F6")

    with tab2:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Mức Độ Quen Thuộc Với LLM")

            labels = [x[0] for x in llm_familiarity]
            sizes = [x[1] for x in llm_familiarity]
            colors_pie = [x[2] for x in llm_familiarity]

            fig, ax = plt.subplots(figsize=(7, 6))
            wedges, texts, autotexts = ax.pie(
                sizes, labels=None, colors=colors_pie, autopct='%1.0f%%',
                startangle=90, pctdistance=0.75, wedgeprops={'edgecolor': 'white', 'linewidth': 3}
            )
            for at in autotexts:
                at.set_fontsize(13)
                at.set_fontweight('bold')
                at.set_color('#0F172A')

            centre_circle = plt.Circle((0, 0), 0.40, fc='white')
            ax.add_artist(centre_circle)

            ax.legend(wedges, labels, loc='lower center', ncol=2, fontsize=10, framealpha=0.9)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("##### 📌 Nhận xét")
            insight_box("<strong>52%</strong> người dùng đã sử dụng LLM thường xuyên trong công việc.", "#10B981")
            insight_box("<strong>80%</strong> (Thường xuyên + Có kinh nghiệm) đã tiếp xúc trực tiếp với LLM.", "#3B82F6")
            insight_box("Chỉ <strong>5%</strong> chưa từng nghe về LLM — tỉ lệ nhận biết rất cao trong ngành KHMT.", "#EF4444")

    with tab3:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Thái Độ Với AI Trong Công Việc")

            attitude_labels = ["Hoàn toàn đồng ý", "Phần nào đồng ý", "Trung lập", "Phần nào KĐY", "Không đồng ý"]
            attitude_colors = ["#10B981", "#68D391", "#EAB308", "#F59E0B", "#EF4444"]

            names = [a[0] for a in ai_attitudes]
            data_matrix = np.array([[a[1], a[2], a[3], a[4], a[5]] for a in ai_attitudes])

            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(names))
            w = 0.6
            bottom = np.zeros(len(names))

            for i in range(5):
                ax.bar(x, data_matrix[:, i], w, bottom=bottom, label=attitude_labels[i],
                       color=attitude_colors[i], zorder=3)
                bottom += data_matrix[:, i]

            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=9)
            ax.set_ylabel('%', fontsize=11)
            ax.legend(fontsize=8, loc='upper right', ncol=2)
            ax.grid(axis='y', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("##### 📌 Nhận xét")
            insight_box("<strong>70%</strong> đồng ý AI nên xử lý công việc tẻ nhạt — đây là ứng dụng được chấp nhận rộng rãi nhất.", "#10B981")
            insight_box("Với công việc quan trọng, tỉ lệ <strong>phản đối lên tới 35%</strong> — rõ ràng có ranh giới.", "#EF4444")

# ══════════════════════════════════════════════════════════════════════
# SECTION 5: TASK CHARACTERISTICS
# ══════════════════════════════════════════════════════════════════════
elif page == "⚡ 5. Task Characteristics":
    st.markdown('<div class="section-header"><h2>⚡ DASHBOARD 5: Task Characteristics</h2><p>Năng lực AI theo đặc trưng tác vụ — ưu điểm và nhược điểm</p></div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2 = st.tabs(["Scatter: Physical vs Interpersonal", "Heatmap đặc trưng tác vụ"])

    def get_cap_color(val):
        if val >= 4: return "#10B981"
        if val >= 3.5: return "#3B82F6"
        if val >= 3: return "#F59E0B"
        return "#EF4444"

    def get_cell_color_hm(val):
        if val >= 4: return "#FCA5A5"
        if val >= 3.5: return "#FECACA"
        if val >= 3: return "#FEF3C7"
        if val >= 2: return "#DBEAFE"
        return "#BBF7D0"

    with tab1:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Năng Lực AI: Physical vs Interpersonal")
            st.caption("🟢 Xanh = AI làm tốt | 🔴 Đỏ = AI làm kém")

            fig, ax = plt.subplots(figsize=(9, 7))
            xs = [s[1] for s in scatter_phys_inter]
            ys = [s[2] for s in scatter_phys_inter]
            colors_scatter = [get_cap_color(s[3]) for s in scatter_phys_inter]

            ax.scatter(xs, ys, c=colors_scatter, s=80, edgecolors='white', linewidth=2, zorder=5)
            for s in scatter_phys_inter:
                ax.annotate(s[0], (s[1], s[2]), textcoords="offset points", xytext=(0, 6),
                           fontsize=7, ha='center', color='#475569')

            ax.set_xlabel('Physical Action Requirement', fontsize=11)
            ax.set_ylabel('Interpersonal Comm.', fontsize=11)
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 5)
            ax.grid(alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("##### 📌 Nhận xét")
            insight_box("AI <strong>KÉM</strong> khi tác vụ yêu cầu tương tác con người cao (interpersonal) và hành động vật lý (physical) cao.", "#EF4444")
            insight_box("AI <strong>TỐT</strong> khi tác vụ ít yêu cầu vật lý và giao tiếp — khu vực góc trái dưới biểu đồ.", "#10B981")
            insight_box("Nghề <strong>IS Managers</strong> và <strong>CS Researchers</strong> (góc phải trên) có đặc thù con người mà AI khó thay thế.", "#3B82F6")

    with tab2:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Heatmap Đặc Trưng Tác Vụ Theo Nghề KHMT")
            st.caption("Sắp xếp theo Năng Lực AI giảm dần | 🟢 Xanh = AI tốt | 🔴 Đỏ = AI khó")

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')

            col_keys = ["automation", "physical", "uncertainty", "expertise", "interpersonal", "agency"]
            col_labels = ["Năng lực AI", "Vật lý", "Không chắc chắn", "Chuyên môn", "Giao tiếp", "Human Agency"]

            # Header
            ax.text(-0.2, len(task_heatmap) + 0.5, "Nghề", fontsize=9, fontweight='bold', ha='left', va='center', color='#64748B')
            for ci, label in enumerate(col_labels):
                ax.text(ci + 1.2, len(task_heatmap) + 0.5, label, fontsize=8, fontweight='bold', ha='center', va='center', color='#64748B')

            for ri, row in enumerate(task_heatmap):
                y = len(task_heatmap) - 0.5 - ri
                ax.text(-0.2, y, row[0], fontsize=8, ha='left', va='center', color='#0F172A')

                for ci, key in enumerate(col_keys):
                    val = row[ci + 1]
                    x = ci + 1.2
                    if key == "automation":
                        bg = "#BBF7D0" if val >= 3.8 else "#DBEAFE" if val >= 3.3 else "#FEF3C7" if val >= 3 else "#FECACA"
                    else:
                        bg = get_cell_color_hm(val)

                    rect = FancyBboxPatch((x - 0.38, y - 0.3), 0.76, 0.6,
                                          boxstyle="round,pad=0.05", facecolor=bg, edgecolor='none')
                    ax.add_patch(rect)
                    ax.text(x, y, f"{val:.2f}", fontsize=9, fontweight='bold', ha='center', va='center', color='#0F172A')

            ax.set_xlim(-0.5, 6.5)
            ax.set_ylim(0, len(task_heatmap) + 1)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("##### 📌 Nhận xét")
            insight_box("AI <strong>mạnh</strong> khi tác vụ có tính: Structured, Repetitive, Predictable — ít yêu cầu vật lý và giao tiếp.", "#10B981")
            insight_box("AI <strong>yếu</strong> khi tác vụ cần: Chuyên môn sâu, Giao tiếp người, Xử lý tình huống không chắc chắn.", "#EF4444")
            insight_box("<strong>Kết luận:</strong> Đặc điểm nhiệm vụ quyết định trực tiếp mức độ tự động hoá — không phải tên nghề.", "#F59E0B")

# ══════════════════════════════════════════════════════════════════════
# SECTION 6: FUTURE AI DEMAND
# ══════════════════════════════════════════════════════════════════════
elif page == "🚀 6. Future AI Demand":
    st.markdown('<div class="section-header"><h2>🚀 DASHBOARD 6: Future AI Demand</h2><p>Nhu cầu AI Agent tương lai & yêu cầu tri thức người dùng – Ngành Khoa học Máy tính</p></div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Radar: Hiện tại vs Tương lai", "Gap Analysis", "Tri thức người dùng"])

    with tab1:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Năng Lực AI: Hiện Tại (2025) vs Dự Kiến (2028)")

            labels = [r[0] for r in radar_data]
            current = [r[1] for r in radar_data]
            future = [r[2] for r in radar_data]

            num_vars = len(labels)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]
            current_vals = current + current[:1]
            future_vals = future + future[:1]

            fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
            ax.fill(angles, current_vals, alpha=0.2, color='#3B82F6')
            ax.plot(angles, current_vals, 'o-', linewidth=2, color='#3B82F6', label='Hiện tại 2025')
            ax.fill(angles, future_vals, alpha=0.12, color='#10B981')
            ax.plot(angles, future_vals, 'o--', linewidth=2, color='#10B981', label='Dự kiến 2028', dashes=(5, 3))
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=9, color='#0F172A')
            ax.set_ylim(0, 5.5)
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=8, color='#64748B')
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("##### 📌 Nhận xét")
            insight_box("""<strong>Tăng mạnh nhất:</strong><br>
            🥇 System Design (+1.2)<br>
            🥈 Security Analysis (+1.1)<br>
            🥉 DevOps Automation (+1.0)""", "#10B981")
            insight_box("Code Generation đã ở mức rất cao (4.2) → tăng thêm chỉ +0.6 — đã gần bão hoà.", "#3B82F6")
            insight_box("Hầu hết các lĩnh vực đều hướng tới mức <strong>4.5+</strong> vào 2028 — AI sẽ toàn diện hơn rất nhiều.", "#F59E0B")

    with tab2:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Ưu Tiên Cải Thiện AI Agent (Gap Analysis)")
            st.caption("🟠 Cam = ưu tiên cao | 🟡 Vàng = TB | 🟢 Xanh = ổn")

            gap_sorted = sorted(gap_data_list, key=lambda x: x[1], reverse=True)
            names = [x[0] for x in gap_sorted]
            gaps = [x[1] for x in gap_sorted]
            colors_gap = ['#F59E0B' if g >= 1.0 else '#EAB308' if g >= 0.7 else '#10B981' for g in gaps]

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(names, gaps, color=colors_gap, height=0.5, zorder=3)
            for i, g in enumerate(gaps):
                ax.text(g + 0.02, i, f"+{g:.1f}", fontsize=11, fontweight='bold', va='center', color='#0F172A')
            ax.set_xlim(0, 1.6)
            ax.set_xlabel('Gap (2028 – 2025)', fontsize=11)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("##### 📌 Nhận xét")
            insight_box("<strong>System Design</strong> cần cải thiện nhiều nhất (+1.2) — hiện là điểm yếu lớn nhất của AI Agent.", "#F59E0B")
            insight_box("<strong>Security Analysis</strong> là ưu tiên thứ hai (+1.1) — nhu cầu bảo mật AI đang tăng vọt.", "#EF4444")
            insight_box("<strong>Code Generation</strong> gần đạt ngưỡng bão hoà — chỉ cần cải thiện +0.6 là đạt mức tối đa.", "#10B981")

    with tab3:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Yêu Cầu Tri Thức Người Dùng Khi AI Đảm Nhiệm Tác Vụ Routine")
            st.caption("Con người tập trung vào tư duy bậc cao — AI xử lý routine tasks")

            names = [x[0] for x in knowledge_reqs]
            now_vals = [x[1] for x in knowledge_reqs]
            future_vals = [x[2] for x in knowledge_reqs]
            ai_vals = [x[3] for x in knowledge_reqs]

            fig, ax = plt.subplots(figsize=(9, 5))
            y_pos = range(len(names))
            h = 0.22
            ax.barh([y + h for y in y_pos], now_vals, h, label='Hiện tại 2025', color='#3B82F6', zorder=3)
            ax.barh(y_pos, future_vals, h, label='Tương lai 2028', color='#10B981', zorder=3)
            ax.barh([y - h for y in y_pos], ai_vals, h, label='AI có thể thay thế', color='#EF4444', zorder=3)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlim(0, 6)
            ax.set_xlabel('Điểm (1–5)', fontsize=11)
            ax.invert_yaxis()
            ax.legend(fontsize=9, loc='lower right')
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("##### 📌 Nhận xét")
            insight_box("""<strong>Kỹ năng quan trọng nhất:</strong><br>
            ⭐ Tư duy hệ thống & Kiến trúc<br>
            ⭐ Kiểm tra & Validate output AI<br>
            ⭐ Domain knowledge chuyên sâu""", "#3B82F6")
            insight_box("AI gần như <strong>không thể thay thế</strong> các kỹ năng: Quản lý rủi ro đạo đức, Tư duy hệ thống, Domain knowledge.", "#EF4444")
            insight_box("""<strong>Kết luận:</strong> AI tương lai sẽ chuyển từ:<br>
            <span style="display:inline-block;background:#F1F5F9;border-radius:8px;padding:4px 12px;font-weight:600;">Code Generator</span> 
            → 
            <span style="display:inline-block;background:#DBEAFE;color:#3B82F6;border-radius:8px;padding:4px 12px;font-weight:600;">AI Architect Assistant</span>""", "#10B981")