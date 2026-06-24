# TÁC ĐỘNG CỦA AI AGENT ĐỐI VỚI CÁC NGÀNH MÁY TÍNH
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/657ddab1-fb03-49b2-9e13-2f48cf32324f" />

---

# MỘT SỐ LƯU Ý

### 1. File app.ipynb là file code thuần
### 2. File app_streamlit.py là file kết hợp với streamlit để trực quan hóa trên web, để xem vui lòng tải code xuống và chạy lệnh
```bash
streamlit run app_streamlit.py
```
### 3. Link https://btcn.base44.app/ sử dụng vibe coding để thân thiện với người dùng hơn streamlit ( do streamlit có nhiều hạn chế đối với nhu cầu sử dụng với người dùng hiện tại.
---

## 📌 Giới thiệu dự án

Dự án trực quan hóa dữ liệu nhằm phân tích tác động của Trí tuệ nhân tạo (AI) đối với các nghề nghiệp thuộc lĩnh vực Khoa học Máy tính (Computer Science).

Dashboard tập trung khám phá:

- Mức độ tự động hóa của AI trong các công việc CNTT.
- Sự khác biệt giữa đánh giá của chuyên gia và người lao động.
- Lý do người lao động mong muốn hoặc không mong muốn AI thay thế công việc.
- Các hình thức ứng dụng của mô hình ngôn ngữ lớn (LLM).
- Đặc trưng nhiệm vụ phù hợp với AI.
- Xu hướng phát triển và nhu cầu AI trong tương lai.

---

# 🎯 Mục tiêu nghiên cứu

Nghiên cứu được xây dựng nhằm trả lời các câu hỏi:

1. AI đang tác động đến các nghề nghiệp trong lĩnh vực Khoa học Máy tính như thế nào?
2. Người lao động và chuyên gia có đánh giá giống nhau về khả năng tự động hóa của AI không?
3. Vì sao người lao động muốn hoặc không muốn AI thay thế công việc?
4. LLM hiện đang được sử dụng như thế nào trong công việc?
5. AI làm tốt nhất những loại nhiệm vụ nào?
6. Nhu cầu và vai trò của AI trong tương lai sẽ thay đổi ra sao?

---

# 🧠 Cách tư duy xây dựng Dashboard

Dashboard được thiết kế theo tư duy kể chuyện dữ liệu (Data Storytelling).

## Giai đoạn 1 – Hiện trạng

**Câu hỏi**

> AI hiện đang tác động đến đâu trong ngành CNTT?

**Biểu đồ**

- Figure 1: AI Landscape Overview

**Mục đích**

Xác định mức độ ảnh hưởng của AI tới từng nhóm nghề nghiệp.

---

## Giai đoạn 2 – Khoảng cách nhận thức

**Câu hỏi**

> Người lao động có đánh giá đúng mức độ thay thế của AI hay không?

**Biểu đồ**

- Figure 2: Expert vs Worker Comparison
- Confusion Matrix

**Mục đích**

Phân tích sự khác biệt giữa góc nhìn chuyên gia và người lao động.

---

## Giai đoạn 3 – Động lực và rào cản

**Câu hỏi**

> Tại sao người lao động muốn hoặc không muốn AI tham gia công việc?

**Biểu đồ**

- Figure 3: Motivation Analysis

**Mục đích**

Khám phá động lực thúc đẩy và những lo ngại liên quan đến AI.

---

## Giai đoạn 4 – Hành vi sử dụng AI

**Câu hỏi**

> Người lao động đang sử dụng AI như thế nào?

**Biểu đồ**

- Figure 4: LLM Usage Analysis

**Mục đích**

Xác định các mục đích sử dụng AI phổ biến trong thực tế.

---

## Giai đoạn 5 – Năng lực AI

**Câu hỏi**

> AI hoạt động hiệu quả nhất với những loại nhiệm vụ nào?

**Biểu đồ**

- Figure 5: Task Characteristics Analysis

**Mục đích**

Xác định các đặc trưng giúp AI đạt hiệu suất cao.

---

## Giai đoạn 6 – Tương lai

**Câu hỏi**

> Những kỹ năng và năng lực AI nào sẽ được yêu cầu trong tương lai?

**Biểu đồ**

- Figure 6: Future AI Demand
- Radar Chart
- Knowledge Analysis

**Mục đích**

Dự báo xu hướng phát triển AI trong giai đoạn tiếp theo.

---

# 🔄 Sơ đồ luồng chức năng

```text
Raw Datasets
        │
        ▼
Data Preparation
        │
        ▼
Computer Science Filtering
        │
        ▼
Metric Calculation
        │
        ▼
Dashboard Visualization
        │
        ▼
Insight Discovery
        │
        ▼
Future Trend Analysis
```

---

# 🖼️ Tổng quan Dashboard

## Figure 1 – AI Landscape Overview

Mô tả bức tranh tổng quan về tác động của AI tới các nghề nghiệp trong lĩnh vực Computer Science.

<img width="1768" height="1085" alt="image" src="https://github.com/user-attachments/assets/621b7a18-c309-4681-a35f-d5a60d6cfd4a" />

---

## Figure 2 – Expert vs Worker Comparison

So sánh đánh giá giữa chuyên gia và người lao động về khả năng tự động hóa.

<img width="1541" height="914" alt="image" src="https://github.com/user-attachments/assets/e6125446-c3dd-4e5e-831e-05662b0fa218" />


---

## Figure 3 – Motivation Analysis

Phân tích lý do người lao động muốn hoặc không muốn AI tham gia công việc.

<img width="1636" height="938" alt="image" src="https://github.com/user-attachments/assets/c5d66a7d-c1d5-4124-93a0-0d7f2a510304" />


---

## Figure 4 – LLM Usage Analysis

Phân tích các hình thức ứng dụng mô hình ngôn ngữ lớn trong công việc.

<img width="1600" height="916" alt="image" src="https://github.com/user-attachments/assets/c1cd2bef-d222-43c3-a830-a3b8d01bf69d" />


---

## Figure 5 – Task Characteristics Analysis

Khám phá những loại nhiệm vụ AI thực hiện hiệu quả nhất.

<img width="1780" height="999" alt="image" src="https://github.com/user-attachments/assets/81f47bab-0838-410b-aa83-125d6d1ed3d3" />


---

## Figure 6 – Future AI Demand Analysis

Dự báo nhu cầu AI và kỹ năng liên quan trong tương lai.

<img width="1808" height="1073" alt="image" src="https://github.com/user-attachments/assets/6ed0c8c0-03a9-43f2-8fa8-0e2e64a3b60c" />


---

# 📈 Kết quả nổi bật

## 1. Tỷ lệ tiếp nhận AI cao nhưng chủ yếu tập trung vào các tác vụ giản đơn
Mặc dù có đến 89% người làm trong ngành Máy tính đã sử dụng LLM trong công việc và 65.5% sử dụng thường xuyên, việc ứng dụng hàng ngày lại bị lệch hẳn về các tác vụ cơ bản. Cụ thể, các tác vụ được dùng hàng ngày nhiều nhất là Giao tiếp/Email (44%) và Tra cứu thông tin (41%). Ngược lại, đối với các tác vụ cốt lõi và phức tạp như Thiết kế hệ thống (System Design), có tới 39% người dùng chọn mức "Không dùng". Điều này cho thấy AI hiện tại đang đóng vai trò như một trợ lý hành chính cá nhân hơn là một kỹ sư chuyên nghiệp.

## 2. Vấn đề niềm tin của con người với AI
Ma trận nhầm lẫn chỉ ra một điểm nghẽn lớn trong việc tự động hóa: Có đến 30.5% tổng số tác vụ mà AI CÓ THỂ tự động hóa, nhưng người lao động KHÔNG MUỐN giao phó cho AI. Sự "kháng cự" này dẫn đến chỉ số F1-Score rất thấp (0.170), phản ánh một khoảng cách lớn về niềm tin. Người lao động dường như chưa sẵn sàng hoặc chưa tin tưởng để nhường lại quyền kiểm soát cho AI ngay cả khi công nghệ đã đáp ứng được yêu cầu kỹ thuật của tác vụ đó.

## 3. Yếu tố con người là cốt lõi đối với các tác vụ có tính chuyên môn và độ bất định cao
Phân tích lý do không muốn tự động hóa và heat map năng lực cho thấy sự thống nhất: AI làm rất kém và con người cũng không muốn dùng AI ở những công việc có độ không chắc chắn cao (Uncertainty), yêu cầu kiến thức chuyên môn sâu (Domain Expertise) và giao tiếp giữa người với người (Human Interaction). Đây là lý do tại sao AI được đánh giá làm tốt nhất ở các vai trò như Web Admins hay Web Devs, nhưng lại đạt điểm rất thấp và bị từ chối tự động hóa ở các vị trí quản lý hoặc yêu cầu nghiệp vụ ngách như IT PM, IS Managers hay Aerospace Engineers. Cụ thể, 35.3% người dùng cần giữ lại yếu tố con người vì "Cần kiến thức chuyên sâu" và 33.3% vì "Cần kiểm soát trực tiếp".

## 4. Động lực cốt lõi của tự động hóa là tối ưu hóa tài nguyên thay vì nâng cao tư duy sáng tạo
Lý do lớn nhất khiến người lao động MUỐN sử dụng AI chủ yếu mang tính thực dụng: Tiết kiệm thời gian (74.8%), Mở rộng quy mô (54.9%), và Giảm lỗi người dùng (52.7%). Họ mong đợi AI giải quyết các tác vụ lặp lại (50.2%) để giải phóng sức lao động. Điều này khẳng định định vị hiện tại của AI trong mắt người lao động là một "công cụ tăng năng suất" (productivity tool) hơn là một đối tác tư duy chiến lược.

## 5. Khoảng trống năng lực tương lai của AI 
Biểu đồ Radar về nhu cầu AI Agent tương lai tiết lộ rằng Code Generation không còn là ưu tiên cải thiện hàng đầu (chỉ cần khoảng cách cải thiện +0.6). Thay vào đó, kỳ vọng của người dùng đang chuyển dịch mạnh mẽ lên các khía cạnh vĩ mô và rủi ro cao hơn. Thiết kế hệ thống (System Design) và Phân tích bảo mật (Security Analysis) là hai mảng có khoảng cách cần cải thiện lớn nhất (lần lượt là +1.2 và +1.1) từ nay đến năm 2028.

## 6. Sự chuyển dịch: Từ "Thực thi" sang "Kiểm định và Kiến trúc"
Khi AI dần đảm nhiệm các tác vụ lặp lại và viết code cơ bản, yêu cầu tri thức đối với người lao động đang thay đổi hoàn toàn. (Hình 6) cho thấy hai kỹ năng được đánh giá có tầm quan trọng tuyệt đối (mức độ 5.0/5.0) là "Tư duy hệ thống & kiến trúc" (AI không thay thế được) và "Kiểm tra & validate output AI" (fact-checking, review code). Điều này cung cấp insight quan trọng cho định hướng nghề nghiệp: Kỹ sư tương lai không cạnh tranh với AI về tốc độ viết code, mà cạnh tranh về khả năng thiết kế hệ thống tổng thể và năng lực thẩm định tính chính xác/an toàn của sản phẩm do AI tạo ra.

---

# 📌 Kết luận

AI không chỉ thay thế một số nhiệm vụ mà còn đang tái định hình cách thức làm việc trong ngành Khoa học Máy tính.

Những công việc mang tính lặp lại sẽ được tự động hóa ngày càng nhiều, trong khi các kỹ năng sáng tạo, tư duy chiến lược và khả năng ra quyết định vẫn là lợi thế cạnh tranh quan trọng của con người trong tương lai.

---

# 🛠️ Công nghệ sử dụng

- Python
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

---

# 👥 Đối tượng sử dụng

- Sinh viên CNTT
- Data Analyst
- Data Scientist
- AI Researcher
- Product Manager
- Doanh nghiệp nghiên cứu tác động của AI đến lực lượng lao động
