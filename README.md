# ẢNH HƯỞNG CỦA AI AGENT ĐỐI VỚI CÁC NGÀNH LĨNH VỰC KHOA HỌC MÁY TÍNH

## Giới thiệu dự án

Dự án trực quan hóa dữ liệu nhằm phân tích tác động của Trí tuệ nhân tạo (AI) đối với các nghề nghiệp thuộc lĩnh vực Khoa học Máy tính (Computer Science).

Dashboard tập trung khám phá:

- Mức độ tự động hóa của AI trong các công việc CNTT.
- Sự khác biệt giữa đánh giá của chuyên gia và người lao động.
- Lý do người lao động mong muốn hoặc không mong muốn AI thay thế công việc.
- Các hình thức ứng dụng của mô hình ngôn ngữ lớn (LLM).
- Đặc trưng nhiệm vụ phù hợp với AI.
- Xu hướng phát triển và nhu cầu AI trong tương lai.

---

## Mục tiêu nghiên cứu

Nghiên cứu được xây dựng nhằm trả lời các câu hỏi:

1. AI đang tác động đến các nghề nghiệp trong lĩnh vực Khoa học Máy tính như thế nào?
2. Người lao động và chuyên gia có đánh giá giống nhau về khả năng tự động hóa của AI không?
3. Vì sao người lao động muốn hoặc không muốn AI thay thế công việc?
4. LLM hiện đang được sử dụng như thế nào trong công việc?
5. AI làm tốt nhất những loại nhiệm vụ nào?
6. Nhu cầu và vai trò của AI trong tương lai sẽ thay đổi ra sao?

---

# Cách tư duy xây dựng Dashboard

Dashboard được thiết kế theo tư duy kể chuyện dữ liệu (Data Storytelling).

## Giai đoạn 1 – Hiện trạng

**Câu hỏi**

> AI hiện đang tác động đến đâu trong ngành Máy tính?

**Biểu đồ**

<img width="1768" height="1085" alt="image" src="https://github.com/user-attachments/assets/cf63068d-cb1f-49be-8845-dd9f4857a7ee" />

**Mục đích**

Xác định mức độ ảnh hưởng của AI tới từng nhóm nghề nghiệp.

---

## Giai đoạn 2 – Khoảng cách nhận thức

**Câu hỏi**

> Người lao động có đánh giá đúng mức độ thay thế của AI hay không?

**Biểu đồ**

<img width="1541" height="914" alt="image" src="https://github.com/user-attachments/assets/1c1ccf6a-03a0-4c8e-8c88-52d57ae9798a" />

**Mục đích**

Phân tích sự khác biệt giữa góc nhìn chuyên gia và người lao động.

---

## Giai đoạn 3 – Động lực và rào cản

**Câu hỏi**

> Tại sao người lao động muốn hoặc không muốn AI tham gia công việc?

**Biểu đồ**

<img width="1636" height="938" alt="image" src="https://github.com/user-attachments/assets/d44e74b7-3329-42e4-9a16-0d215c3ee6ce" />


**Mục đích**

Khám phá động lực thúc đẩy và những lo ngại liên quan đến AI.

---

## Giai đoạn 4 – Hành vi sử dụng AI

**Câu hỏi**

> Người lao động đang sử dụng AI như thế nào?

**Biểu đồ**

<img width="1600" height="916" alt="image" src="https://github.com/user-attachments/assets/f550fec0-2aa3-4e6a-b2dd-3477c08f4ed9" />

**Mục đích**

Xác định các mục đích sử dụng AI phổ biến trong thực tế.

---

## Giai đoạn 5 – Năng lực AI

**Câu hỏi**

> AI hoạt động hiệu quả nhất với những loại nhiệm vụ nào?

**Biểu đồ**

<img width="1780" height="999" alt="image" src="https://github.com/user-attachments/assets/ce81fb0c-74fc-474b-b2e4-1463a1674531" />

**Mục đích**

Xác định các đặc trưng giúp AI đạt hiệu suất cao.

---

## Giai đoạn 6 – Tương lai

**Câu hỏi**

> Những kỹ năng và năng lực AI nào sẽ được yêu cầu trong tương lai?

**Biểu đồ**

<img width="1808" height="1073" alt="image" src="https://github.com/user-attachments/assets/1d6f839c-bc70-4f7c-a185-2c4a9a50daad" />


**Mục đích**

Dự báo xu hướng phát triển AI trong giai đoạn tiếp theo.

---

# 🔄 Sơ đồ luồng chức năng

```text
Raw Datasets
│
├── Worker Survey
├── Expert Assessment
├── Task Metadata
└── Occupation Information
        │
        ▼
Data Cleaning
        │
        ▼
Data Transformation
        │
        ▼
Feature Engineering
        │
        ▼
Exploratory Analysis
        │
        ▼
Visualization Layer
        │
        ├── Figure 1
        ├── Figure 2
        ├── Figure 3
        ├── Figure 4
        ├── Figure 5
        └── Figure 6
        │
        ▼
Insight Extraction
        │
        ▼
Conclusion & Future Trends
```

---

# 🖼️ Tổng quan Dashboard

## Figure 1 – AI Landscape Overview

Mô tả bức tranh tổng quan về tác động của AI tới các nghề nghiệp trong lĩnh vực Computer Science.

![Figure1](images/figure1.png)

---

## Figure 2 – Expert vs Worker Comparison

So sánh đánh giá giữa chuyên gia và người lao động về khả năng tự động hóa.

![Figure2](images/figure2.png)

---

## Figure 3 – Motivation Analysis

Phân tích lý do người lao động muốn hoặc không muốn AI tham gia công việc.

![Figure3](images/figure3.png)

---

## Figure 4 – LLM Usage Analysis

Phân tích các hình thức ứng dụng mô hình ngôn ngữ lớn trong công việc.

![Figure4](images/figure4.png)

---

## Figure 5 – Task Characteristics Analysis

Khám phá những loại nhiệm vụ AI thực hiện hiệu quả nhất.

![Figure5](images/figure5.png)

---

## Figure 6 – Future AI Demand Analysis

Dự báo nhu cầu AI và kỹ năng liên quan trong tương lai.

![Figure6](images/figure6.png)

---

# 📈 Kết quả nổi bật

## Insight 1

AI đang tác động mạnh tới các công việc có tính lặp lại và quy trình chuẩn hóa cao.

## Insight 2

Người lao động và chuyên gia tồn tại sự khác biệt đáng kể trong đánh giá khả năng tự động hóa.

## Insight 3

Lợi ích lớn nhất của AI là:

- Tiết kiệm thời gian
- Tăng năng suất
- Hỗ trợ xử lý công việc lặp lại

## Insight 4

LLM đang trở thành công cụ hỗ trợ phổ biến trong:

- Sinh mã nguồn
- Viết tài liệu
- Tìm kiếm thông tin
- Gỡ lỗi chương trình

## Insight 5

Các nhiệm vụ đòi hỏi:

- Sáng tạo
- Thiết kế hệ thống
- Tư duy chiến lược

vẫn cần vai trò quan trọng của con người.

## Insight 6

Nhu cầu về:

- Code Generation
- Automated Testing
- DevOps Automation
- Security Analysis

được dự báo sẽ tăng mạnh trong giai đoạn 2025–2030.

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
