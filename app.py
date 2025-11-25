import streamlit as st

st.set_page_config(
    page_title="影像處理期末考複習",
    page_icon="📸",
    layout="wide"
)

st.title("📸 影像處理期末考複習系統")

st.markdown("""
### 歡迎使用複習系統

本系統旨在協助複習影像處理期末考的重點主題。
請從左側側邊欄選擇您想要練習的主題。

#### 目前收錄主題：
- **01_Morphological**: 形態學運算 (Dilation, Erosion, Opening, Closing)
- (更多主題陸續新增中...)

#### 系統特色：
1. **互動式參數調整**：即時查看不同輸入對結果的影響。
2. **手算步驟模擬**：程式碼模擬手算邏輯，而非僅呼叫函式庫。
3. **自動驗證**：將手算結果與 OpenCV/NumPy 運算結果比對，確保正確性。
4. **視覺化推導**：透過矩陣圖形化顯示運算過程。

---
*Created for Image Processing Final Exam Review.*
""")
