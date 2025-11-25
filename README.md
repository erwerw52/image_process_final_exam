# 影像處理期末考複習系統 (Image Processing Final Exam Review)

這是一個使用 Streamlit 建構的互動式複習系統，旨在協助學生準備影像處理期末考。系統不僅展示運算結果，更著重於**模擬手算過程**與**視覺化推導**，幫助理解演算法背後的邏輯。

## 🌟 系統特色

1.  **互動式學習**：使用者可以選擇不同的題目與參數，即時觀察結果變化。
2.  **手算步驟模擬**：程式碼內建手算邏輯模擬，而非僅呼叫函式庫，展示運算細節。
3.  **自動驗證**：將手算模擬結果與 OpenCV/NumPy 標準函式庫運算結果比對，確保觀念正確。
4.  **視覺化推導**：透過矩陣圖形化顯示 Input、Kernel 與 Output，直觀理解運算過程。

## 📚 目前收錄主題

*   **01_Morphological (形態學運算)**
    *   Dilation (膨脹)
    *   Erosion (侵蝕)
    *   Opening (斷開)
    *   Closing (閉合)
    *   *包含自定義結構元素 (Kernel) 與圖形化驗證*

## 🚀 如何執行

請確保您已安裝 Python 與相關套件 (Streamlit, OpenCV, NumPy, Matplotlib)。

1.  開啟終端機 (Terminal)。
2.  執行以下指令啟動 Streamlit 應用程式：

```bash
streamlit run app.py
```

3.  系統將自動在瀏覽器中開啟。

## 📂 專案結構

*   `app.py`: 應用程式主入口，包含首頁介紹。
*   `pages/`: 存放各個主題的頁面程式碼。
    *   `01_Morphological.py`: 形態學運算頁面。
*   `utils.py`: 共用工具函式 (如繪圖功能)。
*   `problem_generation_prompt.md`: 題目生成提示詞 (開發用)。

---
*Created for Image Processing Final Exam Review.*