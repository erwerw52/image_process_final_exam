# 影像處理期末考複習 App - 新增例題 Prompt

## 角色設定
你是一個 Python 影像處理與 Streamlit 開發專家。我正在製作一個用來複習期末考的互動式網頁。

## 任務
請幫我為現有的 `app.py` 新增一個新的主題與計算例題。

## 程式碼架構要求
請提供以下兩部分的 Python 程式碼：

### 1. 解題函式 (`solve_{topic}`)
- **輸入**：輸入影像 (矩陣) 與參數。
- **輸出**：回傳 `(manual_res, formula, desc, lib_res)`。
    - `manual_res`: 手動演算法計算的結果 (請用 Python 迴圈模擬手算過程，不要直接呼叫 OpenCV)。
    - `formula`: 該演算法的 LaTeX 公式。
    - `desc`: 簡單的文字解說。
    - `lib_res`: 使用 OpenCV (`cv2`) 或 NumPy 直接計算的結果 (用來驗證)。
- **邏輯**：必須包含「手算模擬」與「函式庫驗證」兩個部分。

### 2. Streamlit Page 檔案 (`pages/`)
- 請將此內容寫成一個獨立的 Python 檔案，檔名建議為 `pages/XX_主題名稱.py` (例如 `pages/02_Histogram.py`)。
    - **注意**：檔名開頭的數字 `XX` 代表在側邊欄的排序，請根據目前已有的檔案編號遞增。
- 該檔案將作為 Streamlit 的一個獨立頁面運作，擁有獨立的畫面。
- **檔案結構**：
    1.  **Imports**: 匯入必要的套件 (`streamlit`, `numpy`, `cv2`, `matplotlib.pyplot` 等) 以及共用工具 (如 `from utils import plot_grid`)。
    2.  **解題函式**: 定義 `solve_{topic}`。
    3.  **頁面主程式**: 直接撰寫 Streamlit 頁面邏輯 (不需要包在 `main` 函式中，Streamlit 會直接執行)。
- **頁面流程**：
    1.  **標題與說明**：設定 `st.title` 與 `st.markdown` 說明。
    2.  **題目選擇**：如果有透過 `st.sidebar` 或 `st.selectbox` 選擇不同子題 (例如 Dilation/Erosion)。
    3.  **題目設定**：定義 Input Matrix 與參數。
    4.  **推導公式**：顯示 LaTeX 公式 (`st.latex`)。
    5.  **算式推導與圖形**：使用 `plot_grid` 繪製視覺化圖表。
    6.  **結果與驗證**：顯示手算結果，並比對 `lib_res`。

## 目前已有的 Helper Function
專案中已有 `utils.py`，包含 `plot_grid(matrix, title, ax, highlight_coords=None)`。
請在程式碼開頭加入 `from utils import plot_grid`。

## 我的需求
**主題**：[請在此填入主題，例如：Histogram Equalization]
**內容**：[請在此填入細節，例如：我需要直方圖均化與直方圖匹配的例題]
