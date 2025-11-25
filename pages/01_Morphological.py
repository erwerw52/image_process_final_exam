import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import plot_grid

st.set_page_config(page_title="Morphological Operations", layout="wide")

def solve_morphology(input_grid, kernel, op_type):
    """
    核心解題邏輯
    回傳: (結果矩陣, 推導步驟文字, OpenCV驗證結果)
    """
    steps = []
    h, w = input_grid.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # OpenCV 驗證
    kernel_uint8 = kernel.astype(np.uint8)
    input_uint8 = input_grid.astype(np.uint8)
    
    if op_type == 'Dilation':
        cv_res = cv2.dilate(input_uint8, kernel_uint8)
        formula = r"A \oplus B = \{ z | (\hat{B})_z \cap A \neq \emptyset \}"
        desc = "只要 Kernel 覆蓋區域內 Input 有 1，中心點即為 1 (OR 運算)。"
        # 手動推導模擬
        manual_res = np.zeros_like(input_grid)
        padded = np.pad(input_grid, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        for i in range(h):
            for j in range(w):
                roi = padded[i:i+kh, j:j+kw]
                if np.sum(roi * kernel) > 0:
                    manual_res[i, j] = 1
                    steps.append(f"({i},{j}): Kernel ∩ A ≠ Ø (Hit) -> 1")
                else:
                    # steps.append(f"位置 ({i},{j}): 無交集 -> 設為 0")
                    pass

    elif op_type == 'Erosion':
        cv_res = cv2.erode(input_uint8, kernel_uint8)
        formula = r"A \ominus B = \{ z | (B)_z \subseteq A \}"
        desc = "Kernel 必須完全落在 Input 的 1 區域內，中心點才為 1 (AND 運算)。"
        manual_res = np.zeros_like(input_grid)
        padded = np.pad(input_grid, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        for i in range(h):
            for j in range(w):
                roi = padded[i:i+kh, j:j+kw]
                # 檢查 Kernel 為 1 的地方，ROI 是否也都為 1
                match = True
                for ki in range(kh):
                    for kj in range(kw):
                        if kernel[ki, kj] == 1 and roi[ki, kj] != 1:
                            match = False
                            break
                if match:
                    manual_res[i, j] = 1
                    steps.append(f"({i},{j}): Kernel ⊆ A (Fit) -> 1")
                else:
                    # steps.append(f"位置 ({i},{j}): 未完全包含 -> 設為 0")
                    pass

    elif op_type == 'Opening':
        cv_res = cv2.morphologyEx(input_uint8, cv2.MORPH_OPEN, kernel_uint8)
        formula = r"A \circ B = (A \ominus B) \oplus B"
        desc = "先 Erosion (侵蝕) 再 Dilation (膨脹)。用於消除細小雜訊。"
        # 複用上面的邏輯
        temp_res, _, _, _, _ = solve_morphology(input_grid, kernel, 'Erosion')
        manual_res, _, _, _, _ = solve_morphology(temp_res, kernel, 'Dilation')
        steps.append("步驟 1: 執行 Erosion (結果如中間圖)")
        steps.append("步驟 2: 對 Erosion 結果執行 Dilation")

    elif op_type == 'Closing':
        cv_res = cv2.morphologyEx(input_uint8, cv2.MORPH_CLOSE, kernel_uint8)
        formula = r"A \bullet B = (A \oplus B) \ominus B"
        desc = "先 Dilation (膨脹) 再 Erosion (侵蝕)。用於填補破洞。"
        temp_res, _, _, _, _ = solve_morphology(input_grid, kernel, 'Dilation')
        manual_res, _, _, _, _ = solve_morphology(temp_res, kernel, 'Erosion')
        steps.append("步驟 1: 執行 Dilation (結果如中間圖)")
        steps.append("步驟 2: 對 Dilation 結果執行 Erosion")
        
    return manual_res, formula, desc, cv_res, steps

st.title("Morphological Operations (形態學)")
st.markdown("---")

# 定義題目資料
problems = [
    {"name": "Dilation (膨脹)", "op": "Dilation"},
    {"name": "Erosion (侵蝕)", "op": "Erosion"},
    {"name": "Opening (斷開)", "op": "Opening"},
    {"name": "Closing (閉合)", "op": "Closing"},
]

selected_prob = st.selectbox("選擇例題", [p["name"] for p in problems])
current_op = next(p["op"] for p in problems if p["name"] == selected_prob)

# 1. 題目設定
st.subheader("1. 題目設定")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Input Image (A)**")
    # 模擬附件的複雜圖形 (H-shape with notches)
    input_grid = np.zeros((12, 16), dtype=int)
    
    # Left Block
    input_grid[2:10, 2:7] = 1
    # Left Notch
    input_grid[5:7, 2:4] = 0
    
    # Right Block
    input_grid[2:10, 9:14] = 1
    # Right Notch
    input_grid[5:7, 12:14] = 0
    
    # Bridge
    input_grid[5:7, 7:9] = 1

    # 針對特定操作加入雜訊或破洞以凸顯效果
    if current_op == 'Opening':
        # 加入一些細小雜訊 (會被 Opening 消除)
        input_grid[1, 8] = 1 
        input_grid[10, 3] = 1
    if current_op == 'Closing':
        # 加入一些內部破洞 (會被 Closing 填補)
        input_grid[3, 5] = 0
        input_grid[8, 10] = 0

    st.write(input_grid)

with col2:
    st.markdown("**Structuring Element (Kernel B)**")
    # 修改為三角形結構元素
    kernel = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0]
    ])
    st.write(kernel)

# 2. 推導公式
st.subheader("2. 推導公式")
manual_res, formula, desc, cv_res, steps = solve_morphology(input_grid, kernel, current_op)
st.latex(formula)
st.info(desc)

# 3. 算式推導與圖形
st.subheader("3. 算式推導與結果")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
plot_grid(input_grid, "Input (A)", axes[0])

# 中間步驟圖示 (如果是 Open/Close 顯示中間產物，否則顯示 Kernel)
if current_op in ['Opening', 'Closing']:
    mid_op = 'Erosion' if current_op == 'Opening' else 'Dilation'
    mid_res, _, _, _, _ = solve_morphology(input_grid, kernel, mid_op)
    plot_grid(mid_res, f"Step 1: {mid_op}", axes[1])
else:
    plot_grid(kernel, "Kernel (B)", axes[1])
    
plot_grid(manual_res, f"Result: {current_op}", axes[2])
st.pyplot(fig)

with st.expander("查看詳細推導步驟 (文字)"):
    st.markdown("僅列出結果為 **1** 的關鍵座標：")
    st.text("\n".join(steps))

# 4. 驗證
st.subheader("4. 正確性驗證 (Library Check)")
is_correct = np.array_equal(manual_res, cv_res)

col_v1, col_v2 = st.columns(2)
with col_v1:
    st.markdown("**手算結果 (Manual)**")
    st.write(manual_res)
with col_v2:
    st.markdown(f"**OpenCV 結果 ({'✅ 正確' if is_correct else '❌ 錯誤'})**")
    st.write(cv_res)
    
if not is_correct:
    st.error("警告：手算結果與函式庫運算不符，請檢查推導邏輯！")
else:
    st.success("驗證成功！手算邏輯與 OpenCV 運算結果一致。")
