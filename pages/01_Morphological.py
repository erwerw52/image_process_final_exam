import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from utils import plot_grid

st.set_page_config(page_title="Morphological Operations", layout="wide")

def solve_morphology_animated(input_grid, kernel, op_type, placeholder, speed=0.1):
    """
    帶動畫的形態學運算，展示 Kernel 滑動過程
    """
    h, w = input_grid.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    manual_res = np.zeros_like(input_grid)
    padded = np.pad(input_grid, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    for i in range(h):
        for j in range(w):
            roi = padded[i:i+kh, j:j+kw]
            
            if op_type == 'Dilation':
                if np.sum(roi * kernel) > 0:
                    manual_res[i, j] = 1
            elif op_type == 'Erosion':
                match = True
                for ki in range(kh):
                    for kj in range(kw):
                        if kernel[ki, kj] == 1 and roi[ki, kj] != 1:
                            match = False
                            break
                if match:
                    manual_res[i, j] = 1
            
            # 繪製動畫幀
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            
            # 左圖：Input 並標示目前 Kernel 位置
            axes[0].imshow(input_grid, cmap='Greys', vmin=0, vmax=1)
            axes[0].set_title(f"Input (A) - Kernel platform: ({i},{j})")
            axes[0].set_xticks(np.arange(-0.5, w, 1), minor=True)
            axes[0].set_yticks(np.arange(-0.5, h, 1), minor=True)
            axes[0].grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
            
            # 繪製 Kernel 覆蓋區域的紅色方框
            rect_y = i - pad_h
            rect_x = j - pad_w
            rect = plt.Rectangle((rect_x - 0.5, rect_y - 0.5), kw, kh, 
                                  fill=False, edgecolor='red', linewidth=3)
            axes[0].add_patch(rect)
            # 標示中心點
            axes[0].plot(j, i, 'ro', markersize=10)
            
            # 中圖：顯示 ROI 與 Kernel 對應
            axes[1].imshow(roi, cmap='Blues', vmin=0, vmax=1)
            axes[1].set_title(f"ROI (screenshot)")
            for (yi, xi), val in np.ndenumerate(roi):
                color = 'white' if val == 1 else 'black'
                axes[1].text(xi, yi, int(val), ha='center', va='center', color=color, fontsize=12)
                # 標示 Kernel 為 1 的位置
                if kernel[yi, xi] == 1:
                    rect_k = plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1, 
                                           fill=False, edgecolor='red', linewidth=2)
                    axes[1].add_patch(rect_k)
            axes[1].set_xticks(np.arange(-0.5, kw, 1), minor=True)
            axes[1].set_yticks(np.arange(-0.5, kh, 1), minor=True)
            axes[1].grid(which='minor', color='black', linestyle='-', linewidth=1)
            
            # 右圖：目前的輸出結果
            axes[2].imshow(manual_res, cmap='Greys', vmin=0, vmax=1)
            result_val = manual_res[i, j]
            axes[2].set_title(f"Output - Now({i},{j})={'1 ✓' if result_val else '0'}")
            axes[2].set_xticks(np.arange(-0.5, w, 1), minor=True)
            axes[2].set_yticks(np.arange(-0.5, h, 1), minor=True)
            axes[2].grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
            # 標示目前正在處理的點
            axes[2].plot(j, i, 'go' if result_val else 'rx', markersize=10)
            
            plt.tight_layout()
            placeholder.pyplot(fig)
            plt.close(fig)
            time.sleep(speed)
    
    return manual_res

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

# 動畫展示區
st.subheader("🎬 動畫展示：Kernel 滑動過程")
st.markdown("點擊下方按鈕，觀看 Kernel 如何在 Input 上逐一滑動並計算結果。")

# 動畫控制
col_anim1, col_anim2 = st.columns([1, 3])
with col_anim1:
    speed = st.slider("動畫速度 (秒/幀)", min_value=0.01, max_value=0.5, value=0.02, step=0.01)
    
# 對所有操作啟用動畫
if current_op in ['Dilation', 'Erosion']:
    st.markdown(f"**使用完整 Input ({input_grid.shape[0]}x{input_grid.shape[1]}) 進行 {current_op} 動畫展示**")
    
    col_preview1, col_preview2 = st.columns(2)
    with col_preview1:
        st.markdown("**Input Image (A)**")
        fig_preview, ax_preview = plt.subplots(figsize=(6, 4))
        plot_grid(input_grid, "Input", ax_preview)
        st.pyplot(fig_preview)
        plt.close(fig_preview)
    with col_preview2:
        st.markdown("**Kernel (B)**")
        st.write(kernel)
    
    if st.button(f"▶️ 播放 {current_op} 動畫 (完整範圍)", type="primary"):
        animation_placeholder = st.empty()
        with st.spinner(f"動畫播放中... (共 {input_grid.shape[0] * input_grid.shape[1]} 幀)"):
            result = solve_morphology_animated(input_grid, kernel, current_op, animation_placeholder, speed)
        st.success("✅ 動畫播放完成！")
        
        # 顯示最終結果比較
        st.markdown("**最終結果比較：**")
        fig_final, axes_final = plt.subplots(1, 3, figsize=(14, 4))
        plot_grid(input_grid, "Input", axes_final[0])
        plot_grid(kernel, "Kernel", axes_final[1])
        plot_grid(result, f"Output ({current_op})", axes_final[2])
        st.pyplot(fig_final)
        
elif current_op in ['Opening', 'Closing']:
    # Opening 和 Closing 是複合運算，顯示兩階段動畫
    first_op = 'Erosion' if current_op == 'Opening' else 'Dilation'
    second_op = 'Dilation' if current_op == 'Opening' else 'Erosion'
    
    st.markdown(f"**{current_op} = {first_op} → {second_op}**")
    st.markdown(f"使用完整 Input ({input_grid.shape[0]}x{input_grid.shape[1]}) 進行動畫展示")
    
    col_preview1, col_preview2 = st.columns(2)
    with col_preview1:
        st.markdown("**Input Image (A)**")
        fig_preview, ax_preview = plt.subplots(figsize=(6, 4))
        plot_grid(input_grid, "Input", ax_preview)
        st.pyplot(fig_preview)
        plt.close(fig_preview)
    with col_preview2:
        st.markdown("**Kernel (B)**")
        st.write(kernel)
    
    if st.button(f"▶️ 播放 {current_op} 動畫 (兩階段)", type="primary"):
        # 第一階段
        st.markdown(f"### 階段 1: {first_op}")
        animation_placeholder1 = st.empty()
        with st.spinner(f"階段 1 ({first_op}) 播放中..."):
            mid_result = solve_morphology_animated(input_grid, kernel, first_op, animation_placeholder1, speed)
        st.success(f"✅ 階段 1 ({first_op}) 完成！")
        
        # 顯示中間結果
        st.markdown("**階段 1 結果：**")
        fig_mid, axes_mid = plt.subplots(1, 2, figsize=(10, 4))
        plot_grid(input_grid, "Original Input", axes_mid[0])
        plot_grid(mid_result, f"After {first_op}", axes_mid[1])
        st.pyplot(fig_mid)
        plt.close(fig_mid)
        
        # 第二階段
        st.markdown(f"### 階段 2: {second_op}")
        animation_placeholder2 = st.empty()
        with st.spinner(f"階段 2 ({second_op}) 播放中..."):
            final_result = solve_morphology_animated(mid_result, kernel, second_op, animation_placeholder2, speed)
        st.success(f"✅ 階段 2 ({second_op}) 完成！")
        
        # 顯示最終結果比較
        st.markdown("**最終結果比較：**")
        fig_final, axes_final = plt.subplots(1, 3, figsize=(14, 4))
        plot_grid(input_grid, "Original Input", axes_final[0])
        plot_grid(mid_result, f"After {first_op}", axes_final[1])
        plot_grid(final_result, f"Final ({current_op})", axes_final[2])
        st.pyplot(fig_final)

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
