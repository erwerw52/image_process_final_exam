import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import plot_grid

st.set_page_config(page_title="Histogram Processing", layout="wide")

def calculate_histogram_manual(image):
    """手算直方圖"""
    hist = np.zeros(256, dtype=int)
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            val = image[i, j]
            hist[val] += 1
    return hist

def solve_histogram_equalization(image, levels=256):
    """
    手算直方圖均化 (Histogram Equalization)
    """
    h, w = image.shape
    total_pixels = h * w
    
    # 1. 計算原始直方圖
    hist = calculate_histogram_manual(image)
    
    # 2. 計算 PDF (機率密度函數)
    pdf = hist / total_pixels
    
    # 3. 計算 CDF (累積分布函數)
    cdf = np.zeros(256)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + pdf[i]
        
    # 4. 計算映射表 (Map to L-1)
    # s_k = floor((L-1) * cdf_k)
    map_table = np.floor((levels - 1) * cdf).astype(np.uint8)
    
    # 5. 映射新影像
    output_image = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            output_image[i, j] = map_table[image[i, j]]
            
    # 產生推導步驟文字
    steps = []
    steps.append(f"影像大小: {h}x{w} = {total_pixels} pixels")
    steps.append(f"灰階層級 (L): {levels}")
    steps.append("步驟 1: 計算原始直方圖 (nk)")
    steps.append("步驟 2: 計算 PDF (pr = nk / N)")
    steps.append("步驟 3: 計算 CDF (sk = sum(pr))")
    steps.append(f"步驟 4: 映射數值 (round(sk * {levels-1}))")
    
    # 建立詳細表格數據 (僅列出有出現的灰階值)
    # 如果是 3-bit (L=8)，我們列出 0-7
    if levels == 8:
        display_range = range(8)
    else:
        display_range = np.nonzero(hist)[0]
        
    table_data = []
    for v in display_range:
        # 避免超出範圍 (雖然 hist 是 256)
        if v >= 256: continue
        
        # PDF Calculation
        pdf_calc = f"{hist[v]}/{total_pixels} = {pdf[v]:.4f}"
        
        # CDF Calculation
        if v == 0:
            cdf_calc = f"{pdf[v]:.4f}"
        else:
            cdf_calc = f"{cdf[v-1]:.4f} + {pdf[v]:.4f} = {cdf[v]:.4f}"
            
        # Output Calculation
        out_val_raw = (levels - 1) * cdf[v]
        out_calc = f"{levels-1} * {cdf[v]:.4f} = {out_val_raw:.2f} → {map_table[v]}"
        
        table_data.append({
            "r_k": v,
            "n_k": hist[v],
            "p_r = n_k / MN": pdf_calc,
            "s_k = Σ p_r": cdf_calc,
            f"Output = round({levels-1} * s_k)": out_calc
        })
        
    return output_image, map_table, steps, table_data

def solve_histogram_matching(source_img, target_hist_spec):
    """
    手算直方圖匹配 (Histogram Matching / Specification)
    """
    h, w = source_img.shape
    total_pixels = h * w
    
    # --- 來源影像處理 (S domain) ---
    # 1. Source Histogram
    src_hist = calculate_histogram_manual(source_img)
    # 2. Source PDF
    src_pdf = src_hist / total_pixels
    # 3. Source CDF (S_k)
    src_cdf = np.cumsum(src_pdf)
    # 4. Source Equalized Level (s_k) - 這裡保持小數以便比對，或轉為整數皆可，通常比對 CDF 值
    # 課本通常是: s_k = T(r_k) = (L-1) * CDF
    src_s = np.round(255 * src_cdf).astype(int)

    # --- 目標直方圖處理 (Z domain) ---
    # target_hist_spec 應該是一個 normalized 的 PDF 或是 counts
    # 這裡假設輸入是 counts，先轉 PDF
    target_total = np.sum(target_hist_spec)
    tgt_pdf = target_hist_spec / target_total
    # 1. Target CDF (v_k)
    tgt_cdf = np.cumsum(tgt_pdf)
    # 2. Target Equalized Level (v_q) = G(z_q)
    tgt_v = np.round(255 * tgt_cdf).astype(int)
    
    # --- 映射 (Mapping) ---
    # 對於每個 r_k，找到 z_q 使得 |v_q - s_k| 最小
    # map_table[r_k] = z_q
    map_table = np.zeros(256, dtype=np.uint8)
    
    mapping_steps = []
    
    # 僅針對來源影像中有出現的像素值做計算
    unique_src = np.nonzero(src_hist)[0]
    
    for r_k in unique_src:
        s_val = src_s[r_k]
        # 尋找最接近的 v_q
        diff = np.abs(tgt_v - s_val)
        min_idx = np.argmin(diff) # 找到最小差異的索引 z_q
        z_q = min_idx
        
        map_table[r_k] = z_q
        mapping_steps.append({
            "r_k": r_k,
            "s_k (Eq. Source)": s_val,
            "Match z_q": z_q,
            "v_q (Eq. Target)": tgt_v[z_q]
        })

    # 產生結果影像
    output_image = map_table[source_img]
    
    return output_image, map_table, mapping_steps

st.title("Histogram Processing (直方圖處理)")
st.markdown("---")

# 選擇例題
topic = st.selectbox("選擇例題", [
    "1. Histogram Calculation (直方圖計算)",
    "2. Histogram Equalization (直方圖均化，送分題)",
    "3. Histogram Matching (直方圖匹配)"
])

if topic == "1. Histogram Calculation (直方圖計算)":
    st.subheader("1. 題目設定")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Input Image (4x4, 3-bit grayscale)**")
        # 產生一個簡單的 4x4 影像，數值範圍 0-7
        input_img = np.array([
            [2, 3, 3, 2],
            [4, 2, 4, 3],
            [3, 2, 3, 5],
            [2, 4, 2, 4]
        ], dtype=np.uint8)
        st.write(input_img)
        
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_grid(input_img, "Input Image", ax)
        st.pyplot(fig)
        
    with col2:
        st.markdown("**說明**")
        st.info("直方圖 (Histogram) 統計影像中每個灰階值出現的次數。")
        st.latex(r"h(r_k) = n_k")
        st.markdown("其中 $r_k$ 是第 $k$ 個灰階值，$n_k$ 是該灰階值在影像中出現的次數。")

    st.subheader("2. 計算結果")
    # 手算
    hist_manual = calculate_histogram_manual(input_img)
    # OpenCV 驗證
    hist_cv = cv2.calcHist([input_img], [0], None, [256], [0, 256]).flatten().astype(int)
    
    # 顯示非零的部分
    valid_indices = np.nonzero(hist_manual)[0]
    st.write("非零像素統計：")
    st.json({f"灰階 {i}": int(hist_manual[i]) for i in valid_indices})
    
    # 繪圖
    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
    ax_hist.bar(np.arange(256), hist_manual, color='gray', width=1.0, edgecolor='black')
    ax_hist.set_title("Histogram")
    ax_hist.set_xlabel("Gray Level")
    ax_hist.set_ylabel("Count")
    ax_hist.set_xlim(-0.5, 7.5) # 因為是 3-bit 範例，只顯示前段
    st.pyplot(fig_hist)
    
    # 驗證
    is_correct = np.array_equal(hist_manual, hist_cv)
    if is_correct:
        st.success(f"驗證成功！手算結果與 cv2.calcHist 一致。")
    else:
        st.error("驗證失敗！")

elif topic == "2. Histogram Equalization (直方圖均化，送分題)":
    st.subheader("1. 題目設定")
    st.markdown("直方圖均化用於增強影像對比度，使灰階值分佈更均勻。")
    
    prob_type = st.radio("選擇題目類型", ["基礎觀念 (4x4)", "課本模擬 (64x64, 3-bit)"])

    if prob_type == "基礎觀念 (4x4)":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Input Image (Low Contrast)**")
            # 模擬低對比影像 (數值集中在中間)
            input_img = np.array([
                [50, 55, 52, 55],
                [52, 50, 55, 50],
                [55, 52, 50, 52],
                [50, 55, 50, 55]
            ], dtype=np.uint8)
            st.write(input_img)
            
        with col2:
            st.markdown("**公式**")
            st.latex(r"s_k = T(r_k) = (L-1) \sum_{j=0}^{k} p_r(r_j)")
            st.markdown("目標是將原始累積分布函數 (CDF) 線性化。")

        # 計算
        manual_res, map_table, steps, table_data = solve_histogram_equalization(input_img, levels=256)
        cv_res = cv2.equalizeHist(input_img)
        
        st.subheader("2. 推導過程")
        st.table(table_data)
        
        st.subheader("3. 結果比較")
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        plot_grid(input_img, "Input (Low Contrast)", axes[0])
        plot_grid(manual_res, "Equalized Output", axes[1])
        st.pyplot(fig)
        
        # 驗證
        is_correct = np.array_equal(manual_res, cv_res)
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("**手算結果**")
            st.write(manual_res)
        with col_v2:
            st.markdown(f"**OpenCV 結果 ({'✅ 正確' if is_correct else '❌ 錯誤'})**")
            st.write(cv_res)
            
    else:
        # 課本模擬 (64x64, 3-bit)
        st.info("此模式模擬課本範例：3-bit 影像 (L=8)，大小 64x64 (MN=4096)。")
        
        case_id = st.selectbox("選擇隨機題號", ["Case 1", "Case 2", "Case 3"])
        
        # 設定隨機種子以確保每次選擇同一題號時結果一致
        seed_map = {"Case 1": 42, "Case 2": 100, "Case 3": 999}
        np.random.seed(seed_map[case_id])
        
        # 產生隨機直方圖分佈 (總和 4096)
        # 為了讓題目有趣，我們隨機產生 8 個數值，然後正規化到 4096
        raw_counts = np.random.randint(100, 1000, size=8)
        counts = np.round(raw_counts / raw_counts.sum() * 4096).astype(int)
        # 修正誤差，確保總和剛好 4096
        diff = 4096 - counts.sum()
        counts[np.argmax(counts)] += diff
        
        # 根據 counts 產生影像 (為了讓 solve_histogram_equalization 可以運作)
        pixels = []
        for val, count in enumerate(counts):
            pixels.extend([val] * count)
        np.random.shuffle(pixels)
        input_img_3bit = np.array(pixels, dtype=np.uint8).reshape((64, 64))
        
        st.markdown(f"### {case_id} 題目數據 (Input Distribution)")
        st.markdown("假設影像大小 $64 \\times 64$ ($MN=4096$)，灰階層級 $L=8$。")
        
        # 顯示原始直方圖數據
        col_data1, col_data2 = st.columns([1, 2])
        with col_data1:
            st.write("原始直方圖 ($n_k$):")
            st.write({f"r_{i}": c for i, c in enumerate(counts)})
        with col_data2:
            fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
            ax_hist.bar(np.arange(8), counts, color='skyblue', edgecolor='black')
            ax_hist.set_title("Original Histogram")
            ax_hist.set_xticks(np.arange(8))
            st.pyplot(fig_hist)

        # 計算
        manual_res, map_table, steps, table_data = solve_histogram_equalization(input_img_3bit, levels=8)
        
        st.subheader("2. 推導過程與結果")
        st.markdown("計算公式： $s_k = T(r_k) = (8-1) \\sum_{j=0}^{k} p_r(r_j) = 7 \\times CDF$")
        st.table(table_data)
        
        st.subheader("3. 均化後直方圖")
        # 計算均化後的直方圖
        out_hist = calculate_histogram_manual(manual_res)
        
        fig_out, ax_out = plt.subplots(figsize=(8, 3))
        # 這裡只顯示 0-7
        ax_out.bar(np.arange(8), out_hist[:8], color='lightgreen', edgecolor='black')
        ax_out.set_title("Equalized Histogram")
        ax_out.set_xticks(np.arange(8))
        for i, v in enumerate(out_hist[:8]):
            if v > 0:
                ax_out.text(i, v + 50, str(v), ha='center')
        st.pyplot(fig_out)
        
        st.markdown("**觀察：**")
        st.markdown("可以看到直方圖分佈變得較為平坦（雖然因為離散特性無法完全平坦），且像素值被重新映射到新的層級。")

elif topic == "3. Histogram Matching (直方圖匹配)":
    st.subheader("1. 題目設定")
    st.markdown("將來源影像的直方圖轉換為「指定的」目標直方圖形狀。")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Source Image (A)**")
        src_img = np.array([
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3]
        ], dtype=np.uint8)
        st.write(src_img)
        
    with col2:
        st.markdown("**Target Histogram (Spec)**")
        # 指定一個目標直方圖 (例如想要變暗或變亮)
        # 這裡指定一個傾向於高灰階的分布
        target_hist = np.zeros(256)
        # 簡單設定：希望 0->0個, 1->0個, 2->0個, ..., 5->4個, 6->6個, 7->6個
        # 總像素數要一樣 (16)
        target_vals = {5: 4, 6: 6, 7: 6} 
        for k, v in target_vals.items():
            target_hist[k] = v
            
        st.write("Target Counts (Total 16):")
        st.write(target_vals)
        
    with col3:
        st.markdown("**原理**")
        st.latex(r"z_q = G^{-1}(T(r_k))")
        st.markdown("1. 對 Source 做 Equalization 得到 $s_k$")
        st.markdown("2. 對 Target 做 Equalization 得到 $v_q$")
        st.markdown("3. 找最接近的 $v_q \approx s_k$，對應的 $z_q$ 即為結果")

    # 計算
    manual_res, map_table, mapping_steps = solve_histogram_matching(src_img, target_hist)
    
    st.subheader("2. 推導過程 (Mapping Table)")
    st.table(mapping_steps)
    
    st.subheader("3. 結果展示")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_grid(src_img, "Source Image", axes[0])
    plot_grid(manual_res, "Matched Output", axes[1])
    st.pyplot(fig)
    
    st.markdown("**結果驗證**")
    st.markdown("檢查 Output 的直方圖是否接近 Target Histogram。")
    out_hist = calculate_histogram_manual(manual_res)
    
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    ax1.bar(np.arange(8), target_hist[:8], color='blue', alpha=0.6, label='Target Spec')
    ax1.set_title("Target Specification")
    ax1.set_ylim(0, 10)
    
    ax2.bar(np.arange(8), out_hist[:8], color='green', alpha=0.6, label='Output Hist')
    ax2.set_title("Actual Output Histogram")
    ax2.set_ylim(0, 10)
    st.pyplot(fig2)
    st.info("注意：由於數位影像的離散特性，Output 直方圖通常無法「完全」等於 Target，只能近似。")

