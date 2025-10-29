# -*- coding: utf-8 -*- 
"""
Block Blast - Pieces Only UI (VN)
- UI chỉ hiển thị 3 pieces
- Trích xuất pieces từ ảnh bằng valley-projection (ổn định với viền/đổ bóng)
- Version 6.2 (Template Fix): Sử dụng Template Matching để phát hiện các hình dạng '7' và '┘'.
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


# ============================== VISION ==============================

class PiecesExtractor:
    def __init__(self):
        # Định nghĩa 2 hình dạng "L-5" (dạng '7' và dạng '┘') 3x3
        # shape A: hàng trên đủ 3 ô, cột phải đi xuống (dạng '7')
        # 1 1 1
        # 0 0 1
        # 0 0 1
        self.shape_A = np.array([[1, 1, 1],
                                 [0, 0, 1],
                                 [0, 0, 1]], dtype=int)

        # shape B: hàng dưới đủ 3 ô, cột phải đi lên (dạng '┘')
        # 0 0 1
        # 0 0 1
        # 1 1 1
        self.shape_B = np.array([[0, 0, 1],
                                 [0, 0, 1],
                                 [1, 1, 1]], dtype=int)

        # Convert these shapes to image templates
        self.template_A = self._get_template(self.shape_A)
        self.template_B = self._get_template(self.shape_B)

    def _get_template(self, shape):
        """Convert a shape array to an image template for template matching."""
        size = shape.shape[0]
        template = np.ones((size * 40 + 2, size * 40 + 2, 3), dtype=np.uint8) * 255
        for r, c in np.argwhere(shape == 1):
            y1, x1 = r * 40 + 1, c * 40 + 1
            y2, x2 = (r + 1) * 40 + 1, (c + 1) * 40 + 1
            cv2.rectangle(template, (x1, y1), (x2, y2), (80, 80, 80), -1)
        return template

    def _find_board_region(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < 50000: continue
            ar = w / max(1, h)
            if 0.7 <= ar <= 1.3:
                score = area * (1 - abs(1 - ar))
                if score > best_score:
                    best, best_score = (x, y, w, h), score
        if best is None:
            size = int(min(image.shape[:2]) * 0.7); x = (image.shape[1] - size) // 2
            y = int(image.shape[0] * 0.15); best = (x, y, size, size)
        return best

    def _find_piece_boxes(self, image, board_box):
        H, W = image.shape[:2]
        bx, by, bw, bh = board_box
        y1 = min(H, by + bh + int(0.02 * H)); x1 = max(0, bx - int(0.25 * bw))
        x2 = min(W, bx + bw + int(0.25 * bw)); roi = image[y1:H, x1:x2]
        Rh, Rw = roi.shape[:2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (8, 60, 80), (165, 255, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < (Rh * Rw) * 0.002 or area > (Rh * Rw) * 0.2: continue
            if max(w / h, h / w) > 6: continue
            boxes.append((x + x1, y + y1, w, h))
        return sorted(boxes, key=lambda b: b[0])[:3]

    def _decode_grid_from_roi(self, roi_bgr):
        if roi_bgr.shape[0] < 10 or roi_bgr.shape[1] < 10: return np.array([[1]])
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        def _seams_by_projection(g, axis=0):
            prof = cv2.GaussianBlur(g, (11, 1), 0).mean(axis=1) if axis == 0 else cv2.GaussianBlur(g, (1, 11), 0).mean(axis=0)
            pmin, pmax = prof.min(), prof.max()
            if pmax - pmin < 1e-3: return []
            norm = (prof - pmin) / (pmax - pmin)
            thr = float(norm.mean() - 0.20)
            idxs = np.where(norm < thr)[0]
            if not idxs.size: return []
            seams, start = [], idxs[0]
            for i in range(1, len(idxs)):
                if idxs[i] != idxs[i-1] + 1:
                    seams.append((start + idxs[i-1]) // 2); start = idxs[i]
            seams.append((start + idxs[-1]) // 2)
            merged = []
            min_sep = max(6, int(g.shape[axis] * 0.06))
            for s in seams:
                if not merged or s - merged[-1] >= min_sep: merged.append(s)
                else: merged[-1] = int((merged[-1] + s) / 2)
            return [s for s in merged if 3 < s < (g.shape[axis] - 3)]

        rows = _seams_by_projection(gray, axis=0)
        cols = _seams_by_projection(gray, axis=1)

        # === _bounds đơn giản & chính xác ===
        def _bounds_simple(length, seams):
            bounds = sorted(list(set([0] + seams + [length])))
            return [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]

        rb, cb = _bounds_simple(gray.shape[0], rows), _bounds_simple(gray.shape[1], cols)
        
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        grid = np.zeros((len(rb), len(cb)), dtype=int)
        for i, (rs, re) in enumerate(rb):
            for j, (cs, ce) in enumerate(cb):
                cell_hsv = hsv[rs:re, cs:ce]
                # Kiểm tra xem cell có đủ pixel để tính mean không
                if cell_hsv.size > 0:
                    h, s, v = cv2.split(cell_hsv)
                    is_piece_color = ((h > 8) & (h < 165) & (s > 60) & (v > 80)).mean()
                    grid[i, j] = 1 if is_piece_color > 0.40 else 0
                else:
                    grid[i, j] = 0
        return grid

    def _clean_grid_by_component(self, grid):
        """Giữ lại thành phần (cụm) pixel lớn nhất và loại bỏ mọi nhiễu."""
        if grid.size == 0 or np.sum(grid) == 0: return grid
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grid.astype(np.uint8), 4, cv2.CV_32S)
        if num_labels <= 1: return grid
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return np.where(labels == largest_label, 1, 0).astype(int)

    def extract_pieces(self, image_path):
        img = cv2.imread(image_path)
        if img is None: raise ValueError(f"Không thể đọc ảnh: {image_path}")
        board = self._find_board_region(img)
        boxes = self._find_piece_boxes(img, board)
        pieces = []
        H, W = img.shape[:2]
        for (x, y, w, h) in boxes:
            pad = int(max(w, h) * 0.15)
            roi = img[max(0, y-pad):min(H, y+h+pad), max(0, x-pad):min(W, x+w+pad)]
            
            grid = self._decode_grid_from_roi(roi)
            clean_grid = self._clean_grid_by_component(grid)
            
            # Template matching for specific pieces
            result_A = cv2.matchTemplate(roi, self.template_A, cv2.TM_CCOEFF_NORMED)
            result_B = cv2.matchTemplate(roi, self.template_B, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8  # Set a threshold for matching

            if np.max(result_A) > threshold:
                clean_grid = self.shape_A  # Match shape A
            elif np.max(result_B) > threshold:
                clean_grid = self.shape_B  # Match shape B

            pieces.append(clean_grid)

        while len(pieces) < 3: pieces.append(np.array([[1]]))
        return pieces[:3]


# ============================== UI ==============================
class PiecesOnlyUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Block Blast - Pieces Reader")
        self.root.geometry("900x520")
        # Áp dụng màu nền cho cửa sổ gốc để thấy rõ thay đổi theme
        try:
            self.root.configure(bg="#eef2ff")
        except Exception:
            pass
        self.colors = {
            'background': '#eef2ff',
            'primary': '#2563eb',
            'text_primary': '#0f172a',
            'text_secondary': '#475569',
        }
        self.extractor = PiecesExtractor()
        self._build_ui()

    def _build_ui(self):
        self.root.configure(bg=self.colors['background'])
        top = tk.Frame(self.root, bg=self.colors['background']); top.pack(fill=tk.X, padx=12, pady=10)
        btn = tk.Button(top, text="📁 Chọn ảnh…", command=self._select_image, bg=self.colors['primary'], fg="#fff", relief=tk.FLAT, bd=0, padx=12, pady=6, activebackground="#1d4ed8", activeforeground="#fff")
        btn.pack(side=tk.LEFT)
        self._bind_button_hover(btn, self.colors['primary'])
        self.path_var = tk.StringVar(value="Chưa chọn ảnh")
        tk.Label(top, textvariable=self.path_var, fg=self.colors['text_secondary'], bg=self.colors['background'], wraplength=700).pack(side=tk.LEFT, padx=10)
        wrapper = tk.Frame(self.root, bg=self.colors['background']); wrapper.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        tk.Label(wrapper, text="Detected Pieces", font=("Arial", 12, "bold"), fg=self.colors['text_primary'], bg=self.colors['background']).pack(anchor="w", pady=(0, 8))
        board = tk.Frame(wrapper, bd=1, relief=tk.RIDGE); board.pack(fill=tk.BOTH, expand=True)
        self.piece_labels = []
        for i in range(3):
            frame = tk.Frame(board, bd=1, relief=tk.SUNKEN, width=260, height=260, bg='white')
            frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=8, pady=8); frame.propagate(False)
            lbl = tk.Label(frame, text=f"Piece {i+1}", fg="#666", bg='white'); lbl.pack(expand=True)
            self.piece_labels.append(lbl)

    def _bind_button_hover(self, button, base_color):
        def on_enter(_):
            button.configure(bg=self._shade_color(base_color, -12))
        def on_leave(_):
            button.configure(bg=base_color)
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

    def _shade_color(self, hex_color, percent):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16); g = int(hex_color[2:4], 16); b = int(hex_color[4:6], 16)
        def clamp(x): return max(0, min(255, x))
        r = clamp(int(r + (255 - r) * (percent / 100.0)))
        g = clamp(int(g + (255 - g) * (percent / 100.0)))
        b = clamp(int(b + (255 - b) * (percent / 100.0)))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _select_image(self):
        fp = filedialog.askopenfilename(title="Chọn ảnh Block Blast", filetypes=[("Image", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not fp: return
        self.path_var.set(os.path.basename(fp))
        try:
            self.root.config(cursor="watch"); self.root.update()
            pieces = self.extractor.extract_pieces(fp)
            self._show_pieces(pieces)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi:\n{e}")
            import traceback; traceback.print_exc()
        finally:
            self.root.config(cursor="")

    def _piece_to_image(self, grid, cell_size=40):
        h, w = grid.shape
        img = np.full(((h * cell_size) + 2, (w * cell_size) + 2, 3), 255, dtype=np.uint8)
        for r, c in np.argwhere(grid == 1):
            y1, x1 = r * cell_size + 1, c * cell_size + 1
            y2, x2 = (r + 1) * cell_size + 1, (c + 1) * cell_size + 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 80), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (150, 150, 150), 1)
        return Image.fromarray(img, 'RGB')

    def _show_pieces(self, pieces):
        TARGET_DIM = 250
        for i, grid in enumerate(pieces):
            if not isinstance(grid, np.ndarray) or grid.size == 0 or np.sum(grid) == 0:
                self.piece_labels[i].configure(image='', text=f"Piece {i+1}")
                self.piece_labels[i].image = None
                continue
            h, w = grid.shape
            cell_size = int(TARGET_DIM / max(h, w, 1))
            pil_img = self._piece_to_image(grid, cell_size=cell_size)
            ph = ImageTk.PhotoImage(pil_img)
            self.piece_labels[i].configure(image=ph, text=""); self.piece_labels[i].image = ph

    def run(self): 
        self.root.mainloop()

if __name__ == "__main__":  # Fixed the typo here
    PiecesOnlyUI().run()
