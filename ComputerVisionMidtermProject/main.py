# -*- coding: utf-8 -*-
"""
Block Blast Solver — Full UI (keep original UI), merged with PiecesExtractor (valley-projection)
- Board logic: giữ nguyên từ phần 1
- Pieces logic: thay thế bằng logic phần 2 (valley projection + CC filtering)
- Solver logic: Nâng cấp để tìm chuỗi nước đi tối ưu (sequential simulation)
"""
import os
import cv2
import numpy as np
import tkinter as tk
import tkinter.font as tkfont
from typing import List
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import itertools # Thêm thư viện để tạo hoán vị

class BlockBlastVision:
    """Xử lý ảnh để trích xuất board."""
    def __init__(self, grid_size: int = 8):
        self.grid_size = grid_size

    # --------------------------- BOARD ---------------------------
    def extract_board_from_image(self, image_path: str) -> np.ndarray:
        """Trích xuất board từ ảnh Block Blast."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh: {image_path}")
            board_region = self._find_board_region(image)
            x, y, w, h = board_region
            board_image = image[y:y+h, x:x+w]
            size = min(board_image.shape[:2])
            board_image = cv2.resize(board_image, (size, size))
            board = self._extract_grid_cells_improved(board_image)
            self._debug_board(board)
            return board
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh: {e}")
            return self._create_fallback_board()

    def _find_board_region(self, image):
        """Tìm vùng board trong ảnh Block Blast - phiên bản cải tiến (giữ nguyên)"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour = None
        best_score = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10000:
                continue
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            ar = w_rect / max(1, h_rect)
            if 0.7 <= ar <= 1.3: # gần vuông
                score = area * (1 - abs(1 - ar))
                if score > best_score:
                    best_score = score
                    best_contour = contour
        if best_contour is not None:
            x, y, w_rect, h_rect = cv2.boundingRect(best_contour)
            return (x, y, w_rect, h_rect)
        else:
            # Fallback tương tự bản cũ
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_bright = np.array([0, 30, 50])
            upper_bright = np.array([180, 255, 255])
            mask = cv2.inRange(hsv, lower_bright, upper_bright)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w_rect, h_rect = cv2.boundingRect(largest)
                margin = 50
                x = max(0, x - margin)
                y = max(0, y - margin)
                w_rect = min(w_rect + 2*margin, image.shape[1] - x)
                h_rect = min(h_rect + 2*margin, image.shape[0] - y)
                return (x, y, w_rect, h_rect)
            else:
                center_x, center_y = w // 2, h // 2
                board_size = min(w, h) // 2
                x = center_x - board_size // 2
                y = center_y - board_size // 2
                return (x, y, board_size, board_size)

    def _extract_grid_cells_improved(self, image):
        """Chia ảnh thành grid cells và phân tích màu - giữ nguyên"""
        h, w = image.shape[:2]
        cell_size = min(h, w) // self.grid_size
        margin = int(cell_size * 0.1)
        board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y1 = max(0, row * cell_size + margin)
                y2 = min(h, (row + 1) * cell_size - margin)
                x1 = max(0, col * cell_size + margin)
                x2 = min(w, (col + 1) * cell_size - margin)
                cell = image[y1:y2, x1:x2]
                if cell.size == 0:
                    continue
                color_code = self._analyze_cell_color_improved(cell)
                board[row, col] = color_code
        return board

    def _analyze_cell_color_improved(self, cell):
        """Phân tích màu cell (giữ nguyên)"""
        if cell.size == 0:
            return 0
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        h, s, v = np.mean(hsv, axis=(0, 1))
        if v < 100 or s < 40:
            return 0
        if h < 10 or h > 170: return 1
        elif 10 <= h < 25: return 2
        elif 25 <= h < 45: return 3
        elif 45 <= h < 75: return 4
        elif 75 <= h < 105: return 5
        elif 105 <= h < 135: return 6
        elif 135 <= h < 165: return 7
        else: return 8

    def _debug_board(self, board):
        """In debug (giữ nguyên)"""
        print("\n=== DEBUG BOARD ===")
        for r in range(board.shape[0]):
            row_str = "".join("." if board[r, c] == 0 else str(board[r, c]) for c in range(board.shape[1]))
            print(f"Row {r}: {row_str}")
        total = board.size
        empty = np.sum(board == 0)
        print(f"Fill ratio: {(total-empty)/total*100:.1f}%")
        print("==================\n")

    def _create_fallback_board(self):
        """Board fallback (giữ nguyên)"""
        board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        if self.grid_size >= 8:
            board[0, :3] = 1
            board[1, 1:4] = 2
            board[2, 2:5] = 3
            board[3, 3:6] = 4
            board[4, 4:7] = 5
            board[6, 1] = 6
            board[7, 2] = 7
        return board

# =============================================================================
# VISION: PIECES (logic phần 2 – valley projection) — NEW
# =============================================================================
class PiecesExtractor:
    """
    Trích xuất 3 initial pieces từ ảnh gốc:
    - Tìm board -> lấy dải bên dưới board
    - Segment 3 bbox pieces
    - Valley-projection để tìm rãnh lưới, suy ra grid 0/1
    - Giữ thành phần lớn nhất (connected components) & trim viền
    """
    def __init__(self, vision_for_board: BlockBlastVision = None):
        self.vision = vision_for_board # dùng board finder của phần 1 nếu có

    def _find_board_region(self, image):
        if self.vision is not None:
            return self.vision._find_board_region(image)
        # fallback đơn giản
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            ar = w / max(1, h)
            area = w * h
            if area < 50000: continue
            if 0.7 <= ar <= 1.3:
                score = area * (1 - abs(1 - ar))
                if score > best_score:
                    best, best_score = (x, y, w, h), score
        if best is None:
            size = int(min(image.shape[:2]) * 0.7)
            x = (image.shape[1] - size) // 2
            y = int(image.shape[0] * 0.15)
            best = (x, y, size, size)
        return best

    def _find_piece_boxes(self, image, board_box):
        H, W = image.shape[:2]
        bx, by, bw, bh = board_box
        y1 = min(H, by + bh + int(0.02 * H))
        x1 = max(0, bx - int(0.25 * bw))
        x2 = min(W, bx + bw + int(0.25 * bw))
        roi = image[y1:H, x1:x2]
        Rh, Rw = roi.shape[:2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # giữ vùng KHÔNG ĐỎ có S,V cao (màu pieces)
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
        boxes = sorted(boxes, key=lambda b: b[0])[:3]
        return boxes

    def _seams_by_projection(self, g: np.ndarray, axis: int):
        # axis=0 -> rãnh ngang (quét theo hàng), axis=1 -> rãnh dọc
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
        # gộp seam gần nhau
        merged = []
        min_sep = max(6, int(g.shape[axis] * 0.06))
        for s in seams:
            if not merged or s - merged[-1] >= min_sep:
                merged.append(s)
            else:
                merged[-1] = int((merged[-1] + s) / 2)
        return [s for s in merged if 3 < s < (g.shape[axis] - 3)]

    def _bounds_simple(self, length, seams):
        """Tạo cặp (start, end) theo seam — phiên bản đơn giản, chính xác."""
        bounds = sorted(list(set([0] + seams + [length])))
        return [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]

    def _decode_grid_from_roi(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Valley-projection -> lưới 0/1 từ một ROI piece."""
        if roi_bgr.shape[0] < 10 or roi_bgr.shape[1] < 10:
            return np.array([[1]])
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        rows = self._seams_by_projection(gray, axis=0)
        cols = self._seams_by_projection(gray, axis=1)
        rb = self._bounds_simple(gray.shape[0], rows)
        cb = self._bounds_simple(gray.shape[1], cols)
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        grid = np.zeros((len(rb), len(cb)), dtype=int)
        for i, (rs, re) in enumerate(rb):
            for j, (cs, ce) in enumerate(cb):
                cell = hsv[rs:re, cs:ce]
                if cell.size == 0: continue
                h, s, v = cv2.split(cell)
                is_piece = ((h > 8) & (h < 165) & (s > 60) & (v > 80)).mean()
                grid[i, j] = 1 if is_piece > 0.40 else 0
        return grid

    def _clean_grid_by_component(self, grid: np.ndarray) -> np.ndarray:
        """Giữ CC lớn nhất để loại nhiễu."""
        if grid.size == 0 or np.sum(grid) == 0:
            return grid
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grid.astype(np.uint8), 4, cv2.CV_32S)
        if num_labels <= 1:
            return grid
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return np.where(labels == largest, 1, 0).astype(int)

    def extract_pieces(self, image_path: str) -> List[np.ndarray]:
        """API chính: trả về đúng 3 piece (grid 0/1)."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        board = self._find_board_region(img)
        boxes = self._find_piece_boxes(img, board)
        pieces = []
        H, W = img.shape[:2]
        for (x, y, w, h) in boxes:
            pad = int(max(w, h) * 0.15)
            xs, ys = max(0, x - pad), max(0, y - pad)
            xe, ye = min(W, x + w + pad), min(H, y + h + pad)
            roi = img[ys:ye, xs:xe]
            grid = self._decode_grid_from_roi(roi)
            clean = self._clean_grid_by_component(grid)
            if clean.size > 0 and np.sum(clean) > 0:
                rr = np.where(clean.sum(axis=1) > 0)[0]
                cc = np.where(clean.sum(axis=0) > 0)[0]
                piece = clean[rr[0]:rr[-1]+1, cc[0]:cc[-1]+1]
            else:
                piece = np.array([[1]])
            pieces.append(piece)
        while len(pieces) < 3:
            pieces.append(np.array([[1]]))
        return pieces[:3]

# =============================================================================
# SOLVER MODULE (Nâng cấp với logic xóa dòng)
# =============================================================================
class BlockBlastSolver:
    """Solver cho Block Blast"""
    def __init__(self, grid_size: int = 8):
        self.grid_size = grid_size

    def solve_with_heuristics(self, board: np.ndarray, pieces: List[np.ndarray]):
        """Chọn move điểm cao, bonus clear line."""
        best_move, best_score = None, -1
        for piece_idx, piece in enumerate(pieces):
            for rot_idx, rot in enumerate(self._get_rotations(piece)):
                for r in range(self.grid_size - rot.shape[0] + 1):
                    for c in range(self.grid_size - rot.shape[1] + 1):
                        if self._can_place(board, rot, r, c):
                            # Tính điểm dựa trên bảng TRƯỚC KHI xóa dòng
                            score = self._calculate_score(board, rot, r, c)
                            if score > best_score:
                                best_score = score
                                # Tạo bảng mới sau khi đặt và xóa dòng
                                board_after = self._place_and_clear(board, rot, r, c)
                                best_move = {
                                    'piece_index': piece_idx,
                                    'rotation': rot_idx,
                                    'position': [r, c],
                                    'score': score,
                                    'board_after': board_after,
                                    'piece_used': rot
                                }
        return best_move

    # === THAY ĐỔI DUY NHẤT ĐỂ VÔ HIỆU HÓA XOAY HÌNH ===
    def _get_rotations(self, piece: np.ndarray):
        """
        SỬA LỖI: Chỉ trả về hình dạng gốc của khối, không tự động xoay.
        """
        return [piece.copy()]

    def _can_place(self, board, piece, row, col):
        for r in range(piece.shape[0]):
            for c in range(piece.shape[1]):
                if piece[r, c] == 1:
                    br, bc = row + r, col + c
                    if br >= self.grid_size or bc >= self.grid_size or board[br, bc] != 0:
                        return False
        return True

    def _calculate_score(self, board, piece, row, col):
        """Tính điểm cho một nước đi, bao gồm điểm xóa dòng và phạt lỗ."""
        # Đặt thử mảnh ghép vào bảng tạm
        b_temp = board.copy()
        piece_size = 0
        for r in range(piece.shape[0]):
            for c in range(piece.shape[1]):
                if piece[r, c] == 1:
                    b_temp[row + r, col + c] = 1
                    piece_size += 1

        # Tính điểm xóa dòng
        score = piece_size # Điểm cơ bản bằng kích thước khối
        cleared_lines = 0
        for r in range(self.grid_size):
            if np.all(b_temp[r, :] != 0):
                cleared_lines += 1
        for c in range(self.grid_size):
            if np.all(b_temp[:, c] != 0):
                cleared_lines += 1
        
        # Bonus lớn cho việc xóa dòng
        if cleared_lines > 0:
            score += (10 * cleared_lines) * cleared_lines # Bonus lũy tiến

        # Phạt nếu tạo ra lỗ hổng (có thể tinh chỉnh)
        score -= self._count_holes(b_temp)
        return score

    def _clear_lines(self, board: np.ndarray) -> np.ndarray:
        """
        HÀM MỚI: Xóa các hàng và cột đã đầy.
        """
        b = board.copy()
        rows_to_clear = [r for r in range(self.grid_size) if np.all(b[r, :] != 0)]
        cols_to_clear = [c for c in range(self.grid_size) if np.all(b[:, c] != 0)]

        if rows_to_clear or cols_to_clear:
            for r in rows_to_clear:
                b[r, :] = 0
            for c in cols_to_clear:
                b[:, c] = 0
        return b

    def _place_and_clear(self, board, piece, row, col):
        """
        HÀM ĐƯỢC CẬP NHẬT: Đặt khối và sau đó xóa các dòng đã hoàn thành.
        """
        b = board.copy()
        # Đặt khối
        for r in range(piece.shape[0]):
            for c in range(piece.shape[1]):
                if piece[r, c] == 1:
                    b[row + r, col + c] = 1
        # Xóa dòng
        return self._clear_lines(b)

    def _count_holes(self, board):
        holes = 0
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if board[r, c] == 0:
                    # Kiểm tra xem ô trống có bị "bao vây" không
                    up = (r == 0) or (board[r-1, c] != 0)
                    down = (r == self.grid_size - 1) or (board[r+1, c] != 0)
                    left = (c == 0) or (board[r, c-1] != 0)
                    right = (c == self.grid_size - 1) or (board[r, c+1] != 0)
                    if up and down and left and right:
                        holes += 1
        return holes

# =============================================================================
# GUI MODULE (Cập nhật logic tìm kiếm giải pháp)
# =============================================================================
class BlockBlastGUI:
    """GUI cho Block Blast Solver"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Block Blast Solver")
        self.root.geometry("1200x800")
        # Áp dụng màu nền cho cửa sổ gốc để thấy rõ thay đổi theme
        try:
            self.root.configure(bg="#eef2ff")
        except Exception:
            pass
        # Phông chữ mặc định dễ đọc hơn
        try:
            default_font = tkfont.nametofont("TkDefaultFont")
            default_font.configure(family="Segoe UI", size=10)
            text_font = tkfont.nametofont("TkTextFont")
            text_font.configure(family="Segoe UI", size=10)
        except Exception:
            pass
        self.current_image_path = None
        self.current_board = None
        self.extracted_pieces = None
        self.piece_labels = []
        self.colors = {
            'background': '#eef2ff',
            'primary': '#2563eb',
            'success': '#16a34a',
            'accent': '#14b8a6',
            'light': '#ffffff',
            'text_primary': '#0f172a',
            'text_secondary': '#475569',
        }
        self.create_widgets()

    def create_widgets(self):
        main = tk.Frame(self.root, bg=self.colors['background'])
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        header = tk.Frame(main, bg=self.colors['background'])
        header.pack(fill=tk.X, pady=(0, 20))
        tk.Label(header, text="Block Blast Solver", font=("Segoe UI", 26, "bold"), fg=self.colors['primary'], bg=self.colors['background']).pack()
        up_frame = tk.Frame(main, bg=self.colors['light'], relief=tk.GROOVE, bd=2, highlightthickness=2, highlightbackground="#c7d2fe")
        up_frame.pack(fill=tk.X, pady=(0, 10))
        inner = tk.Frame(up_frame, bg=self.colors['light'])
        inner.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(inner, text="Upload your Block Blast screenshot", font=("Segoe UI", 12, "bold"), fg=self.colors['text_primary'], bg=self.colors['light']).pack(anchor=tk.W, pady=(0, 8))
        self.upload_btn = tk.Button(inner, text="📁 Choose Image", font=("Segoe UI", 10, "bold"), bg=self.colors['primary'], fg=self.colors['light'], relief=tk.FLAT, bd=0, padx=16, pady=8, cursor="hand2", command=self.select_image, activebackground="#2563eb", activeforeground=self.colors['light'])
        self.upload_btn.pack(pady=(0, 8))
        self._bind_button_hover(self.upload_btn, self.colors['primary'])
        self.image_path_var = tk.StringVar()
        tk.Label(inner, textvariable=self.image_path_var, font=("Arial", 9), fg=self.colors['text_secondary'], bg=self.colors['light'], wraplength=600, justify=tk.LEFT).pack(anchor=tk.W, fill=tk.X)
        content = tk.Frame(main, bg=self.colors['background'])
        content.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        left = tk.Frame(content, bg=self.colors['background'])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        current_frame = tk.Frame(left, bg=self.colors['light'], relief=tk.GROOVE, bd=2, highlightthickness=2, highlightbackground="#c7d2fe")
        current_frame.pack(fill=tk.X, pady=(0, 5))
        current_inner = tk.Frame(current_frame, bg=self.colors['light'])
        current_inner.pack(fill=tk.X, padx=10, pady=8)
        tk.Label(current_inner, text="Current Board", font=("Segoe UI", 11, "bold"), fg=self.colors['text_primary'], bg=self.colors['light']).pack(anchor=tk.W, pady=(0, 5))
        self.current_board_label = tk.Label(current_inner, text="No board yet", font=("Segoe UI", 10), fg=self.colors['text_secondary'], bg=self.colors['light'], relief=tk.FLAT, bd=0, highlightthickness=2, highlightbackground="#e5e7eb")
        self.current_board_label.pack(expand=True)
        pieces_frame = tk.Frame(left, bg=self.colors['background'])
        pieces_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        tk.Label(pieces_frame, text="Initial Pieces", font=("Arial", 12, "bold"), fg=self.colors['text_primary'], bg=self.colors['background']).pack(anchor=tk.W, pady=(0, 8))
        pieces_display = tk.Frame(pieces_frame, bg=self.colors['light'], relief=tk.RAISED, bd=1)
        pieces_display.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        for i in range(3):
            lbl = tk.Label(pieces_display, text=f"Piece {i+1}", font=("Segoe UI", 10), fg=self.colors['text_secondary'], bg=self.colors['light'], width=12, height=4, relief=tk.SUNKEN, bd=1)
            lbl.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.BOTH)
            self.piece_labels.append(lbl)
        # Right column: 3 solutions
        right = tk.Frame(content, bg=self.colors['background'])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        sols = tk.Frame(right, bg=self.colors['background'])
        sols.pack(fill=tk.BOTH, expand=True)
        self.sol_labels = []
        for idx in range(3):
            frame = tk.Frame(sols, bg=self.colors['light'], relief=tk.GROOVE, bd=2, highlightthickness=2, highlightbackground="#c7d2fe")
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0 if idx == 0 else 5, 0 if idx == 2 else 5))
            inner = tk.Frame(frame, bg=self.colors['light'])
            inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            tk.Label(inner, text=f"Solution {idx+1}", font=("Segoe UI", 11, "bold"), fg=self.colors['text_primary'], bg=self.colors['light']).pack(anchor=tk.W, pady=(0, 8))
            lbl = tk.Label(inner, text="No solution yet", font=("Segoe UI", 9), fg=self.colors['text_secondary'], bg=self.colors['light'], height=6, relief=tk.FLAT, bd=0, highlightthickness=2, highlightbackground="#e5e7eb")
            lbl.pack(fill=tk.BOTH, expand=True)
            self.sol_labels.append(lbl)
        footer = tk.Frame(main, bg=self.colors['background'])
        footer.pack(fill=tk.X, pady=(20, 0))
        center = tk.Frame(footer, bg=self.colors['background'])
        center.pack()
        self.solver_btn = tk.Button(center, text="🚀 SOLVER", font=("Segoe UI", 16, "bold"), bg=self.colors['success'], fg=self.colors['light'], relief=tk.FLAT, bd=0, padx=44, pady=16, cursor="hand2", command=self.solve, activebackground="#16a34a", activeforeground=self.colors['light'])
        self.solver_btn.pack()
        self._bind_button_hover(self.solver_btn, self.colors['success'])

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
        def clamp(x):
            return max(0, min(255, x))
        r = clamp(int(r + (255 - r) * (percent / 100.0)))
        g = clamp(int(g + (255 - g) * (percent / 100.0)))
        b = clamp(int(b + (255 - b) * (percent / 100.0)))
        return f"#{r:02x}{g:02x}{b:02x}"

    def select_image(self):
        fp = filedialog.askopenfilename(
            title="Chọn ảnh Block Blast",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if fp:
            self.current_image_path = fp
            self.image_path_var.set(fp)
            print(f"Đã chọn ảnh: {os.path.basename(fp)}")

    # ----------------------------- PIPELINE -----------------------------
    def solve(self):
        """Giải: đọc board + đọc pieces + tìm chuỗi 3 gợi ý TỐI ƯU."""
        if not self.current_image_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn ảnh trước")
            return
        try:
            # 1) Đọc board
            vision = BlockBlastVision(8)
            self.current_board = vision.extract_board_from_image(self.current_image_path)
            self.display_current_board(self.current_board)
            # 2) Đọc 3 initial pieces
            pex = PiecesExtractor(vision_for_board=vision)
            self.extracted_pieces = pex.extract_pieces(self.current_image_path)
            self.display_initial_pieces()
            # 3) Tìm chuỗi solutions tốt nhất
            solutions = self.find_best_sequence_solution(self.current_board, self.extracted_pieces)
            # 4) Hiển thị solutions
            self.display_solutions(solutions)
            messagebox.showinfo("Thành công", "Đã tìm thấy chuỗi nước đi tối ưu!")
        except Exception as e:
            print(f"Lỗi: {e}")
            messagebox.showerror("Lỗi", str(e))

    def find_best_sequence_solution(self, board, pieces):
        """
        HÀM ĐƯỢC VIẾT LẠI HOÀN TOÀN:
        Tìm chuỗi 3 nước đi tốt nhất bằng cách thử mọi hoán vị của các khối.
        """
        solver = BlockBlastSolver(8)
        
        # Gán index cho các khối để theo dõi
        indexed_pieces = list(enumerate(pieces))
        
        best_sequence = []
        best_total_score = -1

        # Thử tất cả 6 hoán vị của 3 khối
        for permutation in itertools.permutations(indexed_pieces):
            current_board = board.copy()
            current_sequence = []
            current_total_score = 0
            
            # Mô phỏng chuỗi nước đi cho hoán vị hiện tại
            for original_index, piece in permutation:
                # Tìm nước đi tốt nhất cho khối này trên bảng hiện tại
                best_move = solver.solve_with_heuristics(current_board, [piece])
                
                if best_move:
                    # Nếu có nước đi, lưu lại và cập nhật bảng
                    current_board = best_move['board_after']
                    current_total_score += best_move['score']
                    current_sequence.append({
                        'piece': best_move['piece_used'],
                        'position': best_move['position'],
                        'score': best_move['score'],
                        'description': f"Piece {original_index+1} at {best_move['position']}"
                    })
                else:
                    # Nếu không đặt được, thêm một nước đi "rỗng"
                    current_sequence.append(None)

            # So sánh với chuỗi tốt nhất đã tìm thấy
            if current_total_score > best_total_score:
                best_total_score = current_total_score
                best_sequence = current_sequence

        print(f"Chuỗi tốt nhất có tổng điểm: {best_total_score}")
        
        # Chuẩn bị kết quả để hiển thị (thêm dummy nếu cần)
        final_solutions = []
        for move in best_sequence:
            if move:
                final_solutions.append(move)
            else:
                final_solutions.append({
                    'piece': np.array([[0]]), 'position': [0, 0], 'score': 0, 'description': "N/A (Không thể đặt)"
                })

        while len(final_solutions) < 3:
            final_solutions.append({
                'piece': np.array([[0]]), 'position': [0, 0], 'score': 0, 'description': "N/A"
            })
            
        return final_solutions[:3]

    # ----------------------------- RENDER -----------------------------
    def display_current_board(self, board):
        """Vẽ board bằng màu xám đậm/nhạt, giữ tỉ lệ."""
        try:
            img = self.create_board_image(board)
            max_size = 220
            w0, h0 = img.size
            ratio = min(max_size / w0, max_size / h0)
            img = img.resize((int(w0 * ratio), int(h0 * ratio)), Image.Resampling.LANCZOS)
            ph = ImageTk.PhotoImage(img)
            self.current_board_label.configure(image=ph, text="")
            self.current_board_label.image = ph
        except Exception as e:
            print(f"Lỗi hiển thị board: {e}")
            self.current_board_label.configure(text=f"Board shape: {board.shape}")

    def display_initial_pieces(self):
        """Hiển thị 3 piece đã extract."""
        pieces = self.extracted_pieces if self.extracted_pieces else [np.array([[1]])]*3
        for i in range(3):
            grid = pieces[i] if i < len(pieces) else np.array([[1]])
            img = self.create_piece_image(grid, cell_size=30)
            
            # === THAY ĐỔI DUY NHẤT: GIẢM KÍCH THƯỚC HIỂN THỊ PIECE ===
            max_size = 60
            
            w0, h0 = img.size
            if w0 == 0 or h0 == 0: continue
            ratio = min(max_size / w0, max_size / h0)
            img = img.resize((int(w0 * ratio), int(h0 * ratio)), Image.Resampling.LANCZOS)
            ph = ImageTk.PhotoImage(img)
            self.piece_labels[i].configure(image=ph, text="")
            self.piece_labels[i].image = ph
        # Hover viền sáng cho pieces
        for lbl in self.piece_labels:
            lbl.bind("<Enter>", lambda e: e.widget.configure(highlightbackground="#a5b4fc"))
            lbl.bind("<Leave>", lambda e: e.widget.configure(highlightbackground="#e5e7eb"))

    def display_solutions(self, solutions):
        # Bảng sẽ được cập nhật tuần tự để hiển thị
        board_for_display = self.current_board.copy()
        solver = BlockBlastSolver(8)

        for i in range(3):
            sol = solutions[i] if i < len(solutions) else None
            lbl = self.sol_labels[i]
            
            if sol is None or np.sum(sol['piece']) == 0:
                lbl.configure(text="Không có nước đi")
                lbl.image = None
                continue

            # Tạo ảnh giải pháp trên bảng hiện tại
            img = self.create_solution_image(board_for_display, sol['piece'], sol['position'])
            max_size = 150
            w0, h0 = img.size
            ratio = min(max_size / w0, max_size / h0)
            img = img.resize((int(w0 * ratio), int(h0 * ratio)), Image.Resampling.LANCZOS)
            ph = ImageTk.PhotoImage(img)
            lbl.configure(image=ph, text="")
            lbl.image = ph
            # Hover viền sáng cho solutions
            lbl.bind("<Enter>", lambda e: e.widget.configure(highlightbackground="#a5b4fc"))
            lbl.bind("<Leave>", lambda e: e.widget.configure(highlightbackground="#e5e7eb"))

            # Cập nhật bảng cho lần hiển thị tiếp theo
            board_for_display = solver._place_and_clear(board_for_display, sol['piece'], *sol['position'])

    # ----------------------------- DRAW HELPERS -----------------------------
    def create_board_image(self, board):
        h, w = board.shape
        cell = 80
        base = np.ones((h * cell, w * cell, 3), dtype=np.uint8) * 255
        for r in range(h):
            for c in range(w):
                color = (220, 220, 220) if board[r, c] == 0 else (80, 80, 80)
                y1, x1 = r * cell, c * cell
                y2, x2 = (r + 1) * cell, (c + 1) * cell
                base[y1:y2, x1:x2] = color
                cv2.rectangle(base, (x1, y1), (x2-1, y2-1), (0, 0, 0), 1)
        # Thêm viền sáng và bóng đổ nhẹ
        margin = 12
        shadow = np.ones((base.shape[0] + margin*2, base.shape[1] + margin*2, 3), dtype=np.uint8) * 239
        # Bóng đổ
        cv2.rectangle(shadow, (margin+6, margin+6), (margin+6 + base.shape[1], margin+6 + base.shape[0]), (210, 214, 220), -1)
        # Nền trắng và viền sáng
        cv2.rectangle(shadow, (margin, margin), (margin + base.shape[1], margin + base.shape[0]), (255, 255, 255), -1)
        cv2.rectangle(shadow, (margin, margin), (margin + base.shape[1], margin + base.shape[0]), (199, 210, 254), 2)
        shadow[margin:margin+base.shape[0], margin:margin+base.shape[1]] = base
        return Image.fromarray(cv2.cvtColor(shadow, cv2.COLOR_BGR2RGB))

    def create_piece_image(self, piece, cell_size=25):
        h, w = piece.shape
        base = np.ones((h * cell_size, w * cell_size, 3), dtype=np.uint8) * 255
        for r in range(h):
            for c in range(w):
                color = (80, 80, 80) if piece[r, c] == 1 else (220, 220, 220)
                y1, x1 = r * cell_size, c * cell_size
                y2, x2 = (r + 1) * cell_size, (c + 1) * cell_size
                base[y1:y2, x1:x2] = color
                cv2.rectangle(base, (x1, y1), (x2-1, y2-1), (0, 0, 0), 1)
        margin = 10
        shadow = np.ones((base.shape[0] + margin*2, base.shape[1] + margin*2, 3), dtype=np.uint8) * 239
        cv2.rectangle(shadow, (margin+5, margin+5), (margin+5 + base.shape[1], margin+5 + base.shape[0]), (210, 214, 220), -1)
        cv2.rectangle(shadow, (margin, margin), (margin + base.shape[1], margin + base.shape[0]), (255, 255, 255), -1)
        cv2.rectangle(shadow, (margin, margin), (margin + base.shape[1], margin + base.shape[0]), (199, 210, 254), 2)
        shadow[margin:margin+base.shape[0], margin:margin+base.shape[1]] = base
        return Image.fromarray(shadow)

    def create_solution_image(self, board, piece, pos):
        """Tạo ảnh board + highlight vị trí piece."""
        b = board.copy()
        r0, c0 = pos
        for r in range(piece.shape[0]):
            for c in range(piece.shape[1]):
                if piece[r, c] == 1:
                    br, bc = r0 + r, c0 + c
                    if 0 <= br < b.shape[0] and 0 <= bc < b.shape[1]:
                        b[br, bc] = 9 # mã highlight
        h, w = b.shape
        cell = 80
        base = np.ones((h * cell, w * cell, 3), dtype=np.uint8) * 255
        for r in range(h):
            for c in range(w):
                if b[r, c] == 0:
                    color = (220, 220, 220)
                elif b[r, c] == 9:
                    color = (0, 150, 0) # Màu xanh lá cho highlight
                else:
                    color = (80, 80, 80)
                y1, x1 = r * cell, c * cell
                y2, x2 = (r + 1) * cell, (c + 1) * cell
                base[y1:y2, x1:x2] = color
                cv2.rectangle(base, (x1, y1), (x2-1, y2-1), (0, 0, 0), 1)
        margin = 12
        shadow = np.ones((base.shape[0] + margin*2, base.shape[1] + margin*2, 3), dtype=np.uint8) * 239
        cv2.rectangle(shadow, (margin+6, margin+6), (margin+6 + base.shape[1], margin+6 + base.shape[0]), (210, 214, 220), -1)
        cv2.rectangle(shadow, (margin, margin), (margin + base.shape[1], margin + base.shape[0]), (255, 255, 255), -1)
        cv2.rectangle(shadow, (margin, margin), (margin + base.shape[1], margin + base.shape[0]), (199, 210, 254), 2)
        shadow[margin:margin+base.shape[0], margin:margin+base.shape[1]] = base
        return Image.fromarray(cv2.cvtColor(shadow, cv2.COLOR_BGR2RGB))

    def run(self):
        self.root.mainloop()

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=== BLOCK BLAST SOLVER — Sequential Simulation ===")
    app = BlockBlastGUI()
    app.run()

if __name__ == "__main__":
    main()