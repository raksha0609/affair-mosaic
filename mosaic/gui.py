# mosaic/gui.py
import os, cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
                             QFileDialog, QProgressBar, QSpinBox, QLineEdit, QMessageBox, QGroupBox, QFormLayout, QComboBox)
from PyQt5.QtGui import QPixmap, QImage

from .pipeline import run_pipeline

def cv2_to_qimage(cv_img):
    if cv_img is None: return QImage()
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)

class PipelineWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def run(self):
        try:
            def prog_cb(v): self.progress.emit(int(v))
            self.status.emit("Running pipeline...")
            res = run_pipeline(
                self.cfg['target'], self.cfg['tiles'], self.cfg['out_image'],
                self.cfg['out_json'], self.cfg['out_csv'],
                grid_n=self.cfg['grid_n'], coarse_width=self.cfg['coarse_width'],
                top_k=self.cfg['top_k'], rotations=self.cfg['rotations'],
                flips=self.cfg['flips'], crop_fracs=self.cfg['crop_fracs'],
                zooms=self.cfg['zooms'], translate_fracs=self.cfg['translate_fracs'],
                metric=self.cfg['metric'], ecc_iter=self.cfg['ecc_iter'], ecc_eps=self.cfg['ecc_eps'],
                alpha=self.cfg['alpha'], resume=self.cfg.get('resume', False),
                progress_callback=prog_cb
            )
            self.finished.emit(res)
        except Exception as e:
            self.status.emit(f"Error: {e}")
            self.finished.emit({})

class MosaicGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AFFAIR Mosaic — Rigid + Hungarian")
        self.setMinimumSize(1000, 700)
        self._build_ui()
        self.worker = None

    def _build_ui(self):
        layout = QVBoxLayout()
        ctrl = QGroupBox("Controls")
        form = QFormLayout()

        self.target_edit = QLineEdit(); bt = QPushButton("Select Target"); bt.clicked.connect(self.select_target)
        self.tiles_edit = QLineEdit(); bt2 = QPushButton("Select Tiles"); bt2.clicked.connect(self.select_tiles)

        self.grid_spin = QSpinBox(); self.grid_spin.setRange(2,60); self.grid_spin.setValue(10)
        self.coarse_width = QLineEdit("200")
        self.topk_spin = QSpinBox(); self.topk_spin.setRange(1,50); self.topk_spin.setValue(5)
        self.ecc_iter_spin = QSpinBox(); self.ecc_iter_spin.setRange(10,500); self.ecc_iter_spin.setValue(100)
        self.alpha_edit = QLineEdit("0.0")
        self.metric_combo = QComboBox(); self.metric_combo.addItems(["ssim","hybrid","mse","lab","ncc"])

        self.crop_edit = QLineEdit("0.0,0.2")
        self.zoom_edit = QLineEdit("1.0,0.8,1.2")
        self.translate_edit = QLineEdit("0.0,0.1,-0.1")

        form.addRow(bt, self.target_edit)
        form.addRow(bt2, self.tiles_edit)
        form.addRow("Grid N:", self.grid_spin)
        form.addRow("Coarse width (px):", self.coarse_width)
        form.addRow("Top-K (coarse):", self.topk_spin)
        form.addRow("ECC iterations:", self.ecc_iter_spin)
        form.addRow("Blend alpha (0..1):", self.alpha_edit)
        form.addRow("Metric:", self.metric_combo)
        form.addRow("Crop fracs:", self.crop_edit)
        form.addRow("Zooms:", self.zoom_edit)
        form.addRow("Translate fracs:", self.translate_edit)
        ctrl.setLayout(form)
        layout.addWidget(ctrl)

        hb = QHBoxLayout()
        self.start_btn = QPushButton("Start"); self.start_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("Cancel"); self.cancel_btn.clicked.connect(self.cancel); self.cancel_btn.setEnabled(False)
        hb.addWidget(self.start_btn); hb.addWidget(self.cancel_btn)
        layout.addLayout(hb)

        self.pbar = QProgressBar(); self.pbar.setValue(0)
        self.status_lbl = QLabel("Idle")
        layout.addWidget(self.pbar); layout.addWidget(self.status_lbl)

        ph = QHBoxLayout()
        self.target_lbl = QLabel("Target"); self.target_lbl.setFixedSize(460,460)
        self.output_lbl = QLabel("Mosaic"); self.output_lbl.setFixedSize(460,460)
        ph.addWidget(self.target_lbl); ph.addWidget(self.output_lbl)
        layout.addLayout(ph)

        self.setLayout(layout)

    def select_target(self):
        p, _ = QFileDialog.getOpenFileName(self, "Target image", ".", "Images (*.png *.jpg *.jpeg)")
        if p:
            self.target_edit.setText(p)
            self._set_preview(p, self.target_lbl)

    def select_tiles(self):
        d = QFileDialog.getExistingDirectory(self, "Tiles folder", ".")
        if d:
            self.tiles_edit.setText(d)

    def _set_preview(self, path, label):
        img = cv2.imread(path)
        if img is None: return
        q = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w,_ = q.shape
        qim = QImage(q.data, w, h, 3*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qim).scaled(label.size(), Qt.KeepAspectRatio)
        label.setPixmap(pix)

    def start(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "Worker already running"); return
        target = self.target_edit.text().strip(); tiles = self.tiles_edit.text().strip()
        if not target or not os.path.exists(target):
            QMessageBox.warning(self, "Error", "Choose valid target"); return
        if not tiles or not os.path.isdir(tiles):
            QMessageBox.warning(self, "Error", "Choose valid tiles folder"); return
        try:
            coarse_w = int(self.coarse_width.text().strip())
            alpha = float(self.alpha_edit.text().strip())
            crop_fracs = tuple(float(x) for x in self.crop_edit.text().split(",") if x.strip()!="")
            zooms = tuple(float(x) for x in self.zoom_edit.text().split(",") if x.strip()!="")
            translate_fracs = tuple(float(x) for x in self.translate_edit.text().split(",") if x.strip()!="")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid inputs: {e}"); return

        cfg = {
            "target": target, "tiles": tiles,
            "out_image": os.path.join("results","mosaics","mosaic_gui_result.jpg"),
            "out_json": os.path.join("results","params","mosaic_gui_params.json"),
            "out_csv": os.path.join("results","params","mosaic_gui_params.csv"),
            "grid_n": int(self.grid_spin.value()), "coarse_width": coarse_w, "top_k": int(self.topk_spin.value()),
            "rotations": (0,90,180,270), "flips": (False, True),
            "crop_fracs": crop_fracs, "zooms": zooms, "translate_fracs": translate_fracs,
            "metric": str(self.metric_combo.currentText()), "ecc_iter": int(self.ecc_iter_spin.value()),
            "ecc_eps": 1e-5, "alpha": alpha, "resume": False
        }
        os.makedirs(os.path.dirname(cfg["out_image"]), exist_ok=True); os.makedirs(os.path.dirname(cfg["out_json"]), exist_ok=True)

        self.start_btn.setEnabled(False); self.cancel_btn.setEnabled(True)
        self.status_lbl.setText("Starting pipeline..."); self.pbar.setValue(0)
        self.worker = PipelineWorker(cfg)
        self.worker.progress.connect(lambda v: self.pbar.setValue(int(v)))
        self.worker.status.connect(lambda s: self.status_lbl.setText(s))
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def cancel(self):
        if self.worker:
            self.worker.terminate()
            self.status_lbl.setText("Cancelled")
            self.start_btn.setEnabled(True); self.cancel_btn.setEnabled(False)

    def on_finished(self, res):
        self.start_btn.setEnabled(True); self.cancel_btn.setEnabled(False)
        if res and res.get("mosaic_path") and os.path.exists(res["mosaic_path"]):
            self._set_preview(res["mosaic_path"], self.output_lbl)
            QMessageBox.information(self, "Done", f"Mosaic saved to {res['mosaic_path']}\nTime: {res.get('time_s',0):.1f}s")
        else:
            QMessageBox.warning(self, "Finished", "No mosaic produced (see status).")
