# mosaic/gui.py
import os
import json
import cv2
import traceback
from glob import glob

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QProgressBar, QSpinBox, QLineEdit, QMessageBox, QGroupBox,
    QFormLayout, QComboBox, QCheckBox, QDialog, QListWidget, QListWidgetItem,
    QDialogButtonBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont

from .pipeline import run_pipeline

CONFIG_PATH = "config/defaults.json"


def load_defaults():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_defaults(data):
    os.makedirs(os.path.dirname(CONFIG_PATH) or ".", exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


class MultiSelectDialog(QDialog):
    def __init__(self, title, options, selected=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(260)
        self.selected_set = set(selected or [])
        v = QVBoxLayout()
        self.listw = QListWidget()
        self.listw.setSelectionMode(QListWidget.NoSelection)
        for opt in options:
            it = QListWidgetItem(str(opt))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if str(opt) in self.selected_set else Qt.Unchecked)
            self.listw.addItem(it)
        v.addWidget(self.listw)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.setLayout(v)

    def selected(self):
        out = []
        for i in range(self.listw.count()):
            it = self.listw.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.text())
        return out


class MultiSelectField(QWidget):
    def __init__(self, label, options, selected=None, parent=None):
        super().__init__(parent)
        self.options = [str(x) for x in options]
        self.selected = [str(x) for x in (selected or [])]
        self._build_ui(label)

    def _build_ui(self, label):
        h = QHBoxLayout()
        self.line = QLineEdit(", ".join(self.selected))
        self.line.setReadOnly(True)
        self.btn = QPushButton("Select...")
        self.btn.setFixedWidth(90)
        self.btn.clicked.connect(self.open_dialog)
        h.addWidget(self.line)
        h.addWidget(self.btn)
        h.setContentsMargins(0, 0, 0, 0)
        self.setLayout(h)

    def open_dialog(self):
        dlg = MultiSelectDialog("Select values", self.options, selected=self.selected, parent=self)
        if dlg.exec_():
            self.selected = dlg.selected()
            self.line.setText(", ".join(self.selected))

    def get_values(self):
        if not self.line.text().strip():
            return tuple()
        return tuple(x.strip() for x in self.line.text().split(",") if x.strip())


class PipelineWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def run(self):
        try:
            def prog_cb(v):
                try:
                    vv = int(v)
                except Exception:
                    vv = 0
                vv = max(0, min(100, vv))
                self.progress.emit(vv)

            self.status.emit("Running pipeline...")
            res = run_pipeline(
                self.cfg['target'], self.cfg['tiles'], self.cfg['out_image'],
                self.cfg['out_json'], self.cfg['out_csv'],
                grid_n=self.cfg['grid_n'], coarse_width=self.cfg['coarse_width'],
                top_k=self.cfg['top_k'], rotations=self.cfg['rotations'],
                flips=self.cfg['flips'], crop_fracs=self.cfg['crop_fracs'],
                zooms=self.cfg['zooms'], translate_fracs=self.cfg['translate_fracs'],
                metric=self.cfg['metric'], ecc_iter=self.cfg['ecc_iter'],
                ecc_eps=self.cfg['ecc_eps'], alpha=self.cfg['alpha'],
                resume=self.cfg['resume'], project_path=self.cfg.get('project_path', None),
                progress_callback=prog_cb
            )
            # ensure final progress is 100
            self.progress.emit(100)
            self.finished.emit(res or {})
        except Exception as e:
            tb = traceback.format_exc()
            self.status.emit(f"Error: {e}")
            # include traceback in returned dict so GUI can present it
            self.finished.emit({"_error": str(e), "_traceback": tb})


class MosaicGUI(QWidget):
    def __init__(self):
        super().__init__()
        font = QFont()
        font.setPointSize(8)
        self.setFont(font)

        self.defaults = load_defaults()
        self.setWindowTitle("AFFAIR Mosaic — Configurable Parameters")
        self.setMinimumSize(1080, 760)
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        ctrl = QGroupBox("Controls")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        d = self.defaults

        # Files
        self.target_edit = QLineEdit()
        bt_target = QPushButton("Select Target")
        bt_target.clicked.connect(self.select_target)
        self.tiles_edit = QLineEdit()
        bt_tiles = QPushButton("Select Tiles Folder")
        bt_tiles.clicked.connect(self.select_tiles)

        self.project_path_edit = QLineEdit(d.get("project_path", "results/project/affair_project.json"))
        bt_project = QPushButton("Select Project File")
        bt_project.clicked.connect(self.select_project)
        self.resume_check = QCheckBox("Resume Previous Run")
        self.resume_check.setChecked(d.get("resume", False))

        # Parameters
        self.grid_spin = QSpinBox(); self.grid_spin.setRange(2, 60); self.grid_spin.setValue(d.get("grid_n", 10))
        self.coarse_width = QLineEdit(str(d.get("coarse_width", 200)))
        self.topk_spin = QSpinBox(); self.topk_spin.setRange(1, 50); self.topk_spin.setValue(d.get("top_k", 5))
        self.ecc_iter_spin = QSpinBox(); self.ecc_iter_spin.setRange(10, 2000); self.ecc_iter_spin.setValue(d.get("ecc_iter", 100))
        self.alpha_edit = QLineEdit(str(d.get("alpha", 0.0)))

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["ssim", "hybrid", "mse", "lab", "ncc"])
        self.metric_combo.setCurrentText(d.get("metric", "hybrid"))

        # Multiselects (default includes vertical)
        rot_opts = [0, 90, 180, 270]
        flips_opts = ["None", "Horizontal", "Vertical"]
        crop_opts = [0.0, 0.05, 0.1, 0.2]
        zoom_opts = [0.8, 1.0, 1.2]
        trans_opts = [-0.1, 0.0, 0.1]

        self.rot_field = MultiSelectField("Rotations", rot_opts, selected=[str(x) for x in d.get("rotations", rot_opts)])
        self.flips_field = MultiSelectField("Flips", flips_opts, selected=d.get("flips", ["None"]))
        self.crop_field = MultiSelectField("Crop fracs", crop_opts, selected=[str(x) for x in d.get("crop_fracs", [0.0, 0.2])])
        self.zoom_field = MultiSelectField("Zooms", zoom_opts, selected=[str(x) for x in d.get("zooms", [1.0])])
        self.trans_field = MultiSelectField("Translate fracs", trans_opts, selected=[str(x) for x in d.get("translate_fracs", [0.0])])

        # Layout composition
        form.addRow(bt_target, self.target_edit)
        form.addRow(bt_tiles, self.tiles_edit)
        form.addRow(bt_project, self.project_path_edit)
        form.addRow("Resume:", self.resume_check)
        form.addRow("Grid N:", self.grid_spin)
        form.addRow("Coarse Width:", self.coarse_width)
        form.addRow("Top-K:", self.topk_spin)
        form.addRow("ECC Iter:", self.ecc_iter_spin)
        form.addRow("Blend Alpha:", self.alpha_edit)
        form.addRow("Metric:", self.metric_combo)

        form.addRow("Rotations:", self.rot_field)
        form.addRow("Flips:", self.flips_field)
        form.addRow("Crop Fracs:", self.crop_field)
        form.addRow("Zooms:", self.zoom_field)
        form.addRow("Translate Fracs:", self.trans_field)

        # Save/Load
        btns = QHBoxLayout()
        self.save_btn = QPushButton("Save Parameters")
        self.load_btn = QPushButton("Load Parameters")
        self.save_btn.clicked.connect(self.save_params)
        self.load_btn.clicked.connect(self.load_params)
        btns.addWidget(self.save_btn)
        btns.addWidget(self.load_btn)
        form.addRow(btns)

        ctrl.setLayout(form)
        layout.addWidget(ctrl)

        # Buttons + progress
        hb = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn.clicked.connect(self.cancel)
        hb.addWidget(self.start_btn)
        hb.addWidget(self.cancel_btn)
        layout.addLayout(hb)

        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        self.status_lbl = QLabel("Idle")
        layout.addWidget(self.pbar)
        layout.addWidget(self.status_lbl)

        # Preview
        ph = QHBoxLayout()
        self.target_lbl = QLabel("Target")
        self.output_lbl = QLabel("Mosaic")
        for lbl in [self.target_lbl, self.output_lbl]:
            lbl.setFixedSize(480, 480)
            lbl.setAlignment(Qt.AlignCenter)
        ph.addWidget(self.target_lbl)
        ph.addWidget(self.output_lbl)
        layout.addLayout(ph)

        self.setLayout(layout)

    # -----------------------
    # File selection helpers
    # -----------------------
    def select_target(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Target Image", ".", "Images (*.png *.jpg *.jpeg)")
        if p:
            self.target_edit.setText(p)
            self._set_preview(p, self.target_lbl)

    def select_tiles(self):
        d = QFileDialog.getExistingDirectory(self, "Select Tiles Folder", ".")
        if d:
            self.tiles_edit.setText(d)

    def select_project(self):
        f, _ = QFileDialog.getSaveFileName(self, "Select Project File", "results/project/affair_project.json", "JSON (*.json)")
        if f:
            self.project_path_edit.setText(f)

    def _set_preview(self, path, label):
        img = cv2.imread(path)
        if img is None:
            label.setText("No preview")
            return
        q = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = q.shape
        qim = QImage(q.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qim).scaled(label.size(), Qt.KeepAspectRatio)
        label.setPixmap(pix)

    # -----------------------
    # Save / Load defaults
    # -----------------------
    def save_params(self):
        params = {
            "grid_n": self.grid_spin.value(),
            "coarse_width": int(self.coarse_width.text()),
            "top_k": self.topk_spin.value(),
            "ecc_iter": self.ecc_iter_spin.value(),
            "alpha": float(self.alpha_edit.text()),
            "metric": self.metric_combo.currentText(),
            "rotations": list(self.rot_field.get_values()),
            "flips": list(self.flips_field.get_values()),
            "crop_fracs": list(self.crop_field.get_values()),
            "zooms": list(self.zoom_field.get_values()),
            "translate_fracs": list(self.trans_field.get_values()),
            "project_path": self.project_path_edit.text().strip()
        }
        save_defaults(params)
        QMessageBox.information(self, "Saved", f"Parameters saved to {CONFIG_PATH}")

    def load_params(self):
        d = load_defaults()
        if not d:
            QMessageBox.warning(self, "No file", "No defaults.json found.")
            return
        self.grid_spin.setValue(d.get("grid_n", 10))
        self.coarse_width.setText(str(d.get("coarse_width", 200)))
        self.topk_spin.setValue(d.get("top_k", 5))
        self.ecc_iter_spin.setValue(d.get("ecc_iter", 100))
        self.alpha_edit.setText(str(d.get("alpha", 0.0)))
        self.metric_combo.setCurrentText(d.get("metric", "hybrid"))
        self.rot_field.line.setText(", ".join(map(str, d.get("rotations", []))))
        self.flips_field.line.setText(", ".join(map(str, d.get("flips", []))))
        self.crop_field.line.setText(", ".join(map(str, d.get("crop_fracs", []))))
        self.zoom_field.line.setText(", ".join(map(str, d.get("zooms", []))))
        self.trans_field.line.setText(", ".join(map(str, d.get("translate_fracs", []))))
        self.project_path_edit.setText(d.get("project_path", "results/project/affair_project.json"))
        QMessageBox.information(self, "Loaded", "Defaults loaded successfully.")

    # -----------------------
    # Execution
    # -----------------------
    def start(self):
        target = self.target_edit.text().strip()
        tiles = self.tiles_edit.text().strip()
        project_path = self.project_path_edit.text().strip()
        resume = self.resume_check.isChecked()

        # basic validation
        if not target or not os.path.exists(target):
            QMessageBox.warning(self, "Error", "Choose a valid target file")
            return
        if not tiles or not os.path.isdir(tiles):
            QMessageBox.warning(self, "Error", "Choose a valid tiles folder")
            return

        # check tiles folder contains readable images
        exts = ("*.jpg", "*.png", "*.jpeg")
        found = []
        for e in exts:
            found.extend(glob(os.path.join(tiles, e)))
        if not found:
            QMessageBox.warning(self, "Error", "No tile images found in tiles folder")
            return

        # resume sanity checks
        if resume:
            if not project_path or not os.path.exists(project_path):
                QMessageBox.warning(self, "Resume Warning", "Project file not found. Starting new run.")
                resume = False
            else:
                try:
                    with open(project_path, "r") as f:
                        pj = json.load(f)
                    prev_meta = pj.get("_meta_", {})
                    prev_target = os.path.basename(prev_meta.get("target", "")) if isinstance(prev_meta, dict) else ""
                    if prev_target and prev_target != os.path.basename(target):
                        reply = QMessageBox.question(self, "Resume mismatch",
                                                     "Project file seems for a different target. Resume anyway?")
                        if reply != QMessageBox.Yes:
                            resume = False
                except Exception:
                    QMessageBox.warning(self, "Resume Error", "Could not read project file. Starting new run.")
                    resume = False

        # parse parameters (with safe defaults)
        try:
            coarse_w = int(self.coarse_width.text())
            alpha = float(self.alpha_edit.text())
            rotations = tuple(int(x) for x in self.rot_field.get_values()) if self.rot_field.get_values() else (0, 90, 180, 270)
            flips_raw = self.flips_field.get_values()
            flips = tuple(flips_raw) if flips_raw else ("None", "Horizontal", "Vertical")
            crop_fracs = tuple(float(x) for x in self.crop_field.get_values()) if self.crop_field.get_values() else (0.0, 0.2)
            zooms = tuple(float(x) for x in self.zoom_field.get_values()) if self.zoom_field.get_values() else (1.0,)
            translate_fracs = tuple(float(x) for x in self.trans_field.get_values()) if self.trans_field.get_values() else (0.0,)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid parameter value: {e}")
            return

        cfg = {
            "target": target, "tiles": tiles,
            "out_image": os.path.join("results", "mosaics", "mosaic_gui_result.jpg"),
            "out_json": os.path.join("results", "params", "mosaic_gui_params.json"),
            "out_csv": os.path.join("results", "params", "mosaic_gui_params.csv"),
            "grid_n": int(self.grid_spin.value()), "coarse_width": coarse_w,
            "top_k": int(self.topk_spin.value()), "rotations": rotations,
            "flips": flips,
            "crop_fracs": crop_fracs, "zooms": zooms, "translate_fracs": translate_fracs,
            "metric": self.metric_combo.currentText(),
            "ecc_iter": int(self.ecc_iter_spin.value()), "ecc_eps": 1e-5,
            "alpha": alpha, "resume": resume,
            "project_path": project_path
        }

        # ensure output folders exist
        os.makedirs(os.path.dirname(cfg["out_image"]), exist_ok=True)
        os.makedirs(os.path.dirname(cfg["out_json"]), exist_ok=True)
        if cfg["project_path"]:
            os.makedirs(os.path.dirname(cfg["project_path"]), exist_ok=True)

        # start worker
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status_lbl.setText("Starting pipeline...")
        self.pbar.setValue(1)

        self.worker = PipelineWorker(cfg)
        self.worker.progress.connect(lambda v: self.pbar.setValue(int(v)))
        self.worker.status.connect(lambda s: self.status_lbl.setText(s))
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def cancel(self):
        if self.worker:
            self.worker.terminate()
            self.status_lbl.setText("Cancelled")
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def on_finished(self, res):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        # If pipeline returned an error dict, show it (traceback included)
        if res is None:
            res = {}
        if "_error" in res:
            tb = res.get("_traceback", "")
            QMessageBox.critical(self, "Pipeline Error", f"Pipeline raised an exception:\n{res.get('_error')}\n\nTraceback (first lines):\n{''.join(tb.splitlines(True)[:10])}")
            self.status_lbl.setText("Error")
            return

        mosaic_path = res.get("mosaic_path")
        params_json = res.get("params_json") or res.get("params")
        runtime = res.get("time_s", None)

        # check mosaic file presence
        if mosaic_path and os.path.exists(mosaic_path):
            self._set_preview(mosaic_path, self.output_lbl)
            self.status_lbl.setText("Done")
            msg = f"Mosaic saved to {mosaic_path}"
            if runtime:
                msg += f"\nTime: {runtime:.1f}s"
            QMessageBox.information(self, "Done", msg)
            self.pbar.setValue(100)
            return

        # --- No mosaic file: gather debug info and show helpful message
        debug_lines = []
        debug_lines.append("No mosaic file found after pipeline run.")

        # check project file
        project_path = self.project_path_edit.text().strip()
        if project_path and os.path.exists(project_path):
            try:
                with open(project_path, "r") as f:
                    pj = json.load(f)
                candidates = pj.get("candidates", pj.get("cands", []))
                debug_lines.append(f"Project file exists: {project_path}")
                debug_lines.append(f"Candidate slots: {len(candidates)}")
                # sample few candidate counts
                non_empty = sum(1 for c in candidates if c)
                debug_lines.append(f"Grid patches with candidates: {non_empty}/{len(candidates)}")
            except Exception as e:
                debug_lines.append(f"Could not parse project file: {e}")
        else:
            debug_lines.append("No project file saved (or path invalid).")

        # check tiles folder content
        tiles = self.tiles_edit.text().strip()
        if tiles and os.path.isdir(tiles):
            imgs = []
            for ext in ("*.jpg", "*.png", "*.jpeg"):
                imgs.extend(glob(os.path.join(tiles, ext)))
            debug_lines.append(f"Tiles found: {len(imgs)}")
        else:
            debug_lines.append("Tiles folder missing or invalid.")

        # final user message
        dbgmsg = "\n".join(debug_lines[:20])
        QMessageBox.warning(self, "No mosaic produced", f"No mosaic produced. Debug:\n{dbgmsg}\n\nCheck console/log for more details.")
        self.status_lbl.setText("Finished (no mosaic)")
        # ensure progress bar reaches at least 100 visually
        self.pbar.setValue(100)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = MosaicGUI()
    win.show()
    sys.exit(app.exec_())
