import math
import os
import random
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


def compute_iou(box_a, box_b):
    """
    IoU dla dwóch boxów (x, y, w, h).
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    yB = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    if area_a <= 0 or area_b <= 0:
        return 0.0

    return inter_area / float(area_a + area_b - inter_area + 1e-16)

def xyxy_to_xywh(x1, y1, x2, y2):
    """
    (x1, y1, x2, y2) -> (x, y, w, h).
    """
    return (x1, y1, x2 - x1, y2 - y1)

def yolo_line(cls_id, cx, cy, w, h):
    """
    Zwraca linię YOLO: 'cls cx cy w h'
    """
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# ----------------------------------------------------------------------------
# Klasa filtra cząsteczkowego z twardym resetem, buforem 9 pomiarów,
# potrójnym update i zaostrzonymi parametrami
# ----------------------------------------------------------------------------
class ParticleFilter:
    def __init__(self, num_particles, img_width, img_height):
        self.num_particles = num_particles
        self.img_width = img_width
        self.img_height = img_height

        # Zaostrzamy parametry
        self.sigma_move = 2.0      # mniejszy rozrzut predykcji
        self.sigma_measure = 5.0   # większa "ufność" w pomiar YOLO

        self.particles = np.zeros((self.num_particles, 2), dtype=np.float32)
        self.weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles
        self.inited = False

        # Bufor do 9 ostatnich (cx, cy) z YOLO
        self.yolo_history = deque(maxlen=9)

    def init_around(self, cx, cy, spread=30.0):
        """
        Inicjalizacja w okolicy (cx, cy).
        """
        self.particles[:, 0] = np.random.normal(cx, spread, self.num_particles)
        self.particles[:, 1] = np.random.normal(cy, spread, self.num_particles)
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.img_width - 1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.img_height - 1)
        self.weights.fill(1.0 / self.num_particles)
        self.inited = True

    def predict(self):
        """
        Prosty model ruchu (Gauss(0, sigma_move)).
        """
        if not self.inited:
            return
        dx = np.random.normal(0, self.sigma_move, self.num_particles)
        dy = np.random.normal(0, self.sigma_move, self.num_particles)
        self.particles[:, 0] += dx
        self.particles[:, 1] += dy
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.img_width - 1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.img_height - 1)

    def update(self, cx, cy):
        """
        Aktualizacja wag i resampling.
        """
        if not self.inited:
            return
        dx = self.particles[:, 0] - cx
        dy = self.particles[:, 1] - cy
        dist_sq = dx*dx + dy*dy

        # Bardziej agresywne "zbicie" cząstek
        w = np.exp(-dist_sq / (2.0*(self.sigma_measure**2))) + 1e-12
        self.weights = w / np.sum(w)
        self.resample()

    def resample(self):
        cum_weights = np.cumsum(self.weights)
        step = 1.0 / self.num_particles
        r = random.random() * step
        new_parts = np.zeros_like(self.particles)
        idx = 0
        for i in range(self.num_particles):
            u = r + i*step
            while u > cum_weights[idx] and idx < (self.num_particles - 1):
                idx += 1
            new_parts[i] = self.particles[idx]
        self.particles = new_parts
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        """
        Zwraca (x_mean, y_mean).
        """
        if not self.inited:
            return None, None
        xm = np.mean(self.particles[:, 0])
        ym = np.mean(self.particles[:, 1])
        return xm, ym

    def hard_reset(self):
        """
        Twardy reset:
         - inited=False
         - init_around ostatnim pomiarem
         - potrójny update (predict+update) dla każdego z yolo_history
        """
        self.inited = False
        if len(self.yolo_history) == 0:
            return

        cx_ref, cy_ref = self.yolo_history[-1]
        self.init_around(cx_ref, cy_ref, spread=30.0)

        # Każdy (cx_i, cy_i) => 3-krotnie predict+update
        for (cx_i, cy_i) in self.yolo_history:
            for _ in range(3):
                self.predict()
                self.update(cx_i, cy_i)

    def update_with_box(self, cx, cy, yolo_box):
        """
        Główna metoda w pętli:
          1) dodaj (cx, cy) do bufora
          2) jeśli nieinited => init
          3) predict+update
          4) sprawdzamy, czy PF "wyszedł" poza bounding box YOLO
             => hard_reset
          5) zwracamy True, jeśli nastąpił restart
        """
        restarted = False

        # Bufor
        self.yolo_history.append((cx, cy))

        if not self.inited:
            self.init_around(cx, cy, spread=30.0)

        # Normalny krok
        self.predict()
        self.update(cx, cy)

        # Sprawdzamy, czy PF jest w bounding box YOLO
        x_est, y_est = self.estimate()
        if x_est is None:
            return False

        x_box, y_box, w_box, h_box = yolo_box
        if not (x_box <= x_est <= x_box + w_box and y_box <= y_est <= y_box + h_box):
            # => twardy reset
            self.hard_reset()
            restarted = True

        return restarted


# ---------------------------
# Główny kod: YOLO + Tracker + PF (hard reset)
# ---------------------------
def main():
    videos_dir = r"E:\Cataract\videos\micro"
    output_dir = "output_hybrid_yolo_tracker_pf_hardreset"
    os.makedirs(output_dir, exist_ok=True)

    # Główne foldery
    raw_dir = os.path.join(output_dir, "raw")
    annotated_dir = os.path.join(output_dir, "annotated")
    ann_yolo_dir = os.path.join(annotated_dir, "yolo")
    ann_tracker_dir = os.path.join(annotated_dir, "tracker")
    ann_pf_dir = os.path.join(annotated_dir, "pf")

    # to_check/tracker
    to_check_dir = os.path.join(output_dir, "to_check")
    to_check_tracker_dir = os.path.join(to_check_dir, "tracker")
    tctr_raw_dir = os.path.join(to_check_tracker_dir, "raw")
    tctr_ann_dir = os.path.join(to_check_tracker_dir, "annotated")
    tctr_yolo_dir = os.path.join(tctr_ann_dir, "yolo")
    tctr_tracker_dir = os.path.join(tctr_ann_dir, "tracker")
    tctr_pf_dir = os.path.join(tctr_ann_dir, "pf")

    # to_check/pf
    to_check_pf_dir = os.path.join(to_check_dir, "pf")
    tcpf_raw_dir = os.path.join(to_check_pf_dir, "raw")
    tcpf_ann_dir = os.path.join(to_check_pf_dir, "annotated")
    tcpf_yolo_dir = os.path.join(tcpf_ann_dir, "yolo")
    tcpf_tracker_dir = os.path.join(tcpf_ann_dir, "tracker")
    tcpf_pf_dir = os.path.join(tcpf_ann_dir, "pf")

    for folder in [
        raw_dir, annotated_dir, ann_yolo_dir, ann_tracker_dir, ann_pf_dir,
        to_check_dir, to_check_tracker_dir, tctr_raw_dir, tctr_ann_dir, tctr_yolo_dir, tctr_tracker_dir, tctr_pf_dir,
        to_check_pf_dir, tcpf_raw_dir, tcpf_ann_dir, tcpf_yolo_dir, tcpf_tracker_dir, tcpf_pf_dir
    ]:
        os.makedirs(folder, exist_ok=True)

    # Parametry
    conf_threshold = 0.80
    iou_threshold = 0.4
    keyframe_interval = 30
    skip_frames = 1
    pf_box_size = 50.0

    model_path = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Annotators\Yolo\surgical_tool_detection\exp14\weights\best.pt"
    model = YOLO(model_path)

    def save_frame_and_txts(frame_disp, frame,
                            box_yolo, box_tracker, box_pf,
                            best_cls, width, height,
                            raw_dirX, ann_dirX, yolo_dirX, tracker_dirX, pf_dirX,
                            video_name, frame_idx):
        """
        Zapis klatki i plików .txt
        """
        frame_prefix = f"{video_name}_frame_{frame_idx:07d}"
        raw_path = os.path.join(raw_dirX, f"{frame_prefix}.jpg")
        ann_path = os.path.join(ann_dirX, f"{frame_prefix}.jpg")

        yolo_txt_path = os.path.join(yolo_dirX, f"{frame_prefix}.txt")
        tracker_txt_path = os.path.join(tracker_dirX, f"{frame_prefix}.txt")
        pf_txt_path = os.path.join(pf_dirX, f"{frame_prefix}.txt")

        # 1) raw
        cv2.imwrite(raw_path, frame)
        # 2) annotated
        cv2.imwrite(ann_path, frame_disp)

        # YOLO
        x, y, w, h = box_yolo
        cx_norm = (x + w/2) / width
        cy_norm = (y + h/2) / height
        w_norm = w / width
        h_norm = h / height
        line_yolo = yolo_line(best_cls, cx_norm, cy_norm, w_norm, h_norm)
        with open(yolo_txt_path, "w") as fy:
            fy.write(line_yolo + "\n")

        # Tracker (klasa = 99)
        line_tracker = ""
        if box_tracker is not None:
            tx, ty, tw, th = box_tracker
            cx_t = (tx + tw/2) / width
            cy_t = (ty + th/2) / height
            w_t = tw / width
            h_t = th / height
            line_tracker = yolo_line(99, cx_t, cy_t, w_t, h_t)
        if line_tracker:
            with open(tracker_txt_path, "w") as ftr:
                ftr.write(line_tracker + "\n")

        # PF (klasa = 98)
        x_pf, y_pf, w_pf, h_pf = box_pf
        cx_pf = (x_pf + w_pf/2) / width
        cy_pf = (y_pf + h_pf/2) / height
        w_pf_norm = w_pf / width
        h_pf_norm = h_pf / height
        line_pf = yolo_line(98, cx_pf, cy_pf, w_pf_norm, h_pf_norm)
        with open(pf_txt_path, "w") as fpf:
            fpf.write(line_pf + "\n")

    for video_file in os.listdir(videos_dir):
        if not video_file.lower().endswith(".mp4"):
            continue

        video_path = os.path.join(videos_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        print(f"\nProcessing: {video_name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open {video_file}, skipping...")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Tracker
        tracker = None
        tracker_active = False
        last_reinit_frame = -9999
        tracked_box = None

        # Filtr cząsteczkowy
        pf = ParticleFilter(num_particles=300,
                            img_width=width,
                            img_height=height)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % skip_frames != 0:
                continue

            frame_disp = frame.copy()

            # YOLO
            results = model(frame)
            if len(results) == 0:
                detections = []
            else:
                boxes_xyxy = results[0].boxes.xyxy
                confs = results[0].boxes.conf
                classes = results[0].boxes.cls

                detections = []
                for i in range(len(boxes_xyxy)):
                    c = float(confs[i])
                    if c < conf_threshold:
                        continue
                    x1, y1, x2, y2 = boxes_xyxy[i].tolist()
                    cls_id = int(classes[i])
                    xywh = xyxy_to_xywh(x1, y1, x2, y2)
                    detections.append((xywh, c, cls_id))

            # Tracker update
            if tracker_active and tracker is not None:
                success, new_box = tracker.update(frame)
                if success:
                    tracked_box = new_box
                else:
                    tracker_active = False
                    tracked_box = None

            # PF predict
            pf.predict()

            if len(detections) == 0:
                # brak YOLO => pomijamy klatkę
                continue
            else:
                # Najlepsza
                best_box, best_conf, best_cls = max(detections, key=lambda d: d[1])
                x, y, w, h = best_box
                x2, y2 = x + w, y + h

                # Rys. YOLO
                cv2.rectangle(frame_disp, (int(x), int(y)), (int(x2), int(y2)),
                              (0,255,0), 2)
                label_text = f"YOLO cls={best_cls}, conf={best_conf:.2f}"
                cv2.putText(frame_disp, label_text, (int(x), int(y)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # Tracker init / re-init
                if not tracker_active:
                    tracker = cv2.legacy.TrackerMOSSE_create()
                    tracker.init(frame, best_box)
                    tracker_active = True
                    last_reinit_frame = frame_idx
                    tracked_box = best_box
                elif (frame_idx - last_reinit_frame) >= keyframe_interval:
                    tracker = cv2.legacy.TrackerMOSSE_create()
                    tracker.init(frame, best_box)
                    tracker_active = True
                    last_reinit_frame = frame_idx
                    tracked_box = best_box

                # PF update_with_box
                cx_yolo = x + w/2
                cy_yolo = y + h/2
                did_restart = pf.update_with_box(cx_yolo, cy_yolo, (x, y, w, h))

                # Rysujemy PF
                x_est, y_est = pf.estimate()
                if x_est is None:
                    continue
                half_size = pf_box_size/2
                x_pf = x_est - half_size
                y_pf = y_est - half_size
                w_pf = pf_box_size
                h_pf = pf_box_size
                cv2.rectangle(frame_disp, (int(x_pf), int(y_pf)),
                              (int(x_pf+w_pf), int(y_pf+h_pf)),
                              (0,0,255), 2)
                cv2.putText(frame_disp, "PF", (int(x_pf), int(y_pf)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                # Tracker bounding box (niebieski)
                if tracker_active and tracked_box is not None:
                    tx, ty, tw, th = tracked_box
                    cv2.rectangle(frame_disp, (int(tx), int(ty)),
                                  (int(tx+tw), int(ty+th)),
                                  (255,0,0), 2)
                    cv2.putText(frame_disp, "Tracker", (int(tx), int(ty)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

                # Mismatch tracker => IoU < iou_threshold
                mismatch_tracker = False
                if not tracker_active or tracked_box is None:
                    mismatch_tracker = True
                else:
                    iou_tr = compute_iou((x, y, w, h), tracked_box)
                    if iou_tr < iou_threshold:
                        mismatch_tracker = True

                # Mismatch PF => twardy reset => did_restart = True
                mismatch_pf = did_restart

                # Zapis do folderów
                def save_stuff(rdir, adir, ydir, tdir, pfdir):
                    # YOLO box
                    box_yolo = (x, y, w, h)
                    # tracker box
                    box_tr = tracked_box if (tracker_active and tracked_box) else None
                    # pf box
                    box_pf = (x_pf, y_pf, w_pf, h_pf)

                    save_frame_and_txts(
                        frame_disp, frame,
                        box_yolo, box_tr, box_pf,
                        best_cls, width, height,
                        rdir, adir, ydir, tdir, pfdir,
                        video_name, frame_idx
                    )

                # brak mismatch => do normalnego folderu
                if not mismatch_tracker and not mismatch_pf:
                    save_stuff(raw_dir, annotated_dir, ann_yolo_dir, ann_tracker_dir, ann_pf_dir)

                # mismatch_tracker => do to_check/tracker
                if mismatch_tracker:
                    save_stuff(tctr_raw_dir, tctr_ann_dir, tctr_yolo_dir, tctr_tracker_dir, tctr_pf_dir)

                # mismatch_pf => do to_check/pf
                if mismatch_pf:
                    save_stuff(tcpf_raw_dir, tcpf_ann_dir, tcpf_yolo_dir, tcpf_tracker_dir, tcpf_pf_dir)

            cv2.imshow("YOLO + Tracker + PF (hard reset, triple update)", frame_disp)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
