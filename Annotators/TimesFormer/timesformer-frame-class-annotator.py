import os

import cv2


class FrameClassAnnotator:
    def __init__(self, video_path, output_dir="dataset_classes"):
        self.video_path = video_path
        self.output_dir = output_dir

        # Inicjalizacja wideo
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Nie można otworzyć pliku wideo!")

        # Parametry wideo
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

        # Utworzenie głównego katalogu wyjściowego
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Instrukcje
        self.instructions = [
            "Sterowanie:",
            "',' (przecinek) - poprzednia klatka",
            "'.' (kropka) - następna klatka",
            "'a'/'d' - skok o 40 klatek",
            "'+'/-' - skok o 20 klatek",
            "Klawisze 0-9 - przypisanie klasy i zapis klatki",
            "'q' - wyjscie",
            "Uzyj paska do przewijania filmu"
        ]

        # Utworzenie okna i paska przewijania
        cv2.namedWindow('Frame Annotator')
        cv2.createTrackbar('Frame', 'Frame Annotator', 0, self.total_frames - 1, self.on_trackbar)

    def on_trackbar(self, value):
        """Callback dla paska przewijania"""
        self.current_frame = value

    def save_frame(self, frame, class_idx):
        """Zapisanie klatki do folderu odpowiedniej klasy"""
        # Utworzenie folderu klasy jeśli nie istnieje
        class_dir = os.path.join(self.output_dir, str(class_idx))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            print(f"Utworzono folder dla klasy {class_idx}")

        frame_path = os.path.join(class_dir, f"frame_{self.current_frame:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        print(f"Zapisano klatkę {self.current_frame} do klasy {class_idx}")

    def show_info(self, frame):
        """Wyświetlenie informacji na klatce"""
        # Wyświetl numer klatki
        cv2.putText(frame, f"Frame: {self.current_frame}/{self.total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Wyświetl instrukcje
        y = 70
        for instruction in self.instructions:
            cv2.putText(frame, instruction, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20

        return frame

    def run(self):
        """Główna pętla programu"""
        while True:
            # Ustaw pozycję klatki
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()

            if not ret:
                print("Koniec wideo lub błąd odczytu klatki")
                break

            # Dodaj informacje na klatkę
            display_frame = self.show_info(frame.copy())

            # Wyświetl klatkę
            cv2.imshow("Frame Annotator", display_frame)

            # Aktualizuj pozycję paska
            cv2.setTrackbarPos('Frame', 'Frame Annotator', self.current_frame)

            # Obsługa klawiszy
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):  # Wyjście
                break
            elif key == ord(','):  # Poprzednia klatka
                self.current_frame = max(0, self.current_frame - 1)
            elif key == ord('.'):  # Następna klatka
                self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
            elif key == ord('a'):  # 40 klatek w tył
                self.current_frame = max(0, self.current_frame - 40)
            elif key == ord('d'):  # 40 klatek w przód
                self.current_frame = min(self.total_frames - 1, self.current_frame + 40)
            elif key == ord('+') or key == ord('='):  # 20 klatek w przód (+ lub = bez shifta)
                self.current_frame = min(self.total_frames - 1, self.current_frame + 20)
            elif key == ord('-'):  # 20 klatek w tył
                self.current_frame = max(0, self.current_frame - 20)
            elif key >= ord('0') and key <= ord('9'):  # Przypisanie klasy
                class_idx = key - ord('0')
                self.save_frame(frame, class_idx)
                # Przejdź do następnej klatki po zapisie
                self.current_frame = min(self.total_frames - 1, self.current_frame + 1)

        # Sprzątanie
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    video_path = 'E:/Cataract/videos/micro/train01.mp4'  # Ścieżka do twojego wideo
    output_dir = 'dataset_classes'  # Katalog wyjściowy

    annotator = FrameClassAnnotator(
        video_path=video_path,
        output_dir=output_dir
    )

    annotator.run()


if __name__ == "__main__":
    main()