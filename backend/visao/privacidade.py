"""Desfoca rostos e placas de veículos em frames BGR.

Não altera a detecção de placas de TRÂNSITO do projeto. Usa classificadores
Haar Cascades incluídos no OpenCV como solução inicial (não garante detecção
completa, especialmente para placas brasileiras pequenas ou à distância).
"""
from pathlib import Path
import cv2


class Anonimizador:
    def __init__(self):
        pasta = Path(cv2.data.haarcascades)
        self.rostos = cv2.CascadeClassifier(
            str(pasta / "haarcascade_frontalface_default.xml")
        )
        self.placas_veiculos = cv2.CascadeClassifier(
            str(pasta / "haarcascade_russian_plate_number.xml")
        )
        if self.rostos.empty() or self.placas_veiculos.empty():
            raise RuntimeError("Não foi possível carregar os classificadores do OpenCV")

    @staticmethod
    def _desfocar(frame, x, y, w, h, margem=0.12):
        altura, largura = frame.shape[:2]
        dx, dy = round(w * margem), round(h * margem)
        x1, y1 = max(0, x - dx), max(0, y - dy)
        x2, y2 = min(largura, x + w + dx), min(altura, y + h + dy)
        if x2 <= x1 or y2 <= y1:
            return
        regiao = frame[y1:y2, x1:x2]
        # Pixelização forte: reduz para poucos pixels e amplia com nearest-neighbor.
        # É mais difícil recuperar detalhes do que um desfoque gaussiano leve.
        rw, rh = regiao.shape[1], regiao.shape[0]
        pequeno = cv2.resize(regiao, (max(1, rw // 18), max(1, rh // 18)),
                             interpolation=cv2.INTER_AREA)
        regiao[:] = cv2.resize(pequeno, (rw, rh), interpolation=cv2.INTER_NEAREST)

    def aplicar(self, frame):
        """Altera o frame original no próprio lugar e retorna o frame."""
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cinza = cv2.equalizeHist(cinza)

        for x, y, w, h in self.rostos.detectMultiScale(
            cinza, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        ):
            self._desfocar(frame, int(x), int(y), int(w), int(h), margem=0.18)

        for x, y, w, h in self.placas_veiculos.detectMultiScale(
            cinza, scaleFactor=1.08, minNeighbors=3, minSize=(24, 8)
        ):
            self._desfocar(frame, int(x), int(y), int(w), int(h), margem=0.15)

        return frame
