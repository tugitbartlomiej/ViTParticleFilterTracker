from manim import *
class T(Scene):
    def construct(self):
        self.camera.background_color = "#0E1A24"
        a = Text("Zażółć gęślą jaźń 86,79% → 93,17%", font="Consolas", font_size=40, color="#EAF2F5")
        b = Text("Ćwiczenie: siła paczki, pęd, Łódź", font="Segoe UI", font_size=40, color="#2BB3C0").next_to(a, DOWN)
        c = MathTex(r"\Delta z(t+1) = P(t) + S(t)", color="#F08A2E").next_to(b, DOWN)
        self.add(a, b, c)
